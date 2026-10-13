import test, { afterEach } from 'node:test';
import assert from 'node:assert/strict';

import { apiService, Module3ApiError } from '../src/services/api.js';


const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});


test('classification client downloads one mask and posts multipart provenance', async () => {
  const calls = [];
  globalThis.fetch = async (url, options = {}) => {
    calls.push({ url, options });
    if (calls.length === 1) {
      return new Response(new Blob([new Uint8Array([137, 80, 78, 71])], { type: 'image/png' }));
    }
    assert.equal(options.method, 'POST');
    assert.ok(options.body instanceof FormData);
    assert.equal(options.body.get('track_id'), 'track_007');
    assert.equal(options.body.get('component_id'), '17');
    assert.equal(options.body.get('bbox_width'), '96');
    assert.equal(options.body.get('mm_per_pixel'), '0.2');
    assert.equal(options.body.get('roi_x0'), '41');
    assert.equal(options.body.get('roi_y0'), '72');
    assert.equal(options.body.get('file').name, 'track_007_mask.png');
    return Response.json({ processing_status: 'success', predicted_class: 'Alpha' });
  };

  const result = await apiService.classifyTrackMask('http://backend/mask.png', {
    track_id: 'track_007',
    component_id: 17,
    bbox_width: 96,
    mm_per_pixel: 0.2,
    roi_x0: 41,
    roi_y0: 72,
  });
  assert.equal(result.predicted_class, 'Alpha');
  assert.equal(calls.length, 2);
  assert.match(calls[1].url, /\/classification\/classify$/);
});

test('calibration lookup uses the exact image filename and treats 404 as optional', async () => {
  const calls = [];
  globalThis.fetch = async (url) => {
    calls.push(url);
    if (calls.length === 1) {
      return Response.json({
        image_filename: 'frame [1].png',
        pixel_distance: 250,
        real_world_distance_mm: 50,
        mm_per_pixel: 0.2,
        pixels_per_mm: 5,
      });
    }
    return new Response('', { status: 404 });
  };

  const calibrated = await apiService.getCalibration('frame [1].png');
  const uncalibrated = await apiService.getCalibration('other.png');
  assert.equal(calibrated.mm_per_pixel, 0.2);
  assert.equal(uncalibrated, null);
  assert.match(calls[0], /image_filename=frame\+%5B1%5D\.png/);
});

test('saved camera and chamber calibration profiles can be loaded', async () => {
  globalThis.fetch = async () => Response.json({
    profiles: [{ profile_id: 'setup-a', profile_name: 'Setup A', mm_per_pixel: 0.1 }],
  });
  const profiles = await apiService.getCalibrationProfiles();
  assert.equal(profiles.length, 1);
  assert.equal(profiles[0].profile_name, 'Setup A');
});

test('four-point calibration is saved as measurement metadata without an image upload', async () => {
  globalThis.fetch = async (url, options) => {
    assert.match(url, /\/calibration\/calculate$/);
    assert.equal(options.method, 'POST');
    assert.equal(options.headers['Content-Type'], 'application/json');
    const payload = JSON.parse(options.body);
    assert.equal(payload.point_4_x, 10);
    assert.equal(payload.chamber_width_mm, 300);
    assert.equal('rectification_enabled' in payload, false);
    return Response.json({ calibration_type: 'four_point_measurement', profile_id: 'setup-a' });
  };
  const result = await apiService.calculateCalibration({
    point_1_x: 10, point_1_y: 10, point_2_x: 100, point_2_y: 10,
    point_3_x: 100, point_3_y: 200, point_4_x: 10, point_4_y: 200,
    chamber_width_mm: 300, chamber_height_mm: 500,
    profile_name: 'Setup A', source_image_width: 1080, source_image_height: 1920,
  });
  assert.equal(result.calibration_type, 'four_point_measurement');
});

test('structured backend classification errors remain structured', async () => {
  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    if (calls === 1) return new Response(new Blob(['mask'], { type: 'image/png' }));
    return Response.json(
      { processing_status: 'failed', error_code: 'no_foreground', error_message: 'No foreground track.' },
      { status: 422 },
    );
  };

  await assert.rejects(
    apiService.classifyTrackMask('http://backend/mask.png', { track_id: 'track_008' }),
    (error) => error instanceof Module3ApiError
      && error.errorCode === 'no_foreground'
      && error.status === 422
      && error.message === 'No foreground track.',
  );
});

test('no classification request is invented without a Module 2 mask URL', async () => {
  let called = false;
  globalThis.fetch = async () => { called = true; };
  await assert.rejects(
    apiService.classifyTrackMask('', { track_id: 'missing' }),
    (error) => error.errorCode === 'missing_module2_mask',
  );
  assert.equal(called, false);
});

test('Module 1 output is handed to Module 2 as an unchanged downloaded file', async () => {
  const bytes = new Uint8Array([137, 80, 78, 71, 13, 10]);
  globalThis.fetch = async (url) => {
    assert.equal(url, 'http://backend/processed.png');
    return new Response(new Blob([bytes], { type: 'image/png' }));
  };

  const file = await apiService.imageUrlToFile(
    'http://backend/processed.png',
    'module1_sample_1.png',
  );
  assert.equal(file.name, 'module1_sample_1.png');
  assert.equal(file.type, 'image/png');
  assert.deepEqual(new Uint8Array(await file.arrayBuffer()), bytes);
});

test('Module 1 handoff reports a missing backend output', async () => {
  globalThis.fetch = async () => new Response('', { status: 404 });
  await assert.rejects(
    apiService.imageUrlToFile('http://backend/missing.png'),
    /preprocessed image is unavailable \(status 404\)/i,
  );
});

test('batch client sends multiple masks through one ordered Module 3 batch request', async () => {
  const calls = [];
  globalThis.fetch = async (url, options = {}) => {
    calls.push({ url, options });
    if (!options.method) {
      return new Response(new Blob([new Uint8Array([137, 80, 78, 71])], { type: 'image/png' }));
    }
    assert.match(url, /\/classification\/classify-batch$/);
    assert.equal(options.method, 'POST');
    assert.ok(options.body instanceof FormData);
    assert.equal(options.body.getAll('files').length, 2);
    const metadata = JSON.parse(options.body.get('provenance_json'));
    assert.deepEqual(metadata.map(({ request_id }) => request_id), ['run:track_1', 'run:track_2']);
    assert.deepEqual(metadata.map(({ mm_per_pixel }) => mm_per_pixel), [0.2, undefined]);
    assert.deepEqual(metadata[0].measurement_homography, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    return Response.json({
      processing_status: 'SUCCESS',
      summary: {
        total_tracks: 2,
        successful_tracks: 2,
        failed_tracks: 0,
        alpha_count: 1,
        electron_count: 1,
        success_rate: 1,
      },
      results: [
        { processing_status: 'success', predicted_class: 'Alpha', provenance: { track_id: 'track_1' } },
        { processing_status: 'success', predicted_class: 'Electron', provenance: { track_id: 'track_2' } },
      ],
    });
  };

  const batch = await apiService.classifyTrackMasks([
    { maskUrl: 'http://backend/one.png', provenance: { request_id: 'run:track_1', track_id: 'track_1', mm_per_pixel: 0.2, measurement_homography: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] } },
    { maskUrl: 'http://backend/two.png', provenance: { request_id: 'run:track_2', track_id: 'track_2' } },
  ]);
  assert.equal(calls.length, 3);
  assert.deepEqual(batch.results.map(({ predicted_class }) => predicted_class), ['Alpha', 'Electron']);
  assert.equal(batch.summary.successful_tracks, 2);
  assert.equal(batch.summary.alpha_count, 1);
  assert.equal(batch.summary.electron_count, 1);
});

test('batch client preserves a mask-download failure without dropping valid siblings', async () => {
  globalThis.fetch = async (url, options = {}) => {
    if (url.endsWith('/missing.png')) return new Response('', { status: 404 });
    if (!options.method) return new Response(new Blob(['mask'], { type: 'image/png' }));
    return Response.json({
      processing_status: 'SUCCESS',
      summary: {
        total_tracks: 1,
        successful_tracks: 1,
        failed_tracks: 0,
        alpha_count: 1,
        electron_count: 0,
        success_rate: 1,
      },
      results: [{ processing_status: 'success', predicted_class: 'Alpha', provenance: { track_id: 'valid' } }],
    });
  };

  const batch = await apiService.classifyTrackMasks([
    { maskUrl: 'http://backend/valid.png', provenance: { track_id: 'valid' } },
    { maskUrl: 'http://backend/missing.png', provenance: { track_id: 'missing' } },
  ]);
  assert.equal(batch.processing_status, 'PARTIAL_SUCCESS');
  assert.equal(batch.results[0].predicted_class, 'Alpha');
  assert.equal(batch.results[1].error_code, 'mask_fetch_failed');
  assert.equal(batch.summary.successful_tracks, 1);
  assert.equal(batch.summary.failed_tracks, 1);
});
