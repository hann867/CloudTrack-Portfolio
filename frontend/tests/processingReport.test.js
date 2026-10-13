import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildProcessingReportModel,
  createProcessingReportDocument,
} from '../src/reports/processingReport.js';

const pixel = {
  dataUrl: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
  width: 1200,
  height: 800,
};

const successfulClassification = (predictedClass, confidence, alpha, electron) => ({
  status: 'success',
  result: {
    predicted_class: predictedClass,
    confidence,
    probabilities: { Alpha: alpha, Electron: electron },
    features: {
      track_length_px: 22.24,
      mean_width_px: 3.42,
      median_width_px: 4,
      max_width_px: 5.66,
      mean_curvature: 0.033,
      max_curvature: 0.1063,
      tortuosity: 1.049,
    },
    physical_features: {
      track_length_mm: 2.224,
      mean_width_mm: 0.342,
      median_width_mm: 0.4,
      max_width_mm: 0.566,
    },
    model: { model_version: 'module3_rf_development_v2', model_stage: 'development' },
  },
});

function fixture() {
  const longName = `${'long-scientific-source-name-'.repeat(5)}alpha.png`;
  const selectedFiles = [{ name: longName }, { name: 'electron.png' }];
  const preprocessing = {
    results: selectedFiles.map((file, index) => ({
      original_filename: file.name,
      status: 'success',
      processed_image_url: `processed-${index}`,
    })),
  };
  const detection = {
    results: selectedFiles.map((file, index) => ({
      source_original_filename: file.name,
      original_filename: `module1-${index}.png`,
      status: 'success',
      run_id: `${'run-identifier-'.repeat(5)}${index}`,
      raw_component_count: 4,
      trajectory_group_count: 2,
      accepted_track_count: 1,
      outputs: {
        annotated_image: `annotated-${index}`,
        binary_mask: `binary-${index}`,
        cleaned_mask: `cleaned-${index}`,
        labelled_mask: `labelled-${index}`,
      },
      tracks: [{
        label_id: index + 1,
        track_id: `track-${index}`,
        bbox_x: 5,
        bbox_y: 8,
        bbox_width: 30,
        bbox_height: 20,
        area_pixels: 200,
        roi_image_url: `roi-${index}`,
        roi_mask_url: `mask-${index}`,
      }],
    })),
  };
  const classifications = {
    [`${detection.results[0].run_id}:track-0`]: successfulClassification('Alpha', 0.96, 0.96, 0.04),
    [`${detection.results[1].run_id}:track-1`]: successfulClassification('Electron', 1, 0, 1),
  };
  return {
    selectedFiles,
    selectedImageName: longName,
    previews: { [longName]: 'raw-0', 'electron.png': 'raw-1' },
    preprocessing,
    detection,
    classifications,
    calibrations: {
      [longName]: { status: 'calibrated', data: {
        profile_name: 'Cloud Chamber Setup A',
        calibration_type: 'four_point_measurement',
        chamber_width_mm: 120,
        chamber_height_mm: 80,
        horizontal_mm_per_pixel: 0.1,
        vertical_mm_per_pixel: 0.102,
        mm_per_pixel: 0.101,
        selected_corners: { A: { x: 0, y: 0 }, B: { x: 1200, y: 0 }, C: { x: 1200, y: 784 }, D: { x: 0, y: 784 } },
      } },
      'electron.png': { status: 'uncalibrated' },
    },
    durationMs: 2345,
    processingStartedAt: Date.parse('2026-09-03T10:00:00Z'),
    generatedAt: new Date('2026-09-03T10:01:00Z'),
  };
}

test('report model keeps multiple-image tracks, classes, and calibration associated', () => {
  const model = buildProcessingReportModel(fixture());
  assert.equal(model.images.length, 2);
  assert.equal(model.detectedTracks, 2);
  assert.equal(model.alphaCount, 1);
  assert.equal(model.electronCount, 1);
  assert.equal(model.images[0].tracks[0].view.title, 'Alpha');
  assert.equal(model.images[0].calibration.status, 'calibrated');
  assert.equal(model.images[1].tracks[0].view.title, 'Electron');
  assert.equal(model.images[1].calibration.status, 'uncalibrated');
  assert.match(model.images[0].tracks[0].view.features[0].physicalValue, /mm$/);
  assert.equal(model.images[0].tracks[0].view.features.at(-1).physicalValue, '');
});

test('structured report renders a readable multi-page PDF for long identifiers', async () => {
  const model = buildProcessingReportModel(fixture());
  const assets = new Map();
  for (const image of model.images) {
    [
      image.rawUrl,
      image.module1.processed_image_url,
      ...Object.values(image.module2.outputs),
      ...image.tracks.flatMap((track) => [track.roi_image_url, track.roi_mask_url]),
    ].forEach((url) => assets.set(url, pixel));
    assets.set(`classified:${image.filename}`, pixel);
  }
  const document = await createProcessingReportDocument(model, assets);
  const bytes = new Uint8Array(document.output('arraybuffer'));
  assert.ok(document.getNumberOfPages() >= 14);
  assert.equal(String.fromCharCode(...bytes.slice(0, 5)), '%PDF-');
  assert.ok(bytes.length > 10_000);
});
