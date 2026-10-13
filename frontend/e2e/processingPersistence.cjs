// Run against Vite using Playwright (NODE_PATH can point to a bundled runtime).
const { chromium } = require('playwright');
const assert = require('node:assert/strict');

(async () => {
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  try {
    for (const type of ['workflow', 'image', 'video']) {
      const context = await browser.newContext();
      const page = await context.newPage();
      const errors = [];
      page.on('pageerror', (error) => errors.push(error.message));
      const video = type === 'video';
      const pathname = video ? '/video-processing' : type === 'workflow' ? '/workflow' : '/processing';
      const filename = video ? 'cloud_chamber.mp4' : 'particle.png';
      const originalUrl = video ? 'http://localhost:8000/api/video/static/original.mp4' : 'http://localhost:8000/api/static/original.png';
      const file = { name: filename, size: 100, type: video ? 'video/mp4' : 'image/png', original_image_url: originalUrl };
      let terminal = 'PROCESSING';
      let offline = false;
      let missing = false;
      let posts = 0;
      let statusRequests = 0;
      const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=', 'base64');
      const result = video ? {
        job_id: 'abc123abc123', original_video_url: originalUrl, classified_video_url: originalUrl,
        video: { filename, duration_seconds: 5, original_fps: 30, target_fps: 30, original_frames: 150, output_frames: 150, width: 320, height: 240 },
        processing: { unique_source_frames_processed: 150, extracted_frames: 150, preprocessed_frames: 150, detected_frames: 150, classified_frames: 150, failed_frames: 0 },
        tracks: { alpha: 2, electron: 3, total_track_detections: 5 }, preview_frames: [], downloads: {},
      } : {
        job_id: 'abc123abc123', preprocessing: { total_images: 1, successful_count: 1, failed_count: 0, results: [{ status: 'success', original_filename: filename, processed_image_url: originalUrl }] },
        detection: type === 'workflow' ? { total_images: 1, successful_count: 1, failed_count: 0, results: [{ status: 'success', original_filename: filename, source_original_filename: filename, accepted_track_count: 0, tracks: [], outputs: {} }] } : null,
        classifications: {}, classificationSummary: { total_tracks: 0, alpha_count: 0, electron_count: 0 }, durationMs: 1000,
      };
      await context.route(/http:\/\/(localhost|127\.0\.0\.1):8000\/api\//, async (route) => {
        const url = route.request().url();
        if (url.includes('/static/')) return route.fulfill({ contentType: 'image/png', body: png });
        if (url.endsWith('/calibration/profiles')) return route.fulfill({ json: { profiles: [] } });
        if (route.request().method() === 'POST') {
          posts++;
          return route.fulfill({ status: 202, json: { job_id: 'abc123abc123' } });
        }
        if (url.includes('/status/')) {
          statusRequests++;
          if (offline) return route.abort('failed');
          if (missing) return route.fulfill({ status: 404, json: { detail: 'Job missing' } });
          return route.fulfill({ json: { job_id: 'abc123abc123', status: terminal,
            stage: terminal === 'FAILED' ? 'CLASSIFYING' : 'PREPROCESSING', progress_percent: 73,
            current_frame: 87, total_frames: 150, filename, size: 100, original_video_url: originalUrl,
            files: [file], calibrations: {}, error: terminal === 'FAILED' ? 'Test processing failure' : undefined } });
        }
        if (url.includes('/result/')) return route.fulfill({ json: result });
        return route.fulfill({ status: 404, json: {} });
      });
      await page.goto(`http://127.0.0.1:5174${pathname}`);
      await page.locator('input[type=file]').setInputFiles({ name: filename, mimeType: file.type, buffer: png });
      await page.getByRole('button', { name: video ? 'Process Video' : type === 'workflow' ? 'Process Images' : /Process All/ }).click();
      const key = `cloudtrack_active_${type}_job`;
      await page.waitForFunction((key) => localStorage.getItem(key) === 'abc123abc123', key).catch(async (error) => {
        console.error({ type, posts, errors, body: await page.locator('body').innerText() });
        throw error;
      });
      await page.getByText(/73%/).first().waitFor();
      assert.equal(posts, 1);
      await page.reload();
      await page.getByText(/73%/).first().waitFor();
      assert.equal(posts, 1, 'reload must not process again');
      assert.ok(await page.getByText(filename, { exact: false }).count());
      // Navigation unmounts the processing page without interrupting its job.
      await page.getByRole('link', { name: video ? 'Image Processing' : 'Video Processing' }).click();
      if (type === 'image') await page.goto(`http://127.0.0.1:5174${pathname}`);
      else await page.getByRole('link', { name: video ? 'Video Processing' : 'Image Processing' }).click();
      await page.getByText(/73%/).first().waitFor();
      // Focus refresh is independent of the normal polling delay.
      const before = statusRequests;
      await page.evaluate(() => window.dispatchEvent(new Event('focus')));
      await page.waitForFunction(() => true);
      await new Promise((resolve) => setTimeout(resolve, 150));
      assert.ok(statusRequests > before);
      offline = true;
      await page.evaluate(() => window.dispatchEvent(new Event('focus')));
      await page.getByText(/Connection interrupted/).first().waitFor();
      assert.equal(await page.evaluate((key) => localStorage.getItem(key), key), 'abc123abc123');
      offline = false;
      // Completion while another tab is active; visibility return refreshes it.
      const other = await context.newPage();
      await other.goto('about:blank');
      terminal = 'COMPLETED';
      await page.bringToFront();
      await page.evaluate(() => document.dispatchEvent(new Event('visibilitychange')));
      const complete = video ? 'Video Analysis Result' : type === 'workflow' ? 'Processing completed' : 'Processing Complete';
      await page.getByText(complete, { exact: true }).waitFor();
      await page.reload();
      await page.getByText(complete, { exact: true }).waitFor();
      assert.equal(posts, 1);
      assert.ok(await page.locator(`${video ? 'video' : 'img'}[src="${originalUrl}"]`).count());
      terminal = 'FAILED';
      await page.reload();
      await page.getByText(/Test processing failure/).first().waitFor();
      missing = true;
      await page.reload();
      await page.getByText(/Previous processing job could not be found/).first().waitFor();
      assert.equal(await page.evaluate((key) => localStorage.getItem(key), key), null);
      assert.equal(posts, 1);
      assert.deepEqual(errors, []);
      console.log(`${type}: upload, polling, remount, reload, focus, tab return, network recovery, completion, failure and 404 passed`);
      await context.close();
    }
  } finally { await browser.close(); }
})().catch((error) => { console.error(error); process.exitCode = 1; });
