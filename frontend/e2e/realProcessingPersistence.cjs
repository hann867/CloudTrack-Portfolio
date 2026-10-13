// Requires Vite on 5174 and: python -m backend.tests.browser_job_server
const { chromium } = require('playwright');
const assert = require('node:assert/strict');

(async () => {
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  try {
    for (const type of ['workflow', 'image', 'video']) {
      const context = await browser.newContext();
      const page = await context.newPage();
      // Forward API requests to the isolated test server, preserving real HTTP processing.
      await context.route(/http:\/\/(localhost|127\.0\.0\.1):8000\/api\//, async (route) => {
        const response = await route.fetch({ url: route.request().url().replace(':8000/', ':8011/') });
        await route.fulfill({ response });
      });
      const errors = [];
      page.on('pageerror', (error) => errors.push(error.message));
      let posts = 0;
      page.on('request', (request) => { if (request.method() === 'POST' && request.url().endsWith('/process')) posts++; });
      const video = type === 'video';
      const path = video ? '/video-processing' : type === 'workflow' ? '/workflow' : '/processing';
      const filename = video ? 'chamber.avi' : 'particle.png';
      const fixture = await context.request.get(`http://127.0.0.1:8011/fixtures/${filename}`);
      await page.goto(`http://127.0.0.1:5174${path}`);
      await page.locator('input[type=file]').setInputFiles({ name: filename, mimeType: video ? 'video/x-msvideo' : 'image/png', buffer: await fixture.body() });
      await page.getByRole('button', { name: video ? 'Process Video' : type === 'workflow' ? 'Process Images' : /Process All/ }).click();
      const key = `cloudtrack_active_${type}_job`;
      await page.waitForFunction((key) => !!localStorage.getItem(key), key);
      const jobId = await page.evaluate((key) => localStorage.getItem(key), key);
      await page.reload();
      await page.getByText(filename, { exact: false }).first().waitFor();
      await page.getByRole('link', { name: video ? 'Image Processing' : 'Video Processing' }).click();
      // The original page is unmounted. Only observe the backend until completion.
      let status;
      const deadline = Date.now() + 120000;
      do {
        const response = await context.request.get(`http://127.0.0.1:8011/api/${video ? 'video' : 'image'}/status/${jobId}`);
        status = await response.json();
        if (status.status === 'COMPLETED' || status.status === 'FAILED') break;
        await new Promise((resolve) => setTimeout(resolve, 500));
      } while (Date.now() < deadline);
      assert.equal(status.status, 'COMPLETED', JSON.stringify(status));
      await page.goto(`http://127.0.0.1:5174${path}`);
      const complete = video ? 'Video Analysis Result' : type === 'workflow' ? 'Processing completed' : 'Processing Complete';
      await page.getByText(complete, { exact: true }).waitFor();
      await page.reload();
      await page.getByText(complete, { exact: true }).waitFor();
      const original = video ? status.original_video_url : status.files[0].original_image_url;
      const saved = await context.request.get(original);
      assert.equal(saved.status(), 200);
      assert.deepEqual(await saved.body(), await fixture.body());
      assert.equal(posts, 1, 'restoration cannot create a second job');
      assert.deepEqual(errors, []);
      if (video) {
        const response = await context.request.get(`http://127.0.0.1:8011/api/video/result/${jobId}`);
        const result = await response.json();
        assert.equal(result.video.output_frames, 150);
        assert.equal((await context.request.get(result.classified_video_url)).status(), 200);
      }
      console.log(`${type}: real processing completed while unmounted; reload, result and original restored; one POST`);
      await context.close();
    }
  } finally { await browser.close(); }
})().catch((error) => { console.error(error); process.exitCode = 1; });
