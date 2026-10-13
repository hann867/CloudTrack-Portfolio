# Processing job persistence

## Cause and scope

Video jobs already ran in FastAPI background workers and saved `status.json`,
`metadata.json`, and `results.json`. The video page only held the job ID inside
its upload/poll function. Route unmounts and reloads lost it, and a network error
ended observation. There was no focus/visibility refresh handler.

The sidebar's Image Processing page is `/workflow` (`ProcessingWorkflow.jsx`),
not the older `/processing` page (`ImageProcessing.jsx`). Both stored results
only in React state. The guided workflow also orchestrated the Module 1 → 2 → 3
handoff in the browser, so backend-independent completion required moving that
coordination into an image background job. The underlying preprocessing,
detection, and classification algorithms and existing module endpoints are unchanged.

No code was found that deliberately resets the page on a tab switch. A normal
tab switch does not remount React; timer throttling, request failures, route
unmounts, and page/Vite reloads were the recovery gaps.

## Behaviour

- `/api/image/process` saves uploads and metadata, queues the existing modules,
  and returns HTTP 202 with a job ID. `workflow=true` runs all three modules;
  otherwise it runs Module 1. The existing atomic status writer is reused.
- `/api/image/status/{id}` and `/api/image/result/{id}` restore file metadata,
  progress, failures, and results. Originals and generated images use the
  existing `/api/static/jobs/{id}/...` mount. Video status now includes original
  media URL, filename, type, and size (when recorded).
- `jobSession.js` and `useProcessingJob.js` share one observer per page type.
  Storage keys are `cloudtrack_active_workflow_job`,
  `cloudtrack_active_image_job`, and `cloudtrack_active_video_job`. Each stores
  only a job ID; results and files never go into localStorage.
- Mount/reload, window focus, and becoming visible trigger status GETs. Normal
  observation uses one 1.5-second timeout after each response, with one in-flight
  request. Status/result reads time out after 15 seconds and retry transient
  failures. Unmount aborts only observation, never the backend worker.
- Completed jobs retain their ID. Failed jobs retain their error/stage and, for
  images, available partial results. Confirmed missing job directories return
  404 and clear the reference. Unreadable job files return retryable errors.
- Explicit new file selection/start-over clears the old reference; starting a
  job saves its new ID. Generation checks and aborted observers prevent stale
  responses from replacing newer jobs. Upload responses still persist when a
  route unmounts while the request is in flight.
- The guided workflow's explicit **Retry analysis** action reruns the analysis
  using its saved original files. Restoration itself only reads status/results.

Restart FastAPI to register the new image routes if the server does not auto-reload.

## Validation

Run from the repository root:

```powershell
.venv/Scripts/python.exe -m unittest backend.tests.test_image_jobs backend.tests.video_processing.test_video_api -q
cd frontend
npm test
npm run lint
npm run build
```

The browser regression scripts require Playwright and Chrome. Use an existing
Playwright installation (or set `NODE_PATH` to a runtime containing it). Start
Vite on `http://127.0.0.1:5174` with `npm run dev -- --host 127.0.0.1 --port 5174 --strictPort`.

```powershell
node frontend/e2e/processingPersistence.cjs
# For real processing, start this isolated backend in another terminal:
.venv/Scripts/python.exe -m backend.tests.browser_job_server
node frontend/e2e/realProcessingPersistence.cjs
```

The isolated server listens on port 8011; the real browser script forwards
frontend API requests there, leaving a development backend on port 8000 alone.
Its files are temporary and removed on graceful shutdown.

Both browser scripts cover all three processing pages. Controlled responses
cover focus/visibility events, route changes, reloads, completion, failure, and
network/404 recovery. Real backend tests cover image preprocessing, the full
image workflow, and a five-second/150-frame video, completion while the page is
unmounted, exact saved-original bytes, completed-result reload, and exactly one
processing POST. Unit tests additionally cover late responses, StrictMode-style
remounts, timer cleanup, calibrated mask handoff, and partial failures.

## Limits

- Recovery starts once the upload response supplies a job ID. A full browser
  reload before that response, or before uploading, cannot restore the browser
  File object. Navigating routes while the upload request continues is supported.
- Browser storage must be available, and the backend output folders must remain.
  References are per browser origin/profile, not a cross-device job history.
- Background workers continue without the frontend, but are not a durable queue
  that resumes interrupted work after a backend process crash/restart. Saved
  completed jobs remain queryable. Restarting a worker mid-job can leave its last
  processing snapshot; automatic crash recovery is outside this change.
- Original media is served byte-for-byte. Browser playback still depends on its
  supported codecs/formats; this change does not transcode original AVI/MKV or TIFF.
- OS application/window switching was represented by focus events in headless
  Chrome; manual desktop Alt-Tab testing was not performed.
