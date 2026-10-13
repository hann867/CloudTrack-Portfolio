// Only the backend job identifier is persisted; all progress comes from the API.
export function createJobSession({ key, startJob, getStatus, getResult, storage,
  windowTarget = window, documentTarget = document, pollMs = 1500, getFailedResult = false }) {
  let savedId = null;
  try { storage ??= windowTarget.localStorage; savedId = storage.getItem(key); } catch { /* Storage may be disabled. */ }
  let state = { jobId: savedId, status: null, result: null, error: '', busy: !!savedId };
  let generation = 0;
  let timer;
  let controller;
  let connections = 0;
  const listeners = new Set();
  const publish = (next) => { state = { ...state, ...next }; listeners.forEach((fn) => fn()); };
  const persist = (id) => {
    try { if (id) storage.setItem(key, id); else storage.removeItem(key); }
    catch { publish({ error: 'Browser storage is unavailable; this job cannot survive a page reload.' }); }
  };
  const stop = () => {
    clearTimeout(timer);
    controller?.abort();
    controller = null;
  };
  const refresh = async () => {
    if (!connections || !state.jobId || controller) return;
    clearTimeout(timer);
    const version = generation;
    const jobId = state.jobId;
    const request = new AbortController();
    controller = request;
    const current = () => version === generation && !request.signal.aborted;
    let retry = true;
    try {
      const status = await getStatus(jobId, request.signal);
      if (!current()) return;
      publish({ status, error: '', busy: status.status !== 'FAILED' && !state.result });
      if (status.status === 'FAILED') {
        retry = false;
        publish({ busy: false, error: `Stage: ${status.stage}. ${status.error || status.message || 'Processing failed.'} (Job ${jobId})` });
        if (getFailedResult && !state.result) {
          const result = await getResult(jobId, request.signal);
          if (current()) publish({ result });
        }
      } else if (status.status === 'COMPLETED') {
        const result = state.result || await getResult(jobId, request.signal);
        if (!current()) return;
        retry = false;
        publish({ result, busy: false });
      }
    } catch (error) {
      if (!current()) return;
      // A missing result is not proof that the job itself has disappeared.
      if (error.status === 404 && error.jobMissing) {
        retry = false;
        persist(null);
        publish({ jobId: null, status: null, result: null, busy: false, error: 'Previous processing job could not be found.' });
      } else {
        publish({ error: retry ? `Connection interrupted; retrying. ${error.message}`
          : `${state.error} Partial results unavailable: ${error.message}` });
      }
    } finally {
      if (controller === request) controller = null;
      if (current() && connections && retry) timer = setTimeout(refresh, pollMs);
    }
  };
  const visible = () => { if (documentTarget.visibilityState === 'visible') void refresh(); };
  return {
    getSnapshot: () => state,
    subscribe: (listener) => { listeners.add(listener); return () => listeners.delete(listener); },
    connect() {
      if (++connections === 1) {
        windowTarget.addEventListener('focus', refresh);
        documentTarget.addEventListener('visibilitychange', visible);
        void refresh();
      }
      return () => {
        if (--connections === 0) {
          stop();
          windowTarget.removeEventListener('focus', refresh);
          documentTarget.removeEventListener('visibilitychange', visible);
        }
      };
    },
    clear() {
      generation += 1;
      stop();
      persist(null);
      publish({ jobId: null, status: null, result: null, error: '', busy: false });
    },
    async start(...args) {
      if (state.busy) return;
      const version = ++generation;
      stop();
      publish({ jobId: null, status: null, result: null, error: '', busy: true });
      try {
        const started = await startJob(...args);
        if (version !== generation) return;
        publish({ jobId: started.job_id });
        persist(started.job_id);
        // Upload completion is retained even when the route has unmounted.
        void refresh();
      } catch (error) {
        if (version === generation) publish({ busy: false, error: error.message });
      }
    },
  };
}
