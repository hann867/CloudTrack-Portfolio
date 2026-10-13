import test from 'node:test';
import assert from 'node:assert/strict';
import { createJobSession } from '../src/services/jobSession.js';

const flush = () => new Promise((resolve) => setImmediate(resolve));
const deferred = () => { let resolve; const promise = new Promise((done) => { resolve = done; }); return { promise, resolve }; };
function fixture(options = {}) {
  const values = new Map([['job', 'saved']]);
  const storage = { getItem: (key) => values.get(key) || null, setItem: (key, value) => values.set(key, value), removeItem: (key) => values.delete(key) };
  const windowTarget = new EventTarget();
  const documentTarget = new EventTarget();
  documentTarget.visibilityState = 'visible';
  const calls = { status: [], result: [], start: 0 };
  const session = createJobSession({ key: 'job', storage, windowTarget, documentTarget, pollMs: 10,
    startJob: async () => { calls.start++; return { job_id: 'new' }; },
    getStatus: async (id) => { calls.status.push(id); return { status: 'COMPLETED' }; },
    getResult: async (id) => { calls.result.push(id); return { id }; }, ...options });
  return { session, storage, calls, windowTarget, documentTarget };
}

test('mount and page reload restore completed result with GETs and retain the ID', async () => {
  const f = fixture();
  const disconnect = f.session.connect();
  await flush();
  assert.deepEqual(f.session.getSnapshot().result, { id: 'saved' });
  assert.equal(f.calls.start, 0);
  assert.equal(f.storage.getItem('job'), 'saved');
  disconnect();
  const reloaded = fixture({ storage: f.storage });
  const close = reloaded.session.connect();
  await flush();
  assert.equal(reloaded.session.getSnapshot().result.id, 'saved');
  close();
});

test('normal polling runs without overlap, stops at completion and cleans listeners', async () => {
  let count = 0;
  const f = fixture({ getStatus: async () => ({ status: ++count < 3 ? 'PROCESSING' : 'COMPLETED', progress_percent: count * 30 }) });
  const close = f.session.connect();
  await new Promise((resolve) => setTimeout(resolve, 65));
  assert.equal(count, 3);
  assert.equal(f.session.getSnapshot().busy, false);
  close();
  f.windowTarget.dispatchEvent(new Event('focus'));
  f.documentTarget.dispatchEvent(new Event('visibilitychange'));
  await flush();
  assert.equal(count, 3);
});

test('visibility and focus immediately refresh, sharing a single in-flight request', async () => {
  let count = 0;
  let next = deferred();
  const f = fixture({ pollMs: 10000, getStatus: () => { count++; return next.promise; } });
  const close = f.session.connect();
  f.windowTarget.dispatchEvent(new Event('focus'));
  f.documentTarget.dispatchEvent(new Event('visibilitychange'));
  assert.equal(count, 1);
  next.resolve({ status: 'PROCESSING', progress_percent: 12 });
  await flush();
  next = deferred();
  f.documentTarget.visibilityState = 'hidden';
  f.documentTarget.dispatchEvent(new Event('visibilitychange'));
  assert.equal(count, 1);
  f.documentTarget.visibilityState = 'visible';
  f.documentTarget.dispatchEvent(new Event('visibilitychange'));
  f.windowTarget.dispatchEvent(new Event('focus'));
  assert.equal(count, 2);
  next.resolve({ status: 'COMPLETED' });
  await flush();
  assert.equal(f.session.getSnapshot().result.id, 'saved');
  close();
});

test('network errors retain identity and polling recovers', async () => {
  let count = 0;
  const f = fixture({ getStatus: async () => { if (++count === 1) throw new Error('offline'); return { status: 'COMPLETED' }; } });
  const close = f.session.connect();
  await flush();
  assert.equal(f.storage.getItem('job'), 'saved');
  assert.match(f.session.getSnapshot().error, /retrying/);
  await new Promise((resolve) => setTimeout(resolve, 35));
  assert.equal(f.session.getSnapshot().result.id, 'saved');
  assert.equal(f.session.getSnapshot().error, '');
  close();
});

test('only a confirmed missing job clears persistence; a missing result does not', async () => {
  for (const jobMissing of [true, false]) {
    const f = fixture({ getStatus: async () => { throw Object.assign(new Error('missing'), { status: 404, jobMissing }); } });
    const close = f.session.connect();
    await flush();
    assert.equal(f.storage.getItem('job'), jobMissing ? null : 'saved');
    close();
  }
});

test('failed jobs restore their stage and error and stop polling', async () => {
  let count = 0;
  const f = fixture({ getFailedResult: true, getStatus: async () => { count++; return { status: 'FAILED', stage: 'CLASSIFYING', error: 'Missing model' }; } });
  const close = f.session.connect();
  await new Promise((resolve) => setTimeout(resolve, 35));
  assert.equal(count, 1);
  assert.equal(f.session.getSnapshot().busy, false);
  assert.match(f.session.getSnapshot().error, /CLASSIFYING.*Missing model/);
  assert.equal(f.session.getSnapshot().result.id, 'saved');
  assert.equal(f.storage.getItem('job'), 'saved');
  close();
});

test('unmount/StrictMode remount ignores the old response and reconnects', async () => {
  const old = deferred();
  let count = 0;
  const f = fixture({ getStatus: () => ++count === 1 ? old.promise : Promise.resolve({ status: 'COMPLETED' }) });
  const close = f.session.connect();
  close();
  const closeAgain = f.session.connect();
  await flush();
  old.resolve({ status: 'FAILED', error: 'stale' });
  await flush();
  assert.equal(f.session.getSnapshot().status.status, 'COMPLETED');
  closeAgain();
});

test('a newer job cannot be replaced by a stale status or result response', async () => {
  const stale = deferred();
  const f = fixture({ getResult: (id) => id === 'saved' ? stale.promise : Promise.resolve({ id }) });
  const close = f.session.connect();
  await flush();
  f.session.clear();
  await f.session.start();
  await flush();
  stale.resolve({ id: 'saved' });
  await flush();
  assert.equal(f.session.getSnapshot().result.id, 'new');
  assert.equal(f.storage.getItem('job'), 'new');
  close();
});

test('upload completion persists after navigation and never starts twice', async () => {
  const uploaded = deferred();
  let starts = 0;
  const f = fixture({ startJob: () => { starts++; return uploaded.promise; } });
  f.session.clear();
  const close = f.session.connect();
  const starting = f.session.start();
  await f.session.start();
  close();
  uploaded.resolve({ job_id: 'uploaded' });
  await starting;
  assert.equal(f.storage.getItem('job'), 'uploaded');
  assert.equal(starts, 1);
  const reconnect = f.session.connect();
  await flush();
  assert.equal(f.session.getSnapshot().result.id, 'uploaded');
  reconnect();
});
