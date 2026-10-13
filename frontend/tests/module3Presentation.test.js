import test from 'node:test';
import assert from 'node:assert/strict';

import {
  classificationView,
  featureRows,
  MODULE3_FEATURES,
} from '../src/pages/module3Presentation.js';


const features = Object.fromEntries(MODULE3_FEATURES.map(({ key }, index) => [key, index + 1.125]));

const successState = (predictedClass) => ({
  status: 'success',
  result: {
    predicted_class: predictedClass,
    confidence: predictedClass === 'Alpha' ? 0.873 : 0.641,
    probabilities: { Alpha: 0.873, Electron: 0.127 },
    features,
    model: { model_version: 'module3_rf_development_v2', model_stage: 'development' },
  },
});


test('successful Alpha response exposes Alpha, confidence, and both probabilities', () => {
  const view = classificationView(successState('Alpha'));
  assert.equal(view.kind, 'success');
  assert.equal(view.title, 'Alpha');
  assert.equal(view.confidence, '87.3%');
  assert.equal(view.alpha, '87.3%');
  assert.equal(view.electron, '12.7%');
});

test('successful Electron response exposes Electron', () => {
  const state = successState('Electron');
  state.result.probabilities = { Alpha: 0.359, Electron: 0.641 };
  const view = classificationView(state);
  assert.equal(view.title, 'Electron');
  assert.equal(view.confidence, '64.1%');
  assert.equal(view.electron, '64.1%');
});

test('development model stage and Development RF v2 label are exposed', () => {
  const view = classificationView(successState('Alpha'));
  assert.equal(view.modelStage, 'development');
  assert.equal(view.modelLabel, 'Development RF v2');
});

test('all seven features are available with pixel units only where appropriate', () => {
  const rows = featureRows(features);
  assert.equal(rows.length, 7);
  assert.deepEqual(rows.map(({ key }) => key), MODULE3_FEATURES.map(({ key }) => key));
  assert.match(rows.find(({ key }) => key === 'track_length_px').value, / px$/);
  assert.doesNotMatch(rows.find(({ key }) => key === 'tortuosity').value, /px|mm/);
});

test('calibrated reporting adds physical values without replacing pixel values', () => {
  const rows = featureRows(features, {
    track_length_mm: 20,
    mean_width_mm: 0.4,
    median_width_mm: 0.6,
    max_width_mm: 0.8,
  });
  const length = rows.find(({ key }) => key === 'track_length_px');
  const tortuosity = rows.find(({ key }) => key === 'tortuosity');
  assert.match(length.value, / px$/);
  assert.equal(length.physicalValue, '20.00 mm');
  assert.equal(tortuosity.physicalValue, '');
});

test('loading, failure, and missing-mask states remain distinct', () => {
  assert.equal(classificationView({ status: 'loading' }).title, 'Classifying…');
  assert.equal(classificationView({ status: 'error', message: 'Invalid track geometry' }).title, 'Module 3 classification failed');
  const unavailable = classificationView({ status: 'unavailable' });
  assert.equal(unavailable.title, 'Module 3 classification unavailable');
  assert.equal(unavailable.message, 'No valid Module 2 mask');
  assert.equal(unavailable.confidence, undefined);
});

test('one failed track does not alter a successful sibling result', () => {
  const views = [
    classificationView(successState('Alpha')),
    classificationView({ status: 'error', message: 'Invalid track geometry' }),
  ];
  assert.equal(views[0].kind, 'success');
  assert.equal(views[0].title, 'Alpha');
  assert.equal(views[1].kind, 'error');
});
