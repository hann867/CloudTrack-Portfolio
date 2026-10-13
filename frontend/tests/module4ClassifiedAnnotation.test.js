import test from 'node:test';
import assert from 'node:assert/strict';

import {
  MODULE4_CLASS_COLOURS,
  classifiedAnnotation,
} from '../src/pages/module4ClassifiedAnnotation.js';

test('Module 4 maps Alpha to purple and preserves the classified label values', () => {
  const annotation = classifiedAnnotation(
    { label_id: 1 },
    { kind: 'success', title: 'Alpha', confidence: '96.0%' },
  );

  assert.deepEqual(annotation, {
    colour: MODULE4_CLASS_COLOURS.Alpha,
    text: 'Track 1 · Alpha 96.0%',
  });
  assert.equal(annotation.colour, '#c084fc');
});

test('Module 4 maps Electron to blue', () => {
  const annotation = classifiedAnnotation(
    { label_id: 2 },
    { kind: 'success', title: 'Electron', confidence: '100.0%' },
  );

  assert.deepEqual(annotation, {
    colour: MODULE4_CLASS_COLOURS.Electron,
    text: 'Track 2 · Electron 100.0%',
  });
  assert.equal(annotation.colour, '#60a5fa');
});

test('Module 4 does not manufacture annotations for unavailable classifications', () => {
  assert.equal(classifiedAnnotation({ label_id: 3 }, { kind: 'unavailable' }), null);
});

test('mixed Module 4 results retain one purple Alpha and one blue Electron annotation', () => {
  const annotations = [
    classifiedAnnotation(
      { label_id: 1 },
      { kind: 'success', title: 'Alpha', confidence: '87.0%' },
    ),
    classifiedAnnotation(
      { label_id: 2 },
      { kind: 'success', title: 'Electron', confidence: '93.0%' },
    ),
  ];

  assert.deepEqual(
    annotations.map(({ colour }) => colour),
    [MODULE4_CLASS_COLOURS.Alpha, MODULE4_CLASS_COLOURS.Electron],
  );
});
