'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const C = require('../client.js');

test('makeUi: drama is passed through', () => {
  assert.deepEqual(C.makeUi({ drama: true, mute: false, other: 1 }), { t: 'ui', mute: false, drama: true });
});

test('makeUi: persona is passed through', () => {
  assert.deepEqual(C.makeUi({ persona: 'haru' }), { t: 'ui', persona: 'haru' });
});

test('displayConstraints: tab audio without browser processing', () => {
  const c = C.displayConstraints();
  assert.equal(c.video, true); // Chrome は映像なしの共有を受け付けない
  assert.equal(c.audio.echoCancellation, false);
  assert.equal(c.audio.autoGainControl, false);
});

test('startDramaCapture: no audio track -> error and video stopped', async () => {
  let stopped = 0;
  const stream = { getVideoTracks: () => [{ stop: () => { stopped += 1; } }], getAudioTracks: () => [] };
  await assert.rejects(
    C.startDramaCapture({ getDisplayMedia: async () => stream, AudioContextCtor: null, workletUrl: '', client: null }),
    /タブの音声/,
  );
  assert.equal(stopped, 1);
});
