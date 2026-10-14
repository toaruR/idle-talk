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

function fakeWs() {
  const made = [];
  class WS {
    constructor(url) { this.url = url; this.sent = []; this.closed = false; made.push(this); }
    send(d) { this.sent.push(d); }
    close() { this.closed = true; if (this.onclose) this.onclose(); }
  }
  return { WS, made };
}

test('createClient: reconnects after close and resets player', () => {
  const { WS, made } = fakeWs();
  const timers = [];
  let resets = 0;
  let closes = 0;
  let opens = 0;
  const c = C.createClient({
    port: 8765, WebSocketCtor: WS, player: { reset: () => { resets += 1; } },
    onOpen: () => { opens += 1; }, onClose: () => { closes += 1; },
    setTimeoutFn: (fn, ms) => timers.push({ fn, ms }), reconnectMs: 2000,
  });
  c.sendFrame(new Uint8Array(4)); // CONNECTING: 送らない (例外にしない)
  assert.equal(made[0].sent.length, 0);
  made[0].onclose(); // つながらなかった・切れた
  assert.equal(resets, 1);
  assert.equal(closes, 1);
  assert.equal(timers.length, 1);
  assert.equal(timers[0].ms, 2000);
  timers[0].fn();
  assert.equal(made.length, 2);
  made[1].onopen();
  assert.equal(opens, 1);
  assert.equal(JSON.parse(made[1].sent[0]).t, 'hello');
  assert.equal(c.ws, made[1]);
});

test('createClient: close() stops reconnecting', () => {
  const { WS, made } = fakeWs();
  const timers = [];
  const c = C.createClient({ port: 1, WebSocketCtor: WS, setTimeoutFn: (fn) => timers.push(fn) });
  c.close();
  assert.equal(timers.length, 0);
  assert.equal(made.length, 1);
});
