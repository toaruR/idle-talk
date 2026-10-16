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

test('snapSize: fits 640x360 keeping aspect, never enlarges', () => {
  assert.deepEqual(C.snapSize(1920, 1080), [640, 360]);
  assert.deepEqual(C.snapSize(1080, 1920), [203, 360]); // 縦長
  assert.deepEqual(C.snapSize(1280, 1024), [450, 360]);
  assert.deepEqual(C.snapSize(320, 180), [320, 180]);
});

test('createClient: snap request goes to onSnap', () => {
  const { WS, made } = fakeWs();
  const seen = [];
  C.createClient({ port: 8765, WebSocketCtor: WS, player: null, onSnap: (m) => seen.push(m), setTimeoutFn: () => {} });
  made[0].onmessage({ data: JSON.stringify({ t: 'snap', id: 3 }) });
  assert.deepEqual(seen, [{ t: 'snap', id: 3 }]);
});

// AudioContext 等を差し替えて、映像トラックを残すか (vision) と止め方を見る
function fakeDrama() {
  const stops = { audio: 0, video: 0 };
  const listeners = {};
  const audio = { stop: () => { stops.audio += 1; }, addEventListener: (ev, fn) => { listeners[ev] = fn; } };
  const video = { stop: () => { stops.video += 1; }, readyState: 'live' };
  const stream = { getVideoTracks: () => [video], getAudioTracks: () => [audio] };
  class Ctx {
    constructor() { this.sampleRate = 48000; this.audioWorklet = { addModule: async () => {} }; }
    createMediaStreamSource() { return { connect: () => {} }; }
    close() {}
  }
  globalThis.AudioWorkletNode = class { constructor() { this.port = {}; } };
  globalThis.MediaStream = class { constructor(t) { this.tracks = t; } };
  return { stops, stream, Ctx };
}

test('startDramaCapture: without vision stops video at once and snapshot is null', async () => {
  const f = fakeDrama();
  const d = await C.startDramaCapture({ getDisplayMedia: async () => f.stream, AudioContextCtor: f.Ctx, workletUrl: '', client: {} });
  assert.equal(f.stops.video, 1);
  assert.deepEqual(await d.snapshot(), { reason: 'vision off at share start' });
  d.stop();
  assert.deepEqual(f.stops, { audio: 1, video: 1 });
});

test('startDramaCapture: vision keeps video until stop', async () => {
  const f = fakeDrama();
  const d = await C.startDramaCapture({
    getDisplayMedia: async () => f.stream, AudioContextCtor: f.Ctx, workletUrl: '', client: {}, vision: Promise.resolve(true),
  });
  assert.equal(f.stops.video, 0);
  assert.equal((await d.snapshot()).jpeg, undefined); // 描けないので画像なし (理由を返す)
  assert.match((await d.snapshot()).reason, /video track|ImageCapture/);
  d.stop();
  assert.deepEqual(f.stops, { audio: 1, video: 1 });
  assert.deepEqual(await d.snapshot(), { reason: 'capture stopped' }); // 止めた後は求めない
});

test('errorResolved: transient errors clear when playback starts', () => {
  const started = { t: 'playback', state: 'started' };
  for (const code of [5001, 5002, 5003]) assert.equal(C.errorResolved(code, started), true);
  assert.equal(C.errorResolved(5002, { t: 'playback', state: 'finished' }), false);
  assert.equal(C.errorResolved(5002, { t: 'snap', id: 1 }), false);
  assert.equal(C.errorResolved(5006, started), false); // 設定の範囲外は再生では解消しない
  assert.equal(C.errorResolved(null, started), false);
});
