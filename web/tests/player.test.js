'use strict';
const test = require('node:test');
const assert = require('node:assert');
const P = require('../player.js');
const C = require('../client.js');

function chunkBytes(epoch, utt, cid, samples = 240) {
  const out = new Uint8Array(16 + samples * 2);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, epoch, true);
  dv.setUint32(4, utt, true);
  dv.setUint32(8, cid, true);
  dv.setUint32(12, samples * 2, true);
  for (let i = 0; i < samples; i++) dv.setInt16(16 + i * 2, i, true);
  return out;
}

function mk() {
  const sent = [];
  const played = [];
  let ended = null;
  let pos = 0;
  const sink = {
    gain: 1,
    play(c, cb) { played.push(c); ended = cb; pos = 0; },
    stop() { ended = null; },
    setGain(g) { this.gain = g; },
    positionMs() { return pos; },
  };
  const player = P.createPlayer({ sink, send: (m) => sent.push(m), pcmRate: 24000 });
  return { player, sink, sent, played, finish: () => { const cb = ended; ended = null; cb(); }, setPos: (p) => { pos = p; } };
}

test('header parse matches injected 4 fields', () => {
  const c = P.parseDownFrame(chunkBytes(7, 8, 9, 10));
  assert.deepStrictEqual([c.epoch, c.utterance_id, c.chunk_id, c.length], [7, 8, 9, 20]);
  assert.strictEqual(c.pcm.length, 10);
  const big = P.parseDownFrame(chunkBytes(4294967295, 4294967295, 4294967295, 2));
  assert.strictEqual(big.epoch, 4294967295);
  assert.strictEqual(P.parseDownFrame(new Uint8Array(15)), null);
});

test('cancel discards queued and playing old epoch, rejects epoch 4 afterwards', () => {
  const m = mk();
  assert.ok(m.player.pushBinary(chunkBytes(3, 1, 0)));
  assert.ok(m.player.pushBinary(chunkBytes(3, 1, 1)));
  m.player.handleMessage({ t: 'cancel', new_epoch: 5 });
  assert.strictEqual(m.player.current, null);
  assert.strictEqual(m.player.queueLength, 0);
  assert.strictEqual(m.player.pushBinary(chunkBytes(4, 2, 0)), false);
  assert.strictEqual(m.played.length, 1);
  assert.ok(m.player.pushBinary(chunkBytes(5, 2, 0)));
  assert.strictEqual(m.played.length, 2);
  const dropped = m.sent.filter((x) => x.state === 'dropped');
  assert.strictEqual(dropped.length, 2);
});

test('cancel: 100 injections yield zero old-epoch plays', () => {
  const m = mk();
  m.player.handleMessage({ t: 'cancel', new_epoch: 5 });
  for (let i = 0; i < 100; i++) m.player.pushBinary(chunkBytes(i % 5, 1, i));
  assert.strictEqual(m.played.filter((c) => c.epoch < 5).length, 0);
  assert.strictEqual(m.played.length, 0);
});

test('duck sets gain 0.2, resume restores 1.0, position never decreases', () => {
  const m = mk();
  m.player.pushBinary(chunkBytes(1, 1, 0, 2400));
  m.setPos(30);
  const p0 = m.player.positionMs();
  m.player.handleMessage({ t: 'duck', epoch: 1, gain: P.DUCK_GAIN });
  assert.strictEqual(m.sink.gain, 0.2);
  m.setPos(50);
  const p1 = m.player.positionMs();
  m.player.handleMessage({ t: 'resume', epoch: 1 });
  assert.strictEqual(m.sink.gain, 1.0);
  m.setPos(70);
  const p2 = m.player.positionMs();
  assert.ok(p0 <= p1 && p1 <= p2);
  assert.strictEqual(m.played.length, 1); // not restarted
});

test('cancel after duck restores gain for the next utterance', () => {
  const m = mk();
  m.player.pushBinary(chunkBytes(1, 1, 0));
  m.player.handleMessage({ t: 'duck', epoch: 1, gain: P.DUCK_GAIN });
  m.player.handleMessage({ t: 'cancel', new_epoch: 2 });
  assert.strictEqual(m.player.ducked, false);
  assert.strictEqual(m.sink.gain, 1.0);
  m.player.pushBinary(chunkBytes(2, 2, 0));
  assert.strictEqual(m.sink.gain, 1.0);
});

test('lookahead is at most one chunk', () => {
  const m = mk();
  assert.ok(m.player.pushBinary(chunkBytes(1, 1, 0)));
  assert.ok(m.player.pushBinary(chunkBytes(1, 1, 1)));
  assert.strictEqual(m.player.pushBinary(chunkBytes(1, 1, 2)), false);
  assert.strictEqual(m.player.queueLength, 1);
});

test('playback messages have the 5 keys and schema-valid values', () => {
  const m = mk();
  m.player.pushBinary(chunkBytes(2, 3, 4, 240));
  m.setPos(5);
  m.finish();
  const states = m.sent.map((x) => x.state);
  assert.deepStrictEqual(states, ['started', 'finished']);
  for (const x of m.sent) {
    assert.strictEqual(x.t, 'playback');
    for (const k of ['epoch', 'utterance_id', 'chunk_id', 'position_ms', 'state']) assert.ok(k in x, k);
    assert.ok(Number.isInteger(x.position_ms) && x.position_ms >= 0);
    assert.deepStrictEqual([x.epoch, x.utterance_id, x.chunk_id], [2, 3, 4]);
  }
  assert.strictEqual(m.sent[1].position_ms, 10); // 240 samples @24k
});

function fakeWs() {
  const created = [];
  class WS {
    constructor(url) { this.url = url; this.sent = []; created.push(this); }
    send(d) { this.sent.push(d); }
    close() {}
  }
  return { WS, created };
}

test('client: onOpen runs after hello', () => {
  const { WS, created } = fakeWs();
  const order = [];
  C.createClient({ port: 1, WebSocketCtor: WS, onOpen: () => order.push(JSON.parse(created[0].sent[0]).t) });
  created[0].onopen();
  assert.deepStrictEqual(order, ['hello']);
});

test('client: hello on open, pong on ping, url, constraints, uplink frame', () => {
  const { WS, created } = fakeWs();
  const m = mk();
  const client = C.createClient({ port: 8765, WebSocketCtor: WS, player: m.player, clientId: 'c1' });
  const ws = created[0];
  assert.strictEqual(ws.url, 'ws://127.0.0.1:8765/ws');
  ws.onopen();
  assert.deepStrictEqual(JSON.parse(ws.sent[0]), { t: 'hello', version: 1, sample_rate: 16000, client_id: 'c1' });
  ws.onmessage({ data: JSON.stringify({ t: 'ping', ts: 42 }) });
  assert.deepStrictEqual(JSON.parse(ws.sent[1]), { t: 'pong', ts: 42 });
  assert.strictEqual(C.micConstraints().audio.echoCancellation, true);

  const framer = C.W.createFramer({ frameSamples: C.FRAME_SAMPLES });
  const [frame] = framer.push(new Float32Array(C.FRAME_SAMPLES));
  client.sendFrame(frame);
  const f = ws.sent[2];
  assert.strictEqual(C.W.readSeq(f), 0);
  assert.strictEqual(C.W.frameSampleCount(f), 320);
});

test('client: downlink binary and cancel routed to player; state/ui', () => {
  const { WS, created } = fakeWs();
  const m = mk();
  const states = [];
  const client = C.createClient({ port: 1, WebSocketCtor: WS, player: m.player, onState: (s) => states.push(s) });
  const ws = created[0];
  ws.onopen();
  ws.onmessage({ data: chunkBytes(1, 1, 0).buffer });
  assert.strictEqual(m.played.length, 1);
  ws.onmessage({ data: JSON.stringify({ t: 'cancel', new_epoch: 2 }) });
  assert.strictEqual(m.player.minEpoch, 2);
  ws.onmessage({ data: JSON.stringify({ t: 'state', state: 'idle', muted: false }) });
  assert.strictEqual(states[0].state, 'idle');
  client.sendUi({ mute: true, on: false, radio: true, interval_s: 30, quiet_start: '23:00', quiet_end: '07:00' });
  assert.deepStrictEqual(JSON.parse(ws.sent[ws.sent.length - 1]), {
    t: 'ui', mute: true, on: false, radio: true, interval_s: 30, quiet_start: '23:00', quiet_end: '07:00' });
});

test('client direct: search and links routed to their handlers', () => {
  const { WS, created } = fakeWs();
  const searches = [];
  const links = [];
  C.createClient({ port: 1, WebSocketCtor: WS, onSearch: (m) => searches.push(m), onLinks: (m) => links.push(m) });
  const ws = created[0];
  ws.onmessage({ data: JSON.stringify({ t: 'search', query: 'q', active: true }) });
  ws.onmessage({ data: JSON.stringify({ t: 'links', items: [{ title: 'A', url: 'https://a.test/' }] }) });
  assert.deepStrictEqual(searches, [{ t: 'search', query: 'q', active: true }]);
  assert.deepStrictEqual(links, [{ t: 'links', items: [{ title: 'A', url: 'https://a.test/' }] }]);
});

test('client direct: malformed/non-object JSON ignored, error routed to onError, no player ok', () => {
  const { WS, created } = fakeWs();
  const errors = [];
  C.createClient({ port: 1, WebSocketCtor: WS, onError: (e) => errors.push(e) });
  const ws = created[0];
  assert.strictEqual(ws.binaryType, 'arraybuffer');
  ws.onmessage({ data: '{not json' });
  ws.onmessage({ data: 'null' });
  ws.onmessage({ data: '5' });
  ws.onmessage({ data: new ArrayBuffer(20) }); // binary without player: no throw
  ws.onmessage({ data: JSON.stringify({ t: 'duck', epoch: 1, gain: 0.2 }) }); // no player: ignored
  ws.onmessage({ data: JSON.stringify({ t: 'error', code: 'E1' }) });
  assert.deepStrictEqual(errors, [{ t: 'error', code: 'E1' }]);
  assert.strictEqual(ws.sent.length, 0);
});

test('client direct: ping->pong echoes ts, duck routed to player, sendJson/sendFrame/close', () => {
  const { WS, created } = fakeWs();
  const m = mk();
  let closed = false;
  WS.prototype.close = function () { closed = true; };
  const client = C.createClient({ port: 9, WebSocketCtor: WS, player: m.player });
  const ws = created[0];
  ws.onopen();
  assert.match(client.clientId, /^web-/);
  ws.onmessage({ data: JSON.stringify({ t: 'ping', ts: 7 }) });
  assert.deepStrictEqual(JSON.parse(ws.sent[1]), { t: 'pong', ts: 7 }); // sent[0] は hello
  m.player.pushBinary(chunkBytes(1, 1, 0, 2400));
  ws.onmessage({ data: JSON.stringify({ t: 'duck', epoch: 1, gain: 0.2 }) });
  assert.strictEqual(m.sink.gain, 0.2);
  const frame = new Uint8Array([1, 2, 3]);
  client.sendFrame(frame);
  assert.strictEqual(ws.sent[ws.sent.length - 1], frame);
  client.sendJson({ t: 'playback', state: 'started' });
  assert.deepStrictEqual(JSON.parse(ws.sent[ws.sent.length - 1]), { t: 'playback', state: 'started' });
  client.close();
  assert.ok(closed);
});

test('client direct: pure helpers makeUi/stateLabel/makeHello/micConstraints/wsUrl', () => {
  assert.deepStrictEqual(C.makeUi({ mute: false, bogus: 1 }), { t: 'ui', mute: false });
  assert.deepStrictEqual(C.makeUi(null), { t: 'ui' });
  assert.deepStrictEqual(C.makeUi({ speaker: 893625024 }), { t: 'ui', speaker: 893625024 });
  assert.strictEqual(C.stateLabel({ state: 'idle', muted: false }, { idle: '待機' }), '待機');
  assert.strictEqual(C.stateLabel({ state: 'idle', muted: true }, { idle: '待機' }), '待機 (休止)');
  assert.strictEqual(C.stateLabel({ state: 'rest', muted: true }, { rest: '休止' }), '休止');
  assert.strictEqual(C.stateLabel({ state: 'x', muted: false }), 'x');
  assert.deepStrictEqual(C.makeHello('z'), { t: 'hello', version: 1, sample_rate: 16000, client_id: 'z' });
  assert.deepStrictEqual(C.micConstraints(), { audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: false, channelCount: 1 } });
  assert.strictEqual(C.wsUrl(1), 'ws://127.0.0.1:1/ws');
});

test('reset: drops queue silently and accepts epoch 0 again', () => {
  const P = require('../player.js');
  const sent = [];
  let stops = 0;
  const sink = { play() {}, stop() { stops += 1; }, setGain() {}, positionMs: () => 0 };
  const p = P.createPlayer({ sink, send: (m) => sent.push(m) });
  p.cancel(5);
  const chunk = (epoch, id) => ({ epoch, utterance_id: 1, chunk_id: id, pcm: new Int16Array(10) });
  assert.equal(p.pushChunk(chunk(5, 0)), true);
  p.pushChunk(chunk(5, 1));
  p.duck(5);
  const n = sent.length;
  p.reset();
  assert.equal(sent.length, n);
  assert.equal(stops, 1);
  assert.equal(p.queueLength, 0);
  assert.equal(p.gain, 1.0);
  assert.equal(p.pushChunk(chunk(0, 0)), true);
});
