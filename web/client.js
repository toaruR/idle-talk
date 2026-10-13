'use strict';
// ブラウザでは classic script として共通のグローバルスコープに読まれるため、IIFE で閉じる。
(function () {
  // Browser client: WebSocket <-> Director (design 3.1, 12.1). Browser APIs are injected.

  const W = (typeof module !== 'undefined' && module.exports)
    ? require('./worklet.js') : globalThis.IdleTalkWorklet;
  const P = (typeof module !== 'undefined' && module.exports)
    ? require('./player.js') : globalThis.IdleTalkPlayer;

  const PROTOCOL_VERSION = 1;
  const SAMPLE_RATE = 16000;
  const FRAME_SAMPLES = 320; // 20ms @16kHz

  function wsUrl(port) {
    return `ws://127.0.0.1:${port}/ws`;
  }

  function micConstraints() {
    // AGC off: the server's VAD calibrates its threshold from the noise floor, which AGC would keep shifting.
    return { audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: false, channelCount: 1 } };
  }

  function makeHello(clientId) {
    return { t: 'hello', version: PROTOCOL_VERSION, sample_rate: SAMPLE_RATE, client_id: clientId };
  }

  function makeUi(settings) {
    const out = { t: 'ui' };
    for (const k of ['mute', 'on', 'radio', 'interval_s', 'quiet_start', 'quiet_end', 'speaker']) {
      if (settings && settings[k] !== undefined) out[k] = settings[k];
    }
    return out;
  }

  function stateLabel(msg, names) {
    const label = (names && names[msg.state]) || msg.state;
    return msg.muted ? `${label} (ミュート)` : label;
  }

  // opts: { port, WebSocketCtor, player, onState(msg), onError(msg), onOpen(), clientId }
  function createClient(opts) {
    const { port, WebSocketCtor, player, onState, onError, onOpen } = opts;
    const clientId = opts.clientId || `web-${Math.random().toString(36).slice(2, 10)}`;
    const ws = new WebSocketCtor(wsUrl(port));
    ws.binaryType = 'arraybuffer';
    const sendJson = (m) => ws.send(JSON.stringify(m));

    ws.onopen = () => {
      sendJson(makeHello(clientId));
      if (onOpen) onOpen(); // hello の後 (サーバは hello 前のメッセージを受け付けない)
    };
    ws.onmessage = (ev) => {
      const d = ev.data;
      if (typeof d !== 'string') { if (player) player.pushBinary(d); return; }
      let msg;
      try { msg = JSON.parse(d); } catch (_e) { return; }
      if (!msg || typeof msg !== 'object') return;
      if (msg.t === 'ping') sendJson({ t: 'pong', ts: msg.ts });
      else if (msg.t === 'state') { if (onState) onState(msg); }
      else if (msg.t === 'error') { if (onError) onError(msg); }
      else if (player) player.handleMessage(msg);
    };

    return {
      ws,
      clientId,
      sendFrame(frame) { ws.send(frame); },
      sendUi(settings) { sendJson(makeUi(settings)); },
      sendJson,
      close() { ws.close(); },
    };
  }

  // Start mic capture -> worklet frames -> client.sendFrame. Browser only.
  async function startCapture({ getUserMedia, AudioContextCtor, workletUrl, client }) {
    const stream = await getUserMedia(micConstraints());
    const ctx = new AudioContextCtor();
    await ctx.audioWorklet.addModule(workletUrl);
    const node = new AudioWorkletNode(ctx, 'capture-processor', {
      processorOptions: { inRate: ctx.sampleRate, frameSamples: FRAME_SAMPLES },
    });
    node.port.onmessage = (ev) => client.sendFrame(ev.data);
    ctx.createMediaStreamSource(stream).connect(node);
    return { ctx, stream, node };
  }

  const api = {
    PROTOCOL_VERSION, SAMPLE_RATE, FRAME_SAMPLES, W, P,
    wsUrl, micConstraints, makeHello, makeUi, stateLabel, createClient, startCapture,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else if (typeof globalThis !== 'undefined') globalThis.IdleTalkClient = api;
})();
