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
  const RECONNECT_MS = 2000; // 切断から再接続までの間隔

  function wsUrl(port) {
    return `ws://127.0.0.1:${port}/ws`;
  }

  function micConstraints() {
    // AGC off: the server's VAD calibrates its threshold from the noise floor, which AGC would keep shifting.
    return { audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: false, channelCount: 1 } };
  }

  // ドラマモード: 共有したタブ (画面) の音だけを使う。Chrome は video 無しの共有を受け付けないので映像は取ってすぐ止める
  function displayConstraints() {
    return {
      video: true,
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      systemAudio: 'include',
    };
  }

  function makeHello(clientId) {
    return { t: 'hello', version: PROTOCOL_VERSION, sample_rate: SAMPLE_RATE, client_id: clientId };
  }

  function makeUi(settings) {
    const out = { t: 'ui' };
    for (const k of ['mute', 'on', 'radio', 'interval_s', 'quiet_start', 'quiet_end', 'speaker', 'drama', 'persona']) {
      if (settings && settings[k] !== undefined) out[k] = settings[k];
    }
    return out;
  }

  function stateLabel(msg, names) {
    const label = (names && names[msg.state]) || msg.state;
    // 休止のチェック中は状態も「休止」になるので重ねて書かない (状態がまだ切り替わる前だけ添える)
    return msg.muted && msg.state !== 'rest' ? `${label} (休止)` : label;
  }

  // opts: { port, WebSocketCtor, player, onState(msg), onError(msg), onSearch(msg), onLinks(msg), onOpen(), onClose(),
  //         clientId, reconnectMs, setTimeoutFn }
  // 接続が切れた・つながらなかったら reconnectMs 後につなぎ直す (サーバの再起動中に開始を押しても待てば動く)。close() で止める
  function createClient(opts) {
    const { port, WebSocketCtor, player, onState, onError, onSearch, onLinks, onOpen, onClose } = opts;
    const clientId = opts.clientId || `web-${Math.random().toString(36).slice(2, 10)}`;
    const reconnectMs = opts.reconnectMs || RECONNECT_MS;
    const setTimeoutFn = opts.setTimeoutFn || setTimeout;
    let ws = null;
    let closed = false;
    // hello 前・切断中は送らない (CONNECTING の send は例外、サーバは hello 前のメッセージを受け付けない)
    let ready = false;
    const sendJson = (m) => { if (ready) ws.send(JSON.stringify(m)); };

    function connect() {
      ws = new WebSocketCtor(wsUrl(port));
      ws.binaryType = 'arraybuffer';
      ws.onopen = () => {
        ready = true;
        sendJson(makeHello(clientId));
        if (onOpen) onOpen(); // hello の後 (サーバは hello 前のメッセージを受け付けない)
      };
      ws.onclose = () => {
        ready = false;
        if (player && player.reset) player.reset(); // 前の接続の音は鳴らさない。新しい接続の epoch は 0 から
        if (closed) return;
        if (onClose) onClose();
        setTimeoutFn(() => { if (!closed) connect(); }, reconnectMs);
      };
      ws.onmessage = onMessage;
    }

    function onMessage(ev) {
      const d = ev.data;
      if (typeof d !== 'string') { if (player) player.pushBinary(d); return; }
      let msg;
      try { msg = JSON.parse(d); } catch (_e) { return; }
      if (!msg || typeof msg !== 'object') return;
      if (msg.t === 'ping') sendJson({ t: 'pong', ts: msg.ts });
      else if (msg.t === 'state') { if (onState) onState(msg); }
      else if (msg.t === 'error') { if (onError) onError(msg); }
      else if (msg.t === 'search') { if (onSearch) onSearch(msg); }
      else if (msg.t === 'links') { if (onLinks) onLinks(msg); }
      else if (player) player.handleMessage(msg);
    }

    connect();
    return {
      get ws() { return ws; },
      clientId,
      sendFrame(frame) { if (ready) ws.send(frame); },
      sendUi(settings) { sendJson(makeUi(settings)); },
      sendJson,
      close() { closed = true; ws.close(); },
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

  // 共有したタブの音 -> worklet (seq に DRAMA_FLAG) -> client.sendFrame。共有が終わったら onEnded。Browser only.
  // 失敗した段階をエラーに付ける (ブラウザの例外は "Not supported" のように段階が分からない)
  async function step(name, fn) {
    try {
      return await fn();
    } catch (e) {
      throw new Error(`${name}: ${e && e.name ? `${e.name}: ${e.message}` : e}`);
    }
  }

  async function startDramaCapture({ getDisplayMedia, AudioContextCtor, workletUrl, client, onEnded }) {
    const stream = await step('タブの共有', async () => {
      try {
        return await getDisplayMedia(displayConstraints());
      } catch (e) {
        // 音声の詳細な制約や systemAudio を受け付けないブラウザ: 最小の制約で取り直す
        if (!e || (e.name !== 'NotSupportedError' && e.name !== 'TypeError' && e.name !== 'OverconstrainedError')) throw e;
        return getDisplayMedia({ video: true, audio: true });
      }
    });
    const audio = stream.getAudioTracks();
    if (!audio.length) {
      for (const t of stream.getVideoTracks()) t.stop();
      throw new Error('音声が共有されていません。共有の画面で「タブの音声も共有する」をオンにしてください');
    }
    for (const t of stream.getVideoTracks()) t.stop();
    const ctx = await step('音声の準備', () => new AudioContextCtor());
    await step('音声処理の読み込み', () => ctx.audioWorklet.addModule(workletUrl));
    const node = await step('音声処理の作成', () => new AudioWorkletNode(ctx, 'capture-processor', {
      processorOptions: { inRate: ctx.sampleRate, frameSamples: FRAME_SAMPLES, seqBase: W.DRAMA_FLAG },
    }));
    node.port.onmessage = (ev) => client.sendFrame(ev.data);
    await step('共有した音の接続', () => ctx.createMediaStreamSource(new MediaStream(audio)).connect(node));
    let stopped = false;
    const stop = () => {
      if (stopped) return;
      stopped = true;
      for (const t of audio) t.stop();
      ctx.close();
      if (onEnded) onEnded();
    };
    audio[0].addEventListener('ended', stop); // ブラウザの「共有を停止」
    return { ctx, stream, node, stop };
  }

  const api = {
    PROTOCOL_VERSION, SAMPLE_RATE, FRAME_SAMPLES, W, P,
    wsUrl, micConstraints, displayConstraints, makeHello, makeUi, stateLabel, createClient,
    startCapture, startDramaCapture,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else if (typeof globalThis !== 'undefined') globalThis.IdleTalkClient = api;
})();
