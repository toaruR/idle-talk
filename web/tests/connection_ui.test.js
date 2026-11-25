'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function element() {
  return { options: [], checked: false, value: '', dataset: {}, handlers: {}, children: [],
    textContent: '', scrollTop: 0, scrollHeight: 0, clientHeight: 0,
    addEventListener(name, fn) { this.handlers[name] = fn; },
    setAttribute() {}, add(o) { this.options.push(o); },
    append(...nodes) { for (const node of nodes) { node.parentNode = this; this.children.push(node); } },
    insertBefore(node, before) { node.parentNode = this; this.children.splice(this.children.indexOf(before), 0, node); },
    remove() {
      const parent = this.parentNode;
      if (parent) parent.children.splice(parent.children.indexOf(this), 1);
      this.parentNode = null;
    },
    get firstElementChild() { return this.children[0]; },
    querySelector(selector) { return selector === 'li.partial' ? this.children.find(n => n.className === 'partial') : null; },
  };
}

function setup({ pendingMic = false, pendingDrama = false } = {}) {
  const html = fs.readFileSync(path.join(__dirname, '../index.html'), 'utf8');
  const elements = new Map();
  for (const id of html.matchAll(/id="([^"]+)"/g)) {
    elements.set(id[1], element());
  }
  elements.get('volume').value = '100';
  elements.get('interval_s').value = '15';
  const clients = [], captures = [], contexts = [], players = [], modes = [];
  let resolveMic, resolveDrama;
  const capture = () => {
    const c = { stops: 0, stop() { this.stops++; } };
    captures.push(c); return c;
  };
  const createMode = (opts) => {
    const m = { stops: 0, stop() { this.stops++; opts.onEnded(); } };
    modes.push(m); return m;
  };
  const clientApi = {
    createClient(opts) {
      const c = { opts, closed: false, sendUi() {}, close() { this.closed = true; } };
      clients.push(c); return c;
    },
    startCapture: () => pendingMic ? new Promise(r => { resolveMic = () => r(capture()); }) : Promise.resolve(capture()),
    startDramaCapture: (opts) => pendingDrama ? new Promise(r => { resolveDrama = () => r(createMode(opts)); }) : Promise.resolve(createMode(opts)),
    startWindowWatch: async (opts) => createMode(opts),
    errorResolved: () => false,
  };
  class AudioContext {
    constructor() { this.closed = false; contexts.push(this); }
    close() { this.closed = true; }
  }
  const events = {};
  const ctx = vm.createContext({
    document: { getElementById: id => elements.get(id), createElement: element },
    IdleTalkClient: clientApi,
    IdleTalkPlayer: { clampVolume: Number, createWebAudioSink: () => ({ setVolume() {} }),
      createPlayer: () => { const p = { resets: 0, reset() { this.resets++; } }; players.push(p); return p; } },
    localStorage: { getItem: () => null },
    window: { addEventListener: (name, fn) => { events[name] = fn; } },
    fetch: async () => ({ ok: true, json: async () => ({ ws_port: 8765, vision: true, follow_window: true }) }),
    navigator: { mediaDevices: {} }, location: { search: '' }, URLSearchParams, AudioContext, WebSocket: class {},
  });
  vm.runInContext([...html.matchAll(/<script>([\s\S]*?)<\/script>/g)][0][1], ctx);
  const flush = async () => { for (let i = 0; i < 12; i++) await Promise.resolve(); };
  const toggle = (checked) => { elements.get('connect').checked = checked; return elements.get('connect').handlers.change(); };
  return { elements, clients, captures, contexts, players, modes, events, flush, toggle,
    resolveMic: () => resolveMic(), resolveDrama: () => resolveDrama() };
}

test('connection toggle stops resources and sharing, disables modes, and reconnects', async () => {
  const f = setup();
  await f.toggle(true);
  await f.elements.get('drama').handlers.click();
  await f.toggle(false);
  assert.equal(f.clients[0].closed, true);
  assert.equal(f.contexts[0].closed, true);
  assert.equal(f.captures[0].stops, 1);
  assert.equal(f.modes[0].stops, 1);
  assert.ok(f.players[0].resets);
  assert.equal(f.elements.get('state').textContent, '停止中');
  for (const id of ['pc', 'drama', 'screen', 'camera']) assert.equal(f.elements.get(id).disabled, true);
  await f.toggle(true);
  assert.equal(f.clients.length, 2);
  assert.equal(f.clients[1].closed, false);
  f.events.pagehide();
  assert.equal(f.clients[1].closed, true);
});

test('stop while microphone permission is pending releases a late stream without reviving UI', async () => {
  const f = setup({ pendingMic: true });
  const pending = f.toggle(true);
  await f.flush();
  await f.toggle(false);
  f.resolveMic();
  await pending;
  assert.equal(f.captures[0].stops, 1);
  assert.equal(f.clients[0].closed, true);
  assert.equal(f.elements.get('connect').checked, false);
  assert.equal(f.elements.get('state').textContent, '停止中');
});

test('a late sharing result from a stopped connection cannot attach to the next connection', async () => {
  const f = setup({ pendingDrama: true });
  await f.toggle(true);
  const pending = f.elements.get('drama').handlers.click();
  await f.flush();
  await f.toggle(false);
  await f.toggle(true);
  f.resolveDrama();
  await pending;
  assert.equal(f.modes[0].stops, 1);
  assert.equal(f.elements.get('drama_state').textContent, '');
  assert.equal(f.clients[1].closed, false);
});

test('AI clauses share one labeled transcript row without changing heard or partial text', async () => {
  const f = setup();
  await f.toggle(true);
  const ui = f.clients[0].opts, list = f.elements.get('transcript');
  ui.onTranscript({ uid: 1, speaker: 'あなた', text: 'こんにちは' });
  ui.onPartial({ seg: 2, text: '続けて話す' });
  ui.onText({ epoch: 0, utterance_id: 99, text: 'こんにちは。' });
  ui.onText({ epoch: 0, utterance_id: 99, text: '<img src=x onerror=alert(1)>' });
  assert.equal(f.elements.get('transcript_box').hidden, false);
  assert.equal(list.children.length, 3);
  assert.equal(list.children[0].children[1].textContent, 'あなた');
  assert.equal(list.children[0].children[2].textContent, 'こんにちは');
  assert.equal(list.children[1].children[1].textContent, 'AI');
  assert.equal(list.children[1].children[2].textContent, 'こんにちは。<img src=x onerror=alert(1)>');
  assert.equal(list.children[1].children[2].children.length, 0); // HTML として挿入しない
  assert.equal(list.children[2].className, 'partial');
  ui.onPartial({ seg: 2, text: '続けて話しています' }); // AI の番号はマイクの確定番号に使わない
  assert.equal(list.children[2].textContent, '続けて話しています');
});

test('new AI turns and epochs get separate rows; reconnect never appends to the previous session', async () => {
  const f = setup();
  await f.toggle(true);
  const ui = f.clients[0].opts, list = f.elements.get('transcript');
  ui.onText({ epoch: 0, utterance_id: 1, text: '最初。' });
  ui.onText({ epoch: 0, utterance_id: 2, text: '次。' });
  ui.onText({ epoch: 1, utterance_id: 2, text: '割り込み後。' });
  ui.onOpen();
  ui.onText({ epoch: 1, utterance_id: 2, text: '再接続後。' });
  assert.deepEqual(list.children.map(li => li.children[2].textContent), ['最初。', '次。', '割り込み後。', '再接続後。']);
});
