'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function setup({ pendingMic = false, pendingDrama = false } = {}) {
  const html = fs.readFileSync(path.join(__dirname, '../index.html'), 'utf8');
  const elements = new Map();
  for (const id of html.matchAll(/id="([^"]+)"/g)) {
    elements.set(id[1], { options: [], checked: false, value: '', dataset: {}, handlers: {},
      addEventListener(name, fn) { this.handlers[name] = fn; },
      setAttribute() {}, add(o) { this.options.push(o); } });
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
    document: { getElementById: id => elements.get(id) },
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
