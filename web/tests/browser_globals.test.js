'use strict';
// index.html / capture_check.html と同じ順に classic script として 1 つのグローバルスコープで読む。
// トップレベルの const 重複は node の require では検出できない。
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function loadScripts(names) {
  const ctx = vm.createContext({});
  ctx.globalThis = ctx;
  for (const n of names) {
    vm.runInContext(fs.readFileSync(path.join(__dirname, '..', n), 'utf8'), ctx, { filename: n });
  }
  return ctx;
}

test('index.html scripts share one global scope', () => {
  const g = loadScripts(['worklet.js', 'player.js', 'client.js']);
  assert.strictEqual(typeof g.IdleTalkWorklet.createFramer, 'function');
  assert.strictEqual(typeof g.IdleTalkPlayer.createPlayer, 'function');
  assert.strictEqual(typeof g.IdleTalkClient.createClient, 'function');
  assert.strictEqual(g.IdleTalkClient.W, g.IdleTalkWorklet);
  assert.strictEqual(g.IdleTalkClient.P, g.IdleTalkPlayer);
});

test('capture_check.html scripts share one global scope', () => {
  const g = loadScripts(['worklet.js', 'capture_check.js']);
  assert.ok(g.IdleTalkWorklet);
});
