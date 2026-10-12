'use strict';
// Operations-phase live check. Not named *.test.js on purpose: excluded from `node --test web/tests/*.test.js`.
// Run: node --test web/tests/capture_record_live.js
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { validateRecord } = require('../capture_check.js');

// requires live run: records/capture_check.json saved from a real browser.
test('live record: records/capture_check.json C1-C4 true and provenance valid', () => {
  const p = path.join(__dirname, '..', '..', 'records', 'capture_check.json');
  const rec = JSON.parse(fs.readFileSync(p, 'utf8'));
  for (const k of ['C1', 'C2', 'C3', 'C4']) assert.equal(rec[k], true, k);
  validateRecord(rec);
});
