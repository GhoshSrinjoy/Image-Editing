// Tests for the WebAssembly build. Run: node tests/wasm.test.mjs (after building web/imgedit.wasm).
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const { loadImgedit, toHalf, fromHalf } = await import(pathToFileURL(join(root, 'web/imgedit.js')).href);
const lib = await loadImgedit(readFileSync(join(root, 'web/imgedit.wasm')));

let passed = 0;
function test(name, fn) {
  fn();
  passed++;
  console.log(`ok ${name}`);
}

function gradient(w, h, alpha = false) {
  const img = lib.create(w, h);
  const px = img.pixels();
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      px[i] = (x * 255) / (w - 1);
      px[i + 1] = (y * 255) / (h - 1);
      px[i + 2] = (x ^ y) & 255;
      px[i + 3] = alpha ? (x + y) & 255 : 255;
    }
  return img;
}

const equal = (a, b) => a.width === b.width && a.height === b.height && a.pixels().every((v, i) => v === b.pixels()[i]);

test('png round trip keeps every pixel', () => {
  for (const alpha of [false, true]) {
    const img = gradient(321, 123, alpha);
    const back = lib.decode(img.encode('png'));
    assert.ok(equal(img, back));
    img.free();
    back.free();
  }
});

test('qoi / tga / bmp round trips', () => {
  const img = gradient(64, 48, true);
  for (const fmt of ['qoi', 'tga', 'bmp']) {
    const back = lib.decode(img.encode(fmt));
    assert.ok(equal(img, back), fmt);
    back.free();
  }
  img.free();
});

test('jpeg decodes to the same size', () => {
  const img = gradient(200, 100);
  const back = lib.decode(img.encode('jpeg', 90));
  assert.equal(back.width, 200);
  assert.equal(back.height, 100);
  img.free();
  back.free();
});

test('geometry', () => {
  const img = gradient(50, 30, true);
  const ref = img.clone();
  img.rotate90(1).rotate90(1).rotate90(1).rotate90(1);
  assert.ok(equal(img, ref));
  img.flip(true).flip(true);
  assert.ok(equal(img, ref));
  img.crop(5, 5, 20, 10);
  assert.equal(img.width, 20);
  assert.equal(img.height, 10);
  img.resize(40, 20);
  assert.equal(img.width, 40);
  img.free();
  ref.free();
});

test('invert twice is a no-op', () => {
  const img = gradient(33, 21, true);
  const ref = img.clone();
  img.invert('rgb').invert('rgb').invert('alpha').invert('alpha');
  assert.ok(equal(img, ref));
  img.free();
  ref.free();
});

test('adjust, filters and background ops run', () => {
  const img = gradient(120, 80);
  img.adjust({ exposure: 0.5, contrast: 0.2, saturation: -0.5, hue: 30 }).blur(2).sharpen(1).autoLevels(true).sepia(0.5);
  img.keyBackground({ color: 0xffffff }).finishCutout({ mode: 'color', color: 0x00ff00 });
  assert.ok(!img.hasAlpha());
  img.free();
});

test('matte pipeline', () => {
  const img = gradient(300, 200);
  const mw = 32, mh = 32, matte = new Float32Array(mw * mh);
  for (let y = 0; y < mh; y++) for (let x = 0; x < mw; x++) matte[y * mw + x] = x > 8 && x < 24 && y > 8 && y < 24 ? 4 : -4; // logits
  img.applyMatte(matte, mw, mh).finishCutout({});
  const px = img.pixels();
  assert.equal(px[3], 0); // corner is background
  assert.ok(px[(100 * 300 + 150) * 4 + 3] > 240); // centre is subject
  const input = img.nnInput(64, 64, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 1 / 255);
  assert.equal(input.length, 3 * 64 * 64);
  img.free();
});

test('half floats', () => {
  const v = new Float32Array([0, 1, -2.5, 0.1, 65504, 1e-6]);
  const back = fromHalf(lib, toHalf(lib, v));
  for (let i = 0; i < v.length; i++) assert.ok(Math.abs(back[i] - v[i]) <= Math.abs(v[i]) * 1e-3 + 1e-7, `${v[i]} -> ${back[i]}`);
});

console.log(`${passed} tests passed`);
