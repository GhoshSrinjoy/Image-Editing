// Thin JavaScript wrapper around imgedit.wasm (the C core compiled to WebAssembly).
// Images stay in wasm memory; JS holds Image objects that wrap their pointers.

export const FORMAT = { png: 1, jpeg: 2, bmp: 3, tga: 4, qoi: 5, ppm: 6, pgm: 7, pam: 8 };
export const INVERT = { rgb: 1, alpha: 2, lightness: 4 };
export const BG = { transparent: 0, color: 1, blur: 2, image: 3 };

export async function loadImgedit(source = new URL('imgedit.wasm', import.meta.url)) {
  let module;
  if (source instanceof ArrayBuffer || ArrayBuffer.isView(source)) {
    module = await WebAssembly.compile(source);
  } else {
    const res = await fetch(source);
    if (!res.ok) throw new Error(`could not load ${source} (${res.status})`);
    module = WebAssembly.compileStreaming
      ? await WebAssembly.compileStreaming(res).catch(async () => WebAssembly.compile(await (await fetch(source)).arrayBuffer()))
      : await WebAssembly.compile(await res.arrayBuffer());
  }
  // The core needs no imports, but give any WASI symbol a harmless stub just in case.
  const stubs = {};
  for (const imp of WebAssembly.Module.imports(module)) {
    (stubs[imp.module] ??= {})[imp.name] = () => 52; // ENOSYS
  }
  const instance = await WebAssembly.instantiate(module, stubs);
  instance.exports._initialize?.();
  return new Imgedit(instance.exports);
}

export class Imgedit {
  constructor(exports) {
    this.x = exports;
  }

  get memory() {
    return this.x.memory.buffer;
  }

  check(code, what) {
    if (code !== 0) throw new Error(`${what} failed (error ${code})`);
  }

  // Copy bytes into wasm memory; returns the pointer (free with dealloc).
  put(bytes) {
    const ptr = this.x.alloc(bytes.byteLength);
    if (!ptr) throw new Error('out of memory');
    new Uint8Array(this.memory, ptr, bytes.byteLength).set(new Uint8Array(bytes.buffer ?? bytes, bytes.byteOffset ?? 0, bytes.byteLength));
    return ptr;
  }

  create(width, height) {
    const ptr = this.x.img_new(width, height);
    if (!ptr) throw new Error(`cannot allocate a ${width}x${height} image`);
    return new Image(this, ptr);
  }

  fromImageData(data) {
    const img = this.create(data.width, data.height);
    img.pixels().set(data.data);
    return img;
  }

  // Decode PNG, JPEG, BMP, GIF, TGA, PSD, HDR, PNM, PAM or QOI bytes.
  decode(bytes) {
    const ptr = this.put(bytes);
    const img = this.x.img_decode(ptr, bytes.byteLength);
    this.x.dealloc(ptr);
    if (!img) throw new Error('unsupported or damaged image');
    return new Image(this, img);
  }

  floats(n) {
    const ptr = this.x.alloc(n * 4);
    if (!ptr) throw new Error('out of memory');
    return ptr;
  }
}

export class Image {
  constructor(lib, ptr) {
    this.lib = lib;
    this.ptr = ptr;
  }

  get width() {
    return this.lib.x.img_width(this.ptr);
  }

  get height() {
    return this.lib.x.img_height(this.ptr);
  }

  // Live view of the RGBA pixels (invalidated when memory grows or the image is resized).
  pixels() {
    return new Uint8ClampedArray(this.lib.memory, this.lib.x.img_pixels(this.ptr), this.width * this.height * 4);
  }

  toImageData() {
    return new ImageData(this.pixels().slice(), this.width, this.height);
  }

  hasAlpha() {
    return !!this.lib.x.img_has_alpha(this.ptr);
  }

  clone() {
    const p = this.lib.x.img_clone(this.ptr);
    if (!p) throw new Error('out of memory');
    return new Image(this.lib, p);
  }

  free() {
    if (this.ptr) this.lib.x.img_free(this.ptr);
    this.ptr = 0;
  }

  encode(format, quality = 0) {
    const x = this.lib.x;
    const p = x.img_encode(this.ptr, typeof format === 'string' ? FORMAT[format] : format, quality);
    if (!p) throw new Error(`encode failed (error ${x.last_error()})`);
    const out = new Uint8Array(this.lib.memory, p, x.last_len()).slice();
    x.dealloc(p);
    return out;
  }

  op(name, ...args) {
    this.lib.check(this.lib.x[name](this.ptr, ...args), name);
    return this;
  }

  invert(mode = 'rgb') { return this.op('invert', INVERT[mode] ?? mode); }
  grayscale() { return this.op('grayscale'); }
  sepia(amount = 1) { return this.op('sepia', amount); }
  autoLevels(perChannel = false, clip = 0.005) { return this.op('auto_levels', clip, perChannel ? 1 : 0); }
  threshold(level = 128) { return this.op('threshold', level); }
  posterize(levels = 4) { return this.op('posterize', levels); }
  flatten(rgb = 0xffffff) { return this.op('flatten', rgb >>> 0); }
  alphaToGray() { return this.op('alpha_to_gray'); }
  adjust(a = {}) {
    return this.op('adjust', a.exposure ?? 0, a.brightness ?? 0, a.contrast ?? 0, a.gamma ?? 1, a.saturation ?? 0, a.hue ?? 0,
      a.temperature ?? 0, a.tint ?? 0);
  }
  resize(w, h) { return this.op('resize', Math.max(1, Math.round(w)), Math.max(1, Math.round(h))); }
  crop(x, y, w, h) { return this.op('crop', Math.round(x), Math.round(y), Math.round(w), Math.round(h)); }
  rotate90(turns = 1) { return this.op('rotate90', turns); }
  rotate(degrees, fillRGBA = 0) { return this.op('rotate', degrees, fillRGBA >>> 0); }
  flip(horizontal = true) { return this.op('flip', horizontal ? 1 : 0); }
  trim(tolerance = 8) { return this.op('trim', tolerance); }
  pad(t, r = t, b = t, l = r, fillRGBA = 0) { return this.op('pad', t, r, b, l, fillRGBA >>> 0); }
  blur(sigma) { return this.op('blur', sigma); }
  sharpen(amount = 0.8, sigma = 1, threshold = 2) { return this.op('sharpen', amount, sigma, threshold); }
  alphaLevels(lo, hi) { return this.op('alpha_levels', lo, hi); }
  alphaGrow(radius) { return this.op('alpha_grow', radius); }
  alphaFeather(sigma) { return this.op('alpha_feather', sigma); }
  alphaClean(minArea = 0.01) { return this.op('alpha_clean', minArea); }
  decontaminate() { return this.op('decontaminate'); }
  keyBackground({ color = null, tolerance = 10, softness = 8, connected = true } = {}) {
    return this.op('key_background', color === null ? 1 : 0, (color ?? 0) >>> 0, tolerance, softness, connected ? 1 : 0);
  }
  finishCutout({ mode = 'transparent', color = 0xffffff, blur = 0, image = null, decontaminate = true } = {}) {
    return this.op('finish_cutout', BG[mode] ?? mode, color >>> 0, blur, image ? image.ptr : 0, decontaminate ? 1 : 0);
  }

  // Model input tensor as a Float32Array (planar RGB, th x tw).
  nnInput(tw, th, mean, std, scale) {
    const lib = this.lib;
    const n = 3 * tw * th;
    const p = lib.floats(n);
    try {
      lib.check(lib.x.nn_input(this.ptr, tw, th, mean[0], mean[1], mean[2], std[0], std[1], std[2], scale, p), 'nn_input');
      return new Float32Array(lib.memory, p, n).slice();
    } finally {
      lib.x.dealloc(p);
    }
  }

  // Use a model's raw output (Float32Array, mw x mh) as this image's alpha.
  applyMatte(raw, mw, mh, { minmax = true, radius = 4, eps = 1e-4 } = {}) {
    const lib = this.lib;
    const p = lib.floats(raw.length);
    try {
      new Float32Array(lib.memory, p, raw.length).set(raw);
      lib.x.nn_output(p, raw.length, minmax ? 1 : 0);
      lib.check(lib.x.matte_apply(this.ptr, p, mw, mh, radius, eps), 'matte_apply');
    } finally {
      lib.x.dealloc(p);
    }
    return this;
  }
}

// float32 <-> float16 for fp16 models, done in wasm.
export function toHalf(lib, f32) {
  const n = f32.length, src = lib.floats(n), dst = lib.x.alloc(n * 2);
  try {
    new Float32Array(lib.memory, src, n).set(f32);
    lib.x.f32_to_f16(src, dst, n);
    return new Uint16Array(lib.memory, dst, n).slice();
  } finally {
    lib.x.dealloc(src);
    lib.x.dealloc(dst);
  }
}

export function fromHalf(lib, u16) {
  const n = u16.length, src = lib.x.alloc(n * 2), dst = lib.floats(n);
  try {
    new Uint16Array(lib.memory, src, n).set(u16);
    lib.x.f16_to_f32(src, dst, n);
    return new Float32Array(lib.memory, dst, n).slice();
  } finally {
    lib.x.dealloc(src);
    lib.x.dealloc(dst);
  }
}
