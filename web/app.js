// imgedit web app: all pixel work runs in imgedit.wasm (the same C core as the
// command-line tool); background-removal models run in ONNX Runtime Web,
// on the GPU through WebGPU when the browser has it.
import { loadImgedit, fromHalf } from './imgedit.js';

const ORT_VERSION = '1.30.0';
const ORT_DIR = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const HF = 'https://huggingface.co/';
const IMAGENET = { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], scale: 1 / 255 };

// gpu: WebGPU variant (f16 models need the shader-f16 feature, else gpuF32);
// cpu: WebAssembly variant. Sizes in MB.
const MODELS = {
  fast: {
    label: 'Fast', about: 'U²-Netp, tiny and quick', size: 320, ...IMAGENET, minmax: true,
    gpu: { url: HF + 'BritishWerewolf/U-2-Netp/resolve/main/onnx/model.onnx', mb: 4.6 },
    cpu: { url: HF + 'BritishWerewolf/U-2-Netp/resolve/main/onnx/model.onnx', mb: 4.6 },
  },
  balanced: {
    label: 'Balanced', about: 'IS-Net, clean edges for most photos', size: 1024,
    mean: [128, 128, 128], std: [256, 256, 256], scale: 1, minmax: true,
    gpu: { url: HF + 'imgly/isnet-general-onnx/resolve/main/onnx/model_fp16.onnx', mb: 88, f16: true },
    gpuF32: { url: HF + 'imgly/isnet-general-onnx/resolve/main/onnx/model.onnx', mb: 176 },
    cpu: { url: HF + 'xrds/isnet-general-onnx-int8/resolve/main/onnx/model_quantized.onnx', mb: 44 },
  },
  best: {
    label: 'Best', about: 'BiRefNet lite, sharpest hair and detail (needs WebGPU)', size: 1024, ...IMAGENET, minmax: true,
    gpu: { url: HF + 'onnx-community/BiRefNet_lite-ONNX/resolve/main/onnx/model_fp16.onnx', mb: 115, f16: true },
    gpuF32: { url: HF + 'onnx-community/BiRefNet_lite-ONNX/resolve/main/onnx/model.onnx', mb: 224 },
  },
  portrait: {
    label: 'Portrait', about: 'MODNet, made for people and hair', size: 0,
    mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5], scale: 1 / 255, minmax: false,
    gpu: { url: HF + 'Xenova/modnet/resolve/main/onnx/model_fp16.onnx', mb: 13, f16: true },
    gpuF32: { url: HF + 'Xenova/modnet/resolve/main/onnx/model.onnx', mb: 26 },
    cpu: { url: HF + 'Xenova/modnet/resolve/main/onnx/model_quantized.onnx', mb: 6.6 },
  },
};

const MIME = { png: 'image/png', jpeg: 'image/jpeg', webp: 'image/webp', avif: 'image/avif', bmp: 'image/bmp', tga: 'image/x-tga', qoi: 'image/qoi', ppm: 'image/x-portable-pixmap' };
const EXT = { png: 'png', jpeg: 'jpg', webp: 'webp', avif: 'avif', bmp: 'bmp', tga: 'tga', qoi: 'qoi', ppm: 'ppm' };

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const el = {
  stage: $('#stage'), drop: $('#drop'), view: $('#view'), canvas: $('#canvas'), crop: $('#crop'),
  busy: $('#busy'), busyText: $('#busy-text'), busyBar: $('#busy-bar'), info: $('#info'), engine: $('#engine'),
  toast: $('#toast'), name: $('#name'), fmt: $('#fmt'), quality: $('#quality'), scale: $('#scale'),
};
const ctx = el.canvas.getContext('2d');

const state = {
  lib: null,
  image: null,       // full-resolution working image
  original: null,    // as opened, for Reset / Compare
  preview: null,     // screen-sized copy of image
  live: null,        // preview with pending slider edits
  history: [],
  future: [],
  cutout: null,      // background-removal session that can still be re-tuned
  bgImage: null,     // replacement background
  model: 'balanced',
  gpu: null,         // null = unknown, false = none, { f16 } = WebGPU
  gpuReady: null,
  sessions: new Map(),
  actual: false,     // 100% zoom
  cropBox: null,
  name: 'image',
};

// ---- small UI helpers -------------------------------------------------------

let toastTimer;
function toast(msg, error = false) {
  el.toast.textContent = msg;
  el.toast.classList.toggle('error', error);
  el.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.toast.hidden = true), error ? 7000 : 3500);
}

const frame = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
let busyDepth = 0;

// Show the busy indicator, give the browser a frame to paint it, then run fn.
async function busy(text, fn) {
  busyDepth++;
  el.busyText.textContent = text;
  el.busyBar.style.width = '0';
  el.busy.hidden = false;
  try {
    await frame();
    return await fn();
  } catch (e) {
    console.error(e);
    toast(e?.message || String(e), true);
  } finally {
    if (--busyDepth === 0) el.busy.hidden = true;
  }
}

function progress(fraction, text) {
  if (text) el.busyText.textContent = text;
  el.busyBar.style.width = `${Math.round(Math.max(0, Math.min(1, fraction)) * 100)}%`;
}

const fmtMs = (ms) => (ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`);

// ---- rendering -----------------------------------------------------------------

function cssScale() {
  if (state.actual) return 1;
  const r = el.view.getBoundingClientRect();
  return Math.min(1, (r.width - 32) / state.image.width, (r.height - 32) / state.image.height);
}

function buildPreview() {
  state.preview?.free();
  state.preview = null;
  const img = state.image;
  const s = Math.min(1, cssScale() * Math.min(devicePixelRatio || 1, 2));
  if (s >= 1 || state.actual) {
    state.preview = img.clone();
  } else {
    state.preview = img.clone().resize(Math.max(1, Math.round(img.width * s)), Math.max(1, Math.round(img.height * s)));
  }
}

function draw(img) {
  const c = el.canvas;
  if (c.width !== img.width || c.height !== img.height) {
    c.width = img.width;
    c.height = img.height;
  }
  ctx.putImageData(img.toImageData(), 0, 0);
  const s = cssScale();
  c.style.width = `${Math.round(state.image.width * s)}px`;
  c.style.height = `${Math.round(state.image.height * s)}px`;
}

function refresh() {
  buildPreview();
  runLive(true);
  updateInfo();
  updateButtons();
  el.rw.value = state.image.width;
  el.rh.value = state.image.height;
  if (state.cropBox) layoutCrop();
}

function updateInfo(extra) {
  const img = state.image;
  if (!img) return (el.info.textContent = 'No image');
  const last = state.history.at(-1)?.label;
  el.info.textContent = `${img.width} × ${img.height}${img.hasAlpha() ? ' · transparent' : ''}${extra ? ` · ${extra}` : last ? ` · ${last}` : ''}`;
}

function updateEngine() {
  const gpu = state.gpu === null ? 'checking…' : state.gpu ? `WebGPU${state.gpu.f16 ? ' (fp16)' : ''}` : 'CPU (no WebGPU)';
  el.engine.textContent = `Models: ${gpu}`;
}

function updateButtons() {
  const has = !!state.image;
  $$('[data-invert],[data-filter],[data-tf],#remove-bg,#key-bg,#adj-apply,#live-apply,#crop-start,#resize,#download,#copy,#compare,#reset,#zoom')
    .forEach((b) => (b.disabled = !has));
  $('#undo').disabled = !state.history.length;
  $('#redo').disabled = !state.future.length;
  $('#zoom').textContent = state.actual ? '100%' : 'Fit';
  if (has) renderModels();
}

// ---- documents and history ------------------------------------------------------

function freeList(list) {
  for (const h of list) h.image.free();
  list.length = 0;
}

function setDocument(img, name) {
  endCutout();
  freeList(state.history);
  freeList(state.future);
  state.original?.free();
  state.image?.free();
  state.image = img;
  state.original = img.clone();
  state.name = name;
  el.name.value = name;
  resetSliders();
  cancelCrop();
  el.drop.hidden = true;
  el.view.hidden = false;
  refresh();
}

function pushHistory(label) {
  state.history.push({ image: state.image, label });
  freeList(state.future);
  // Keep history within ~600 MB of pixels and 40 steps.
  let bytes = 0;
  for (let i = state.history.length - 1; i >= 0; i--) {
    const h = state.history[i];
    bytes += h.image.width * h.image.height * 4;
    if (bytes > 600e6 || state.history.length - i > 40) {
      for (const old of state.history.splice(0, i + 1)) old.image.free();
      break;
    }
  }
}

// Apply fn to a copy of the full image and make that the new state.
async function commit(label, fn) {
  if (!state.image) return;
  const run = () => {
    const next = state.image.clone();
    try {
      fn(next);
    } catch (e) {
      next.free();
      throw e;
    }
    endCutout();
    pushHistory(label);
    state.image = next;
    refresh();
  };
  if (state.image.width * state.image.height > 3e6) await busy(`${label}…`, run);
  else {
    try {
      run();
    } catch (e) {
      toast(e.message, true);
    }
  }
}

function undo() {
  if (!state.history.length) return;
  endCutout();
  const h = state.history.pop();
  state.future.push({ image: state.image, label: h.label });
  state.image = h.image;
  refresh();
}

function redo() {
  if (!state.future.length) return;
  endCutout();
  const f = state.future.pop();
  state.history.push({ image: state.image, label: f.label });
  state.image = f.image;
  refresh();
}

function resetToOriginal() {
  if (!state.original) return;
  endCutout();
  pushHistory('Reset');
  state.image = state.original.clone();
  resetSliders();
  refresh();
}

// ---- opening images --------------------------------------------------------------

async function decodeFile(file) {
  const buf = new Uint8Array(await file.arrayBuffer());
  const jpeg = buf[0] === 0xff && buf[1] === 0xd8;
  // The wasm decoder keeps exact alpha; JPEGs go to the browser, which is
  // faster at them and applies colour profiles. Anything else it cannot read
  // (WebP, AVIF, ...) also goes to the browser.
  if (!jpeg) {
    try {
      return state.lib.decode(buf);
    } catch {
      /* fall through */
    }
  }
  let bmp;
  try {
    bmp = await createImageBitmap(new Blob([buf], { type: file.type || '' }), { imageOrientation: 'from-image', premultiplyAlpha: 'none' });
  } catch {
    throw new Error(`Cannot read ${file.name || 'this file'}: unsupported or damaged image.`);
  }
  const c = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(bmp.width, bmp.height) : Object.assign(document.createElement('canvas'), { width: bmp.width, height: bmp.height });
  const cx = c.getContext('2d', { willReadFrequently: true });
  cx.drawImage(bmp, 0, 0);
  bmp.close?.();
  return state.lib.fromImageData(cx.getImageData(0, 0, c.width, c.height));
}

function isImageFile(f) {
  return f && (f.type.startsWith('image/') || /\.(qoi|tga|psd|hdr|ppm|pgm|pnm|pam)$/i.test(f.name));
}

async function openFiles(files) {
  const file = [...files].find(isImageFile);
  if (!file) return toast('That does not look like an image.', true);
  await busy('Opening…', async () => {
    const t = performance.now();
    const img = await decodeFile(file);
    setDocument(img, file.name.replace(/\.[^.]+$/, '') || 'image');
    updateInfo(`opened in ${fmtMs(performance.now() - t)}`);
  });
}

// ---- live previews (sliders) ---------------------------------------------------------

const adjustDefaults = { exposure: 0, brightness: 0, contrast: 0, saturation: 0, hue: 0, temperature: 0, tint: 0, gamma: 1 };
const liveDefaults = { blur: 0, sharpen: 0, posterize: 0, threshold: 0, rotate: 0 };

function readAdjust() {
  const a = {};
  for (const input of $$('[data-adj]')) a[input.dataset.adj] = +input.value;
  return a;
}

function readLive() {
  const v = {};
  for (const input of $$('[data-live]')) v[input.dataset.live] = +input.value;
  return v;
}

const adjustIsDefault = (a) => Object.keys(adjustDefaults).every((k) => a[k] === adjustDefaults[k]);
const liveIsDefault = (v) => Object.keys(liveDefaults).every((k) => v[k] === liveDefaults[k]);

function applyAdjust(img, a) {
  if (!adjustIsDefault(a)) img.adjust(a);
}

// s: preview scale, so pixel-sized effects look the same as on the full image.
function applyLive(img, v, s = 1) {
  if (v.blur > 0) img.blur(v.blur * s);
  if (v.sharpen > 0) img.sharpen(v.sharpen, Math.max(0.6, s), 2);
  if (v.posterize >= 2) img.posterize(v.posterize);
  if (v.threshold > 0) img.threshold(v.threshold);
  if (v.rotate) img.rotate(v.rotate);
}

let livePending = false;
function runLive(now = false) {
  if (!state.preview) return;
  if (!now) {
    if (!livePending) {
      livePending = true;
      requestAnimationFrame(() => {
        livePending = false;
        runLive(true);
      });
    }
    return;
  }
  state.live?.free();
  state.live = null;
  const a = readAdjust(), v = readLive();
  if (!adjustIsDefault(a) || !liveIsDefault(v)) {
    state.live = state.preview.clone();
    try {
      applyAdjust(state.live, a);
      applyLive(state.live, v, state.preview.width / state.image.width);
    } catch (e) {
      toast(e.message, true);
    }
  }
  draw(state.live ?? state.preview);
  $('#adj-apply').disabled = adjustIsDefault(a);
  $('#live-apply').disabled = liveIsDefault(v);
}

function syncOutputs() {
  for (const input of $$('input[type=range]')) {
    const out = input.parentElement.querySelector('output');
    if (!out) continue;
    const v = +input.value;
    out.textContent = input.dataset.adj === 'hue' || input.dataset.live === 'rotate' ? `${v}°` : Number.isInteger(v) ? v : v.toFixed(2);
  }
}

function resetSliders(which) {
  if (!which || which === 'adjust') for (const i of $$('[data-adj]')) i.value = adjustDefaults[i.dataset.adj];
  if (!which || which === 'live') for (const i of $$('[data-live]')) i.value = liveDefaults[i.dataset.live];
  syncOutputs();
}

// ---- background removal ---------------------------------------------------------------

function renderModels() {
  const box = $('#models');
  if (!box.children.length) {
    for (const [key, m] of Object.entries(MODELS)) {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = m.label;
      b.dataset.model = key;
      b.setAttribute('role', 'radio');
      b.addEventListener('click', () => {
        state.model = key;
        renderModels();
      });
      box.append(b);
    }
  }
  for (const b of box.children) {
    const m = MODELS[b.dataset.model];
    b.setAttribute('aria-checked', String(b.dataset.model === state.model));
    b.disabled = state.gpu === false && !m.cpu;
    b.title = b.disabled ? 'Needs WebGPU' : m.about;
  }
  if (state.gpu === false && !MODELS[state.model].cpu) state.model = 'balanced';
  const m = MODELS[state.model];
  const v = variant(m);
  $('#model-hint').textContent = v ? `${m.about} · ${v.mb} MB download, cached after the first use · runs on ${state.gpu ? 'the GPU' : 'the CPU'}` : m.about;
}

function variant(m) {
  if (state.gpu) return m.gpu.f16 && !state.gpu.f16 ? m.gpuF32 ?? m.cpu : m.gpu;
  return m.cpu ?? null;
}

function detectGpu() {
  state.gpuReady ??= (async () => {
    try {
      const adapter = navigator.gpu && (await navigator.gpu.requestAdapter());
      state.gpu = adapter ? { f16: adapter.features.has('shader-f16') } : false;
    } catch {
      state.gpu = false;
    }
    updateEngine();
    renderModels();
  })();
  return state.gpuReady;
}

let ortPromise;
function loadOrt() {
  ortPromise ??= import(/* @vite-ignore */ `${ORT_DIR}ort.webgpu.min.mjs`).then((ort) => {
    ort.env.wasm.wasmPaths = ORT_DIR;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
    ort.env.logLevel = 'error';
    return ort;
  });
  return ortPromise;
}

// Download a model with progress, keeping a copy in the Cache API.
async function fetchModel(url, mb, label) {
  let cache = null;
  try {
    cache = await caches.open('imgedit-models-v1');
    const hit = await cache.match(url);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch {
    cache = null;
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Model download failed (${res.status}).`);
  const total = +res.headers.get('content-length') || mb * 1e6;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    progress(got / total, `Downloading ${label} model… ${(got / 1e6).toFixed(0)} / ${Math.round(total / 1e6)} MB`);
  }
  const blob = new Blob(chunks);
  cache?.put(url, new Response(blob, { headers: { 'content-type': 'application/octet-stream' } })).catch(() => {});
  return new Uint8Array(await blob.arrayBuffer());
}

async function getSession(key) {
  await detectGpu();
  const m = MODELS[key];
  const v = variant(m);
  if (!v) throw new Error(`${m.label} needs WebGPU, which this browser does not provide. Try Balanced.`);
  const id = `${key}:${v.url}`;
  if (state.sessions.has(id)) return state.sessions.get(id);
  for (const s of state.sessions.values()) s.session.release?.();
  state.sessions.clear();

  progress(0, 'Loading the model runtime…');
  const ort = await loadOrt();
  const bytes = await fetchModel(v.url, v.mb, m.label);
  progress(1, 'Preparing the model…');
  await frame();
  let ep = state.gpu ? 'webgpu' : 'wasm';
  let session;
  try {
    session = await ort.InferenceSession.create(bytes, { executionProviders: [ep], graphOptimizationLevel: 'all' });
  } catch (e) {
    if (ep !== 'webgpu' || !m.cpu) throw e;
    console.warn('WebGPU session failed, using the CPU', e);
    ep = 'wasm';
    const cpuBytes = m.cpu.url === v.url ? bytes : await fetchModel(m.cpu.url, m.cpu.mb, m.label);
    session = await ort.InferenceSession.create(cpuBytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  }
  const entry = { session, ep, ort };
  state.sessions.set(id, entry);
  return entry;
}

const round32 = (v) => Math.max(32, Math.round(v / 32) * 32);

// Run a model on img; returns the raw matte at model resolution.
async function predict(key, img) {
  const m = MODELS[key];
  const { session, ep, ort } = await getSession(key);
  let tw = m.size, th = m.size;
  if (!m.size) {
    const s = 512 / Math.min(img.width, img.height);
    tw = Math.min(2048, round32(img.width * s));
    th = Math.min(2048, round32(img.height * s));
  }
  progress(1, `Finding the subject (${ep === 'webgpu' ? 'GPU' : 'CPU'})…`);
  await frame();
  const input = new ort.Tensor('float32', img.nnInput(tw, th, m.mean, m.std, m.scale), [1, 3, th, tw]);
  const outName = session.outputNames[0];
  const result = await session.run({ [session.inputNames[0]]: input }, [outName]);
  const out = result[outName];
  const [mh, mw] = out.dims.slice(-2);
  let data = out.type === 'float16' ? (out.data instanceof Uint16Array ? fromHalf(state.lib, out.data) : Float32Array.from(out.data)) : out.data;
  data = new Float32Array(data.subarray(0, mw * mh));
  input.dispose?.();
  out.dispose?.();
  return { matte: data, mw, mh, ep, minmax: m.minmax };
}

function cutoutSettings() {
  return {
    refine: $('#refine').checked,
    fix: $('#fix-edges').checked,
    keep: $('#keep-bg').checked,
    grow: +$('#grow').value,
    feather: +$('#feather').value,
    tolerance: +$('#tolerance').value,
    mode: $('#bg-mode').value,
    color: parseInt($('#bg-color').value.slice(1), 16),
    blur: +$('#bg-blur').value,
  };
}

// Builds the cut-out from the session's source image with the current settings.
function renderCutout(c, s, maskOnly = false) {
  const img = c.source.clone();
  try {
    if (c.kind === 'model') img.applyMatte(c.matte, c.mw, c.mh, { minmax: c.minmax, radius: s.refine ? 4 : 0 });
    else img.keyBackground({ tolerance: s.tolerance, softness: 8 });
    if (s.grow) img.alphaGrow(s.grow);
    if (s.feather > 0) img.alphaFeather(s.feather);
    if (s.keep) img.invert('alpha');
    if (maskOnly) img.alphaToGray();
    else {
      const mode = s.mode === 'image' && !state.bgImage ? 'transparent' : s.mode;
      const scale = Math.max(img.width, img.height) / 1000;
      img.finishCutout({ mode, color: s.color, blur: s.blur * scale, image: state.bgImage, decontaminate: s.fix });
    }
    return img;
  } catch (e) {
    img.free();
    throw e;
  }
}

async function applyCutout(first, maskOnly = false) {
  const c = state.cutout;
  if (!c) return;
  const run = () => {
    const img = renderCutout(c, cutoutSettings(), maskOnly);
    if (first) pushHistory(c.kind === 'model' ? 'Background removed' : 'Plain background removed');
    else state.image.free();
    state.image = img;
    refresh();
  };
  if (c.source.width * c.source.height > 2e6) await busy('Updating…', run);
  else {
    try {
      run();
    } catch (e) {
      toast(e.message, true);
    }
  }
}

function startCutout(c) {
  endCutout();
  state.cutout = c;
  $('#cutout-opts').hidden = false;
  $('#tol-row').hidden = c.kind !== 'key';
  $('#refine').parentElement.hidden = c.kind !== 'model';
}

function endCutout() {
  if (!state.cutout) return;
  state.cutout.source.free();
  state.cutout = null;
  $('#cutout-opts').hidden = true;
}

async function removeBackground() {
  if (!state.image) return;
  const key = state.model;
  await busy('Removing background…', async () => {
    const t = performance.now();
    const src = state.image.clone();
    try {
      const r = await predict(key, src);
      startCutout({ kind: 'model', source: src, ...r });
    } catch (e) {
      src.free();
      throw e;
    }
    await applyCutout(true);
    updateInfo(`background removed in ${fmtMs(performance.now() - t)} on ${state.cutout?.ep === 'webgpu' ? 'GPU' : 'CPU'}`);
  });
}

async function keyBackground() {
  if (!state.image) return;
  startCutout({ kind: 'key', source: state.image.clone() });
  await applyCutout(true);
}

// ---- crop tool ---------------------------------------------------------------------

function canvasRectInView() {
  const v = el.view.getBoundingClientRect(), c = el.canvas.getBoundingClientRect();
  return { x: c.left - v.left + el.view.scrollLeft, y: c.top - v.top + el.view.scrollTop, w: c.width, h: c.height };
}

function startCrop() {
  if (!state.image) return;
  const r = canvasRectInView();
  state.cropBox = { x: r.w * 0.1, y: r.h * 0.1, w: r.w * 0.8, h: r.h * 0.8 };
  applyRatio();
  el.crop.hidden = false;
  $('#crop-bar').hidden = false;
  layoutCrop();
}

function cancelCrop() {
  state.cropBox = null;
  el.crop.hidden = true;
  $('#crop-bar').hidden = true;
}

function layoutCrop() {
  const r = canvasRectInView(), b = state.cropBox, box = el.crop.firstElementChild;
  Object.assign(box.style, { left: `${r.x + b.x}px`, top: `${r.y + b.y}px`, width: `${b.w}px`, height: `${b.h}px` });
}

function applyRatio() {
  const ratio = +$('#crop-ratio').value, b = state.cropBox;
  if (!ratio || !b) return;
  const r = canvasRectInView();
  // Ratio is in image pixels; convert through the display scale.
  const sx = r.w / state.image.width, sy = r.h / state.image.height;
  let w = b.w, h = (w / sx / ratio) * sy;
  if (h > r.h) {
    h = r.h;
    w = (h / sy) * ratio * sx;
  }
  b.x = Math.min(Math.max(0, b.x + (b.w - w) / 2), r.w - w);
  b.y = Math.min(Math.max(0, b.y + (b.h - h) / 2), r.h - h);
  b.w = w;
  b.h = h;
  layoutCrop();
}

function setupCropDrag() {
  const box = el.crop.firstElementChild;
  let drag = null;
  box.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    box.setPointerCapture(e.pointerId);
    drag = { mode: e.target.dataset.h || 'move', x: e.clientX, y: e.clientY, start: { ...state.cropBox } };
  });
  box.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const r = canvasRectInView(), s = drag.start, b = state.cropBox;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y, min = 12;
    if (drag.mode === 'move') {
      b.x = Math.min(Math.max(0, s.x + dx), r.w - s.w);
      b.y = Math.min(Math.max(0, s.y + dy), r.h - s.h);
    } else {
      let x0 = s.x, y0 = s.y, x1 = s.x + s.w, y1 = s.y + s.h;
      if (drag.mode.includes('w')) x0 = Math.min(Math.max(0, s.x + dx), x1 - min);
      if (drag.mode.includes('e')) x1 = Math.max(Math.min(r.w, x1 + dx), x0 + min);
      if (drag.mode.includes('n')) y0 = Math.min(Math.max(0, s.y + dy), y1 - min);
      if (drag.mode.includes('s')) y1 = Math.max(Math.min(r.h, y1 + dy), y0 + min);
      const ratio = +$('#crop-ratio').value;
      if (ratio) { /* keep the ratio by adjusting the height to the width */
        const sx = r.w / state.image.width, sy = r.h / state.image.height;
        const h = ((x1 - x0) / sx / ratio) * sy;
        if (drag.mode.includes('n')) y0 = Math.max(0, y1 - h);
        else y1 = Math.min(r.h, y0 + h);
      }
      Object.assign(b, { x: x0, y: y0, w: x1 - x0, h: y1 - y0 });
    }
    layoutCrop();
  });
  const end = () => (drag = null);
  box.addEventListener('pointerup', end);
  box.addEventListener('pointercancel', end);
}

function applyCrop() {
  const b = state.cropBox;
  if (!b) return;
  const r = canvasRectInView();
  const sx = state.image.width / r.w, sy = state.image.height / r.h;
  const x = Math.round(b.x * sx), y = Math.round(b.y * sy), w = Math.round(b.w * sx), h = Math.round(b.h * sy);
  cancelCrop();
  commit('Crop', (img) => img.crop(x, y, Math.max(1, w), Math.max(1, h)));
}

// ---- export --------------------------------------------------------------------------

async function canvasEncode(img, type, quality) {
  const c = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(img.width, img.height) : Object.assign(document.createElement('canvas'), { width: img.width, height: img.height });
  c.getContext('2d').putImageData(img.toImageData(), 0, 0);
  const blob = c.convertToBlob ? await c.convertToBlob({ type, quality }) : await new Promise((r) => c.toBlob(r, type, quality));
  if (!blob || blob.type !== type) throw new Error(`This browser cannot save ${type.split('/')[1].toUpperCase()}; try PNG or JPEG.`);
  return blob;
}

async function encodeImage(img, fmt, quality, scale = 1) {
  let tmp = null;
  if (scale < 1) {
    tmp = img.clone().resize(Math.max(1, Math.round(img.width * scale)), Math.max(1, Math.round(img.height * scale)));
    img = tmp;
  }
  try {
    if (fmt === 'webp' || fmt === 'avif') return await canvasEncode(img, MIME[fmt], quality / 100);
    return new Blob([img.encode(fmt, fmt === 'jpeg' ? quality : 0)], { type: MIME[fmt] });
  } finally {
    tmp?.free();
  }
}

function saveBlob(blob, filename) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 30000);
}

async function download() {
  if (!state.image) return;
  const fmt = el.fmt.value;
  await busy('Saving…', async () => {
    const t = performance.now();
    const blob = await encodeImage(state.image, fmt, +el.quality.value, +el.scale.value);
    const name = `${(el.name.value || 'image').replace(/[\\/:*?"<>|]+/g, '_')}.${EXT[fmt]}`;
    saveBlob(blob, name);
    $('#export-hint').textContent = `${name} · ${(blob.size / 1024).toFixed(0)} KB · encoded in ${fmtMs(performance.now() - t)}`;
  });
}

async function copyImage() {
  if (!state.image) return;
  try {
    const blob = await encodeImage(state.image, 'png', 0, +el.scale.value);
    await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
    toast('Copied to the clipboard as PNG.');
  } catch (e) {
    toast(`Copy failed: ${e.message}`, true);
  }
}

// ---- batch ---------------------------------------------------------------------------

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[i] = c >>> 0;
  }
  return t;
})();

function crc32(data) {
  let c = 0xffffffff;
  for (let i = 0; i < data.length; i++) c = CRC_TABLE[(c ^ data[i]) & 255] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

// Minimal ZIP writer (stored entries: images are already compressed).
function makeZip(files) {
  const parts = [], central = [];
  let offset = 0;
  const now = new Date();
  const time = (now.getHours() << 11) | (now.getMinutes() << 5) | (now.getSeconds() >> 1);
  const date = ((now.getFullYear() - 1980) << 9) | ((now.getMonth() + 1) << 5) | now.getDate();
  for (const { name, data } of files) {
    const n = new TextEncoder().encode(name), crc = crc32(data);
    const h = new DataView(new ArrayBuffer(30));
    h.setUint32(0, 0x04034b50, true); h.setUint16(4, 20, true); h.setUint16(6, 0x0800, true); h.setUint16(8, 0, true);
    h.setUint16(10, time, true); h.setUint16(12, date, true); h.setUint32(14, crc, true);
    h.setUint32(18, data.length, true); h.setUint32(22, data.length, true); h.setUint16(26, n.length, true); h.setUint16(28, 0, true);
    parts.push(h, n, data);
    const c = new DataView(new ArrayBuffer(46));
    c.setUint32(0, 0x02014b50, true); c.setUint16(4, 20, true); c.setUint16(6, 20, true); c.setUint16(8, 0x0800, true);
    c.setUint16(10, 0, true); c.setUint16(12, time, true); c.setUint16(14, date, true); c.setUint32(16, crc, true);
    c.setUint32(20, data.length, true); c.setUint32(24, data.length, true); c.setUint16(28, n.length, true);
    c.setUint16(30, 0, true); c.setUint16(32, 0, true); c.setUint16(34, 0, true); c.setUint16(36, 0, true);
    c.setUint32(38, 0, true); c.setUint32(42, offset, true);
    central.push(c, n);
    offset += 30 + n.length + data.length;
  }
  const size = central.reduce((s, p) => s + p.byteLength, 0);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true); end.setUint16(8, files.length, true); end.setUint16(10, files.length, true);
  end.setUint32(12, size, true); end.setUint32(16, offset, true);
  return new Blob([...parts, ...central, end], { type: 'application/zip' });
}

async function runBatch(fileList) {
  const files = [...fileList].filter(isImageFile);
  if (!files.length) return;
  const remove = $('#b-remove').checked, trim = $('#b-trim').checked, max = +$('#b-max').value || 0, fmt = $('#b-fmt').value;
  const status = $('#b-status');
  const out = [], used = new Set();
  let failed = 0;
  const t0 = performance.now();
  await busy('Batch…', async () => {
    for (const [i, f] of files.entries()) {
      progress(i / files.length, `Batch ${i + 1} / ${files.length}: ${f.name}`);
      status.textContent = el.busyText.textContent;
      let img = null;
      try {
        img = await decodeFile(f);
        if (remove) {
          const r = await predict(state.model, img);
          img.applyMatte(r.matte, r.mw, r.mh, { minmax: r.minmax }).finishCutout({});
        }
        if (trim) img.trim(8);
        if (max > 0 && Math.max(img.width, img.height) > max) {
          const s = max / Math.max(img.width, img.height);
          img.resize(img.width * s, img.height * s);
        }
        const blob = await encodeImage(img, fmt, 90);
        let name = `${f.name.replace(/\.[^.]+$/, '')}.${EXT[fmt]}`;
        for (let k = 2; used.has(name); k++) name = `${f.name.replace(/\.[^.]+$/, '')}-${k}.${EXT[fmt]}`;
        used.add(name);
        out.push({ name, data: new Uint8Array(await blob.arrayBuffer()) });
      } catch (e) {
        console.error(f.name, e);
        failed++;
      } finally {
        img?.free();
      }
    }
    if (out.length) saveBlob(makeZip(out), 'imgedit-batch.zip');
  });
  status.textContent = `${out.length} done${failed ? `, ${failed} failed` : ''} in ${fmtMs(performance.now() - t0)}.`;
}

// ---- wiring ---------------------------------------------------------------------------

function setupUI() {
  const pick = (input) => input.addEventListener('change', () => {
    if (input.files.length) openFiles(input.files);
    input.value = '';
  });
  pick($('#file'));
  pick($('#file2'));

  for (const t of [el.stage, el.drop]) {
    t.addEventListener('dragover', (e) => {
      e.preventDefault();
      el.stage.classList.add('over');
      el.drop.classList.add('over');
    });
    t.addEventListener('dragleave', () => {
      el.stage.classList.remove('over');
      el.drop.classList.remove('over');
    });
    t.addEventListener('drop', (e) => {
      e.preventDefault();
      el.stage.classList.remove('over');
      el.drop.classList.remove('over');
      if (e.dataTransfer.files.length) openFiles(e.dataTransfer.files);
    });
  }
  document.addEventListener('paste', (e) => {
    const files = [...(e.clipboardData?.files || [])];
    if (files.length && !e.target.closest?.('input[type=text],input[type=number]')) {
      e.preventDefault();
      openFiles(files);
    }
  });
  $('#sample').addEventListener('click', () => busy('Loading sample…', async () => {
    const res = await fetch('sample.jpg');
    if (!res.ok) throw new Error('Sample image not found.');
    const img = await decodeFile(new File([await res.blob()], 'sample.jpg', { type: 'image/jpeg' }));
    setDocument(img, 'sample');
  }));

  $('#undo').addEventListener('click', undo);
  $('#redo').addEventListener('click', redo);
  $('#reset').addEventListener('click', resetToOriginal);
  $('#zoom').addEventListener('click', () => {
    state.actual = !state.actual;
    el.view.classList.toggle('actual', state.actual);
    cancelCrop();
    if (state.image) refresh();
  });

  // Compare: hold the button (or C) to see the original.
  let comparing = false;
  const showOriginal = (on) => {
    if (!state.original || on === comparing) return;
    comparing = on;
    if (on) {
      const s = state.preview.width / state.image.width;
      const o = state.original.clone().resize(Math.max(1, Math.round(state.original.width * s)), Math.max(1, Math.round(state.original.height * s)));
      draw(o);
      el.canvas.style.width = `${Math.round(state.original.width * cssScale())}px`;
      el.canvas.style.height = `${Math.round(state.original.height * cssScale())}px`;
      o.free();
    } else runLive(true);
  };
  const cmp = $('#compare');
  cmp.addEventListener('pointerdown', () => showOriginal(true));
  for (const ev of ['pointerup', 'pointerleave', 'pointercancel']) cmp.addEventListener(ev, () => showOriginal(false));

  // Background
  $('#remove-bg').addEventListener('click', removeBackground);
  $('#key-bg').addEventListener('click', keyBackground);
  let cutoutTimer;
  const retune = () => {
    clearTimeout(cutoutTimer);
    cutoutTimer = setTimeout(() => applyCutout(false), 60);
  };
  for (const id of ['#grow', '#feather', '#tolerance', '#bg-blur']) $(id).addEventListener('input', () => { syncOutputs(); retune(); });
  for (const id of ['#refine', '#fix-edges', '#keep-bg', '#bg-color']) $(id).addEventListener('change', retune);
  $('#bg-color').addEventListener('input', retune);
  $('#bg-mode').addEventListener('change', () => {
    const mode = $('#bg-mode').value;
    $('#bg-color-row').hidden = mode !== 'color';
    $('#bg-blur-row').hidden = mode !== 'blur';
    if (mode === 'image') $('#bg-file').click();
    else retune();
  });
  $('#bg-file').addEventListener('change', async () => {
    const f = $('#bg-file').files[0];
    $('#bg-file').value = '';
    if (!f) return;
    try {
      state.bgImage?.free();
      state.bgImage = await decodeFile(f);
      retune();
    } catch (e) {
      toast(e.message, true);
    }
  });
  $('#mask-only').addEventListener('click', () => applyCutout(false, true));

  // Invert, filters, transforms
  for (const b of $$('[data-invert]')) {
    const labels = { rgb: 'Invert colours', lightness: 'Invert lightness', alpha: 'Invert transparency' };
    b.addEventListener('click', () => commit(labels[b.dataset.invert], (img) => img.invert(b.dataset.invert)));
  }
  const filters = {
    'auto-levels': ['Auto levels', (img) => img.autoLevels(false)],
    'auto-color': ['Auto colour', (img) => img.autoLevels(true)],
    grayscale: ['Grayscale', (img) => img.grayscale()],
    sepia: ['Sepia', (img) => img.sepia(1)],
  };
  for (const b of $$('[data-filter]')) b.addEventListener('click', () => commit(...filters[b.dataset.filter]));
  const transforms = {
    rotl: ['Rotate left', (img) => img.rotate90(3)],
    rotr: ['Rotate right', (img) => img.rotate90(1)],
    fliph: ['Flip horizontal', (img) => img.flip(true)],
    flipv: ['Flip vertical', (img) => img.flip(false)],
    trim: ['Trim', (img) => img.trim(8)],
  };
  for (const b of $$('[data-tf]')) b.addEventListener('click', () => { cancelCrop(); commit(...transforms[b.dataset.tf]); });

  // Sliders
  for (const input of $$('[data-adj],[data-live]')) input.addEventListener('input', () => { syncOutputs(); runLive(); });
  $('#adj-apply').addEventListener('click', () => {
    const a = readAdjust();
    resetSliders('adjust');
    commit('Adjust', (img) => applyAdjust(img, a));
  });
  $('#adj-reset').addEventListener('click', () => { resetSliders('adjust'); runLive(); });
  $('#live-apply').addEventListener('click', () => {
    const v = readLive();
    resetSliders('live');
    commit('Filter', (img) => applyLive(img, v, 1));
  });
  $('#live-reset').addEventListener('click', () => { resetSliders('live'); runLive(); });

  // Crop and resize
  setupCropDrag();
  $('#crop-start').addEventListener('click', startCrop);
  $('#crop-cancel').addEventListener('click', cancelCrop);
  $('#crop-apply').addEventListener('click', applyCrop);
  $('#crop-ratio').addEventListener('change', applyRatio);
  el.rw = $('#rw');
  el.rh = $('#rh');
  el.rw.addEventListener('input', () => {
    if ($('#rlock').checked && state.image) el.rh.value = Math.max(1, Math.round((+el.rw.value * state.image.height) / state.image.width));
  });
  el.rh.addEventListener('input', () => {
    if ($('#rlock').checked && state.image) el.rw.value = Math.max(1, Math.round((+el.rh.value * state.image.width) / state.image.height));
  });
  $('#resize').addEventListener('click', () => {
    const w = Math.round(+el.rw.value), h = Math.round(+el.rh.value);
    if (!(w > 0 && h > 0 && w <= 30000 && h <= 30000)) return toast('Enter a size between 1 and 30000 pixels.', true);
    commit(`Resize to ${w}×${h}`, (img) => img.resize(w, h));
  });

  // Export
  const syncFormat = () => ($('#q-row').hidden = !['jpeg', 'webp', 'avif'].includes(el.fmt.value));
  el.fmt.addEventListener('change', syncFormat);
  syncFormat();
  $('#download').addEventListener('click', download);
  $('#copy').addEventListener('click', copyImage);
  $('#b-files').addEventListener('change', () => {
    const files = $('#b-files').files;
    if (files.length) runBatch(files).finally(() => ($('#b-files').value = ''));
  });
  el.quality.addEventListener('input', syncOutputs);

  // Keyboard
  document.addEventListener('keydown', (e) => {
    const typing = e.target.closest?.('input[type=text],input[type=number],select');
    const mod = e.ctrlKey || e.metaKey;
    if (mod && e.key.toLowerCase() === 'z' && !typing) { e.preventDefault(); e.shiftKey ? redo() : undo(); }
    else if (mod && e.key.toLowerCase() === 'y' && !typing) { e.preventDefault(); redo(); }
    else if (mod && e.key.toLowerCase() === 'o') { e.preventDefault(); $('#file').click(); }
    else if (mod && e.key.toLowerCase() === 's') { e.preventDefault(); download(); }
    else if (e.key.toLowerCase() === 'c' && !mod && !typing && !e.repeat) showOriginal(true);
    else if (e.key === 'Escape') cancelCrop();
  });
  document.addEventListener('keyup', (e) => { if (e.key.toLowerCase() === 'c') showOriginal(false); });

  let resizeTimer;
  addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => state.image && refresh(), 120);
  });

  syncOutputs();
  renderModels();
  updateEngine();
  updateButtons();
}

async function main() {
  try {
    state.lib = await loadImgedit();
  } catch (e) {
    toast(`Could not start the image engine: ${e.message}`, true);
    return;
  }
  setupUI();
  detectGpu();
  window.imgedit = { state, MODELS }; // handy for debugging from the console
}

main();
