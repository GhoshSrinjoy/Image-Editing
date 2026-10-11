# Image-Editing · `imgedit`

Fast, light image editing in portable C: **background removal, inversion, adjustments, transforms and format conversion**. The same C core runs as a command-line tool and, compiled to WebAssembly, as a web page that works entirely in your browser.

**Use it in the browser: [ghoshsrinjoy.github.io/Image-Editing](https://ghoshsrinjoy.github.io/Image-Editing/)**
Nothing is uploaded: images are processed on your device.

- **One small C core.** C11, no required dependencies; 0.7 MB native binary, 138 KB (gzipped) WebAssembly module.
- **Background removal with modern models** (IS-Net, BiRefNet, U²-Net, MODNet) through ONNX Runtime, on the GPU when there is one (DirectML, CUDA, CoreML, WebGPU).
- **Edges that hold up.** The model's low-resolution mask is upsampled with a colour guided filter, so the edge follows the real photo, then edge colours are re-estimated (blur-fusion) so the old background doesn't leave a halo.
- **No-model option** for plain backgrounds (studio shots, scans): colour key with border connectivity and speckle cleanup.
- **Multithreaded throughout**, including a parallel PNG encoder.

## Speed

Measured on an Intel Core i7-13700K, NVIDIA RTX 3090, Windows 11.

| Task | Old Tkinter + rembg app | imgedit |
| --- | --- | --- |
| Remove background, 600×399 photo, same model (u2net, CPU) | 0.60 s | 0.61 s for the whole run (start, load model, cut out, save) |
| Remove background, 24 MP photo, same model (u2net, CPU) | 9.2 s | **0.81 s** for the whole run |
| Remove background, 36 MP portrait, default model (CPU) | – | 1.4 s for the whole run |
| Best model (BiRefNet lite), per image in a batch | – | 0.27 s on GPU (DirectML), ~4 s on CPU |

The old-app column replays its default settings (u2net, alpha matting, 10 px erosion) and leaves out Python start-up. Besides being slower, its erosion cut into hair and edges. imgedit keeps them, and its edge-colour cleanup removes the halo.

Editing a 24 MP photo with the CLI: JPEG decode 88 ms · invert 3 ms · all colour adjustments in one pass 7 ms · resize to 1600 px 27 ms · rotate 15° 78 ms · PNG save 172 ms.

## Web page

Open an image (drop, paste or pick), then:

- **Background:** remove it with a model (Fast, Balanced, Best, Portrait) or key out a plain background. Shrink/grow, feather or invert the mask, and replace the background with a colour, a blurred copy of the photo, or another image. All of these re-tune instantly without re-running the model.
- **Invert:** colours, lightness only (light ↔ dark, hues kept), or transparency.
- **Adjust:** exposure, brightness, contrast, saturation, hue, temperature, tint, gamma, with a live preview.
- **Filters:** auto levels, auto colour, grayscale, sepia, blur, sharpen, posterize, threshold.
- **Transform:** rotate, flip, straighten, crop (with aspect presets), resize, trim borders.
- **Export:** PNG, JPEG, WebP, AVIF, BMP, TGA, QOI, PPM; copy to clipboard; batch-process many files into a ZIP.

Undo/redo, hold **C** to compare with the original. Models download once from Hugging Face and are cached by the browser. WebGPU (current Chrome, Edge, Safari and Firefox on Windows) makes removal fast. Without it, Fast, Balanced and Portrait run on the CPU; Best needs WebGPU.

## Command line

```sh
imgedit photo.jpg -r -o cutout.png                    # remove the background
imgedit photo.jpg -r --bg white -o product.jpg        # cut out onto white
imgedit photo.jpg -r --bg blur -o portrait.jpg        # blur the background ("portrait mode")
imgedit shots/ -r -m best --trim -o cutouts/          # a whole folder with the best model
imgedit scan.png --key -o scan-cut.png                # plain background, no model
imgedit scan.png --invert lightness --auto-levels -o dark.png
imgedit photo.jpg --contrast 0.2 --saturation 0.3 --sharpen --resize 1600 -o out.jpg -q 85
imgedit *.png -f jpg -q 85 -o jpg/                    # convert
imgedit models                                        # list models and the runtime in use
```

Operations run in the order given, on every input; folders are expanded. Run `imgedit --help` for everything: `--remove-bg`, `--model`, `--key`, `--refine`, `--hard`, `--shrink/--grow`, `--feather`, `--clean`, `--keep-bg`, `--mask`, `--bg`, `--invert`, `--grayscale`, `--sepia`, `--brightness`, `--contrast`, `--exposure`, `--gamma`, `--saturation`, `--hue`, `--temperature`, `--tint`, `--auto-levels`, `--auto-color`, `--threshold`, `--posterize`, `--resize`, `--crop`, `--rotate`, `--flip`, `--trim`, `--pad`, `--blur`, `--sharpen`, `--flatten`, `-f`, `-q`, `-j`.

Formats: reads PNG, JPEG (EXIF rotation applied), BMP, GIF, TGA, PSD, HDR, PNM, PAM, QOI; writes PNG, JPEG, BMP, TGA, QOI, PPM, PGM, PAM.

### Models

Models download on first use into `~/.u2net` (the folder rembg uses, so existing downloads are reused). Set `--model-dir` or `IMGEDIT_MODELS` to change that.

| `-m` | Model | Download | Notes | License |
| --- | --- | --- | --- | --- |
| `fast` | U²-Netp | 5 MB | very quick, softer edges | Apache-2.0 |
| `balanced` (default) | IS-Net general | 179 MB | clean edges on most photos | Apache-2.0 |
| `best` | BiRefNet lite | 224 MB | sharpest hair and detail; use a GPU | MIT |
| `max` | BiRefNet | 973 MB | highest quality, slow without a GPU | MIT |
| `portrait` | MODNet | 26 MB | people, fine hair | Apache-2.0 |
| `anime`, `human`, `u2net`, `silueta`, `birefnet-portrait` | | | see `imgedit models` | |

Any `.onnx` matting model can be passed as `-m path/to/model.onnx`.

### ONNX Runtime

Model inference uses ONNX Runtime, **loaded at run time**: everything except model-based background removal works without it. imgedit looks for it next to the executable, then on the system. Set `IMGEDIT_ORT` to point at a specific copy.

- **Windows 11** already ships one (`System32\onnxruntime.dll`) with DirectML GPU support, so nothing needs installing. `./build.ps1 -Runtime` fetches a newer DirectML build if you want one.
- **Linux / macOS:** `make runtime` downloads it next to the binary. The release archives include it.

`--device auto` (default) uses the GPU for heavy models and batches, and the CPU for small jobs, where GPU set-up would cost more than it saves. Override with `--device cpu|gpu|cuda|dml|coreml`.

## Building

You need a C11 compiler. [Zig](https://ziglang.org) builds everything, including the WebAssembly module.

```powershell
./build.ps1            # Windows: CLI + web/imgedit.wasm + tests (downloads a portable Zig if needed)
```

```sh
make && make test      # Linux / macOS with cc/clang/gcc
make wasm              # web/imgedit.wasm (needs zig, or WASM_CC=<clang with a wasi sysroot>)
```

Try the web page locally by serving `web/` over HTTP, for example `python -m http.server -d web`.

MSVC works too: `cl /std:c11 /O2 src\*.c shell32.lib` minus `src\wasm.c`.

## How it works

```text
photo ─► resize + normalise (C) ─► model (ONNX Runtime / ONNX Runtime Web) ─► low-res matte
      ─► colour guided filter upsampling (C, full resolution, O(1) per pixel)
      ─► optional clean / shrink / feather / invert
      ─► blur-fusion foreground estimation on edge tiles (C) ─► new background or transparency
      ─► parallel PNG / JPEG / … encoder (C)
```

```text
src/          C core (imgedit.h is the public API) + CLI (cli.c) + wasm exports (wasm.c)
web/          the GitHub Pages app (index.html, app.js, imgedit.js wrapper)
tests/        C unit tests and WebAssembly tests
third_party/  stb_image, stb_image_write, stb_image_resize2, qoi, ONNX Runtime C API headers
```

Pushing to `main` builds the WebAssembly module and deploys the page to GitHub Pages. Pushing a `v*` tag publishes ready-to-run archives for Windows, Linux and macOS.

## License

MIT, see [LICENSE](LICENSE). Bundled and downloaded third-party components are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
