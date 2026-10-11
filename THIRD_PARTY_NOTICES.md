# Third-party notices

## Included in this repository

| Component | Files | License |
| --- | --- | --- |
| stb_image, stb_image_write, stb_image_resize2 (Sean Barrett, Jeff Roberts and contributors) | `third_party/stb/` | MIT or public domain (your choice); text at the end of each header |
| QOI image format (Dominic Szablewski) | `third_party/qoi/` | MIT, `third_party/qoi/LICENSE` |
| ONNX Runtime C API headers | `third_party/onnxruntime/` | MIT, `third_party/onnxruntime/LICENSE` |
| Sample photo "Domestic shorthair cat portrait in grass" (via Wikimedia Commons / Pixnio) | `web/sample.jpg` | CC0 |

## Downloaded at run time (not redistributed here)

- **ONNX Runtime / ONNX Runtime Web**, MIT. Loaded by the CLI when present; the web page loads it from the jsDelivr CDN.
- **Matting models**, fetched on first use:
  - U²-Net / U²-Netp: Xuebin Qin et al., Apache-2.0
  - IS-Net (DIS): Xuebin Qin et al., Apache-2.0
  - BiRefNet: Peng Zheng et al., MIT
  - MODNet: Zhanghan Ke et al., Apache-2.0

  ONNX files come from the rembg model releases and from these Hugging Face repositories: `BritishWerewolf/U-2-Netp`, `studioludens/birefnet-lite-512` (MIT), `jiabins0303/birefnet-lite-1024-webgpu` (MIT; BiRefNet-lite with its graph rewritten so it runs on WebGPU, weights unchanged), `Xenova/modnet`.

The foreground colour estimation follows M. Forte and F. Pitié, "Approximate Fast Foreground Colour Estimation", ICIP 2021. The matte upsampling uses the guided filter of K. He, J. Sun and X. Tang (and its fast variant by K. He and J. Sun).
