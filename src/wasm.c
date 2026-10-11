/*
 * WebAssembly exports for the browser editor. Images live in wasm memory and
 * are passed to JavaScript as pointers to ie_image ({int w, int h, uint8_t *px}).
 * Build: see build.sh / build.ps1 (target wasm32-wasi, reactor model).
 */
#include "imgedit.h"

#include <stdlib.h>
#include <string.h>

#define EXPORT(name) __attribute__((export_name(#name)))

static size_t last_len;
static int last_err;

EXPORT(alloc) void *w_alloc(size_t n) { return malloc(n ? n : 1); }
EXPORT(dealloc) void w_free(void *p) { free(p); }
EXPORT(last_len) size_t w_last_len(void) { return last_len; }
EXPORT(last_error) int w_last_error(void) { return last_err; }

/* ---- images ------------------------------------------------------------ */

EXPORT(img_new) ie_image *w_img_new(int w, int h) { return ie_new(w, h); }
EXPORT(img_clone) ie_image *w_img_clone(const ie_image *img) { return ie_clone(img); }
EXPORT(img_free) void w_img_free(ie_image *img) { ie_free(img); }
EXPORT(img_width) int w_img_width(const ie_image *img) { return img->w; }
EXPORT(img_height) int w_img_height(const ie_image *img) { return img->h; }
EXPORT(img_pixels) uint8_t *w_img_pixels(const ie_image *img) { return img->px; }
EXPORT(img_has_alpha) int w_img_has_alpha(const ie_image *img) { return ie_has_alpha(img); }

EXPORT(img_decode) ie_image *w_img_decode(const uint8_t *data, size_t len)
{
    ie_image *img = NULL;
    last_err = ie_decode(data, len, &img);
    return img;
}

/* Returns a malloc'd buffer (free with dealloc); its length is last_len(). */
EXPORT(img_encode) uint8_t *w_img_encode(const ie_image *img, int fmt, int quality)
{
    uint8_t *out = NULL;
    last_len = 0;
    last_err = ie_encode(img, fmt, quality, &out, &last_len);
    return out;
}

/* ---- operations (return 0 on success) ---------------------------------- */

EXPORT(invert) int w_invert(ie_image *img, int mode) { return ie_invert(img, mode); }
EXPORT(grayscale) int w_grayscale(ie_image *img) { return ie_grayscale(img); }
EXPORT(sepia) int w_sepia(ie_image *img, float amount) { return ie_sepia(img, amount); }
EXPORT(auto_levels) int w_auto_levels(ie_image *img, float clip, int per_channel)
{
    return ie_auto_levels(img, clip, per_channel);
}
EXPORT(threshold) int w_threshold(ie_image *img, int level) { return ie_threshold(img, level); }
EXPORT(posterize) int w_posterize(ie_image *img, int levels) { return ie_posterize(img, levels); }
EXPORT(flatten) int w_flatten(ie_image *img, uint32_t rgb) { return ie_flatten(img, rgb); }
EXPORT(alpha_to_gray) int w_alpha_to_gray(ie_image *img) { return ie_alpha_to_gray(img); }

EXPORT(adjust)
int w_adjust(ie_image *img, float exposure, float brightness, float contrast, float gamma, float saturation, float hue,
             float temperature, float tint)
{
    ie_adjust a;
    a.exposure = exposure;
    a.brightness = brightness;
    a.contrast = contrast;
    a.gamma = gamma;
    a.saturation = saturation;
    a.hue = hue;
    a.temperature = temperature;
    a.tint = tint;
    return ie_apply_adjust(img, &a);
}

EXPORT(resize) int w_resize(ie_image *img, int w, int h) { return ie_resize(img, w, h); }
EXPORT(crop) int w_crop(ie_image *img, int x, int y, int w, int h) { return ie_crop(img, x, y, w, h); }
EXPORT(rotate90) int w_rotate90(ie_image *img, int turns) { return ie_rotate90(img, turns); }
EXPORT(rotate) int w_rotate(ie_image *img, float degrees, uint32_t fill) { return ie_rotate(img, degrees, fill); }
EXPORT(flip) int w_flip(ie_image *img, int horizontal) { return ie_flip(img, horizontal); }
EXPORT(trim) int w_trim(ie_image *img, int tolerance) { return ie_trim(img, tolerance); }
EXPORT(pad) int w_pad(ie_image *img, int t, int r, int b, int l, uint32_t fill) { return ie_pad(img, t, r, b, l, fill); }
EXPORT(blur) int w_blur(ie_image *img, float sigma) { return ie_blur(img, sigma); }
EXPORT(sharpen) int w_sharpen(ie_image *img, float amount, float sigma, int threshold)
{
    return ie_sharpen(img, amount, sigma, threshold);
}

/* ---- background removal ------------------------------------------------ */

EXPORT(nn_input)
int w_nn_input(const ie_image *img, int tw, int th, float m0, float m1, float m2, float s0, float s1, float s2,
               float scale, float *out)
{
    const float mean[3] = {m0, m1, m2}, std[3] = {s0, s1, s2};
    return ie_nn_input(img, tw, th, mean, std, scale, out);
}

EXPORT(nn_output) void w_nn_output(float *v, size_t n, int minmax) { ie_nn_output(v, n, minmax); }

EXPORT(matte_apply) int w_matte_apply(ie_image *img, const float *matte, int mw, int mh, int radius, float eps)
{
    return ie_matte_apply(img, matte, mw, mh, radius, eps);
}

EXPORT(alpha_levels) int w_alpha_levels(ie_image *img, int lo, int hi) { return ie_alpha_levels(img, lo, hi); }
EXPORT(alpha_grow) int w_alpha_grow(ie_image *img, int radius) { return ie_alpha_grow(img, radius); }
EXPORT(alpha_feather) int w_alpha_feather(ie_image *img, float sigma) { return ie_alpha_feather(img, sigma); }
EXPORT(alpha_clean) int w_alpha_clean(ie_image *img, float min_area) { return ie_alpha_clean(img, min_area); }
EXPORT(decontaminate) int w_decontaminate(ie_image *img) { return ie_decontaminate(img); }

EXPORT(key_background)
int w_key_background(ie_image *img, int auto_color, uint32_t rgb, float tolerance, float softness, int connected)
{
    return ie_key_background(img, auto_color, rgb, tolerance, softness, connected);
}

EXPORT(finish_cutout)
int w_finish_cutout(ie_image *img, int mode, uint32_t color, float blur, const ie_image *bg, int decontaminate)
{
    ie_bg_options o;
    o.mode = mode;
    o.color = color;
    o.blur = blur;
    o.image = bg;
    o.decontaminate = decontaminate;
    return ie_finish_cutout(img, &o);
}

/* Half-precision conversion for fp16 models (round to nearest even). */
EXPORT(f32_to_f16) void w_f32_to_f16(const float *src, uint16_t *dst, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        uint32_t x;
        memcpy(&x, &src[i], 4);
        uint32_t sign = (x >> 16) & 0x8000;
        int32_t e = (int32_t)((x >> 23) & 0xff) - 112;
        uint32_t m = x & 0x7fffff;
        uint16_t h;
        if (e <= 0) {
            if (e < -10) {
                h = (uint16_t)sign;
            } else {
                m |= 0x800000;
                uint32_t shift = (uint32_t)(14 - e), half = m >> shift, rest = m & ((1u << shift) - 1);
                uint32_t mid = 1u << (shift - 1);
                if (rest > mid || (rest == mid && (half & 1))) half++;
                h = (uint16_t)(sign | half);
            }
        } else if (e >= 31) {
            h = (uint16_t)(sign | 0x7c00 | (((x >> 23) & 0xff) == 0xff && m ? 0x200 : 0));
        } else {
            uint32_t v = sign | ((uint32_t)e << 10) | (m >> 13), rest = m & 0x1fff;
            if (rest > 0x1000 || (rest == 0x1000 && (v & 1))) v++;
            h = (uint16_t)v;
        }
        dst[i] = h;
    }
}

EXPORT(f16_to_f32) void w_f16_to_f32(const uint16_t *src, float *dst, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        uint32_t h = src[i], sign = (h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, x;
        if (e == 0) {
            if (!m) {
                x = sign;
            } else {
                e = 113;
                while (!(m & 0x400)) m <<= 1, e--;
                x = sign | (e << 23) | ((m & 0x3ff) << 13);
            }
        } else if (e == 31) {
            x = sign | 0x7f800000 | (m << 13);
        } else {
            x = sign | ((e + 112) << 23) | (m << 13);
        }
        memcpy(&dst[i], &x, 4);
    }
}
