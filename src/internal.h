/* Helpers shared between the library's translation units (not public API). */
#ifndef IE_INTERNAL_H
#define IE_INTERNAL_H

#include "imgedit.h"
#include "parallel.h"

#include <stdlib.h>

#define IE_ROW(img, y) ((img)->px + (size_t)(y) * (size_t)(img)->w * 4)

static inline int ie_clamp255(int v) { return v < 0 ? 0 : v > 255 ? 255 : v; }
static inline float ie_clampf(float v, float lo, float hi) { return v < lo ? lo : v > hi ? hi : v; }

/* Replace img's pixels with a malloc'd buffer of the given size. */
static inline void ie_replace(ie_image *img, uint8_t *px, int w, int h)
{
    free(img->px);
    img->px = px;
    img->w = w;
    img->h = h;
}

/* RGBA8 resize, alpha-weighted. srgb selects gamma-correct filtering. */
int ie_resize_rgba(const uint8_t *src, int sw, int sh, uint8_t *dst, int dw, int dh, int srgb);

/* Resize interleaved float pixels with 1, 3 or 4 channels. */
int ie_resize_f32(const float *src, int sw, int sh, float *dst, int dw, int dh, int channels);

/* Gaussian-like blur (three box passes) of `ch` (1..4) interleaved float channels,
 * in place. Edges are clamped. */
int ie_blur_f32(float *data, int w, int h, int ch, float sigma);

/* Mean over a (2r+1)^2 window of a single float plane; the window is clipped
 * at the borders and divided by the number of pixels it covers. */
int ie_box_mean(const float *src, float *dst, int w, int h, int r);

#endif
