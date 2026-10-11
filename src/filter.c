/*
 * Blur and sharpen. A gaussian is approximated by three box blurs, each a
 * running sum, so the cost per pixel does not depend on the radius. Rows are
 * blurred in place; columns are gathered in strips so memory access stays
 * sequential.
 */
#include "internal.h"

#include <math.h>
#include <string.h>

#define STRIP 32

/* Box widths for an n-pass box approximation of a gaussian (Kovesi). */
static void gauss_boxes(float sigma, int *r)
{
    const int n = 3;
    float ideal = sqrtf(12 * sigma * sigma / n + 1);
    int wl = (int)floorf(ideal);
    if (wl % 2 == 0) wl--;
    int wu = wl + 2;
    float mi = (12 * sigma * sigma - n * wl * wl - 4 * n * wl - 3 * n) / (-4.0f * wl - 4);
    int m = (int)lrintf(mi);
    for (int i = 0; i < n; i++) r[i] = ((i < m ? wl : wu) - 1) / 2;
}

/* One box pass over n pixels of CH interleaved channels; edges are extended.
 * All channels share one loop so the compiler can vectorize it; CH is a
 * constant at every call site. */
static inline void box1d(const float *in, float *out, int n, const int CH, int r)
{
    if (r <= 0) {
        if (out != in) memcpy(out, in, sizeof(float) * (size_t)n * CH);
        return;
    }
    const float inv = 1.0f / (2 * r + 1);
    float acc[4] = {0, 0, 0, 0};
    for (int c = 0; c < CH; c++) {
        acc[c] = (r + 1) * in[c];
        for (int k = 1; k <= r; k++) acc[c] += in[(size_t)(k < n ? k : n - 1) * CH + c];
    }
    for (int i = 0; i < n; i++) {
        const float *pa = in + (size_t)(i + r + 1 < n ? i + r + 1 : n - 1) * CH;
        const float *ps = in + (size_t)(i - r > 0 ? i - r : 0) * CH;
        float *po = out + (size_t)i * CH;
        for (int c = 0; c < CH; c++) {
            po[c] = acc[c] * inv;
            acc[c] += pa[c] - ps[c];
        }
    }
}

/* Three passes; the result ends up back in a. */
static inline void blur1d_n(float *a, float *b, int n, const int CH, const int *r)
{
    box1d(a, b, n, CH, r[0]);
    box1d(b, a, n, CH, r[1]);
    box1d(a, b, n, CH, r[2]);
    memcpy(a, b, sizeof(float) * (size_t)n * CH);
}

/* ch is 1..4 (checked by the callers). */
static void blur1d(float *a, float *b, int n, int ch, const int *r)
{
    switch (ch) {
    case 1: blur1d_n(a, b, n, 1, r); break;
    case 2: blur1d_n(a, b, n, 2, r); break;
    case 3: blur1d_n(a, b, n, 3, r); break;
    default: blur1d_n(a, b, n, 4, r); break;
    }
}

/* ---- float planes ------------------------------------------------------ */

typedef struct {
    float *data;
    int w, h, ch;
    int r[3];
} fblur_ctx;

static void fblur_rows(void *p, int y0, int y1)
{
    fblur_ctx *c = (fblur_ctx *)p;
    float *tmp = (float *)malloc(sizeof(float) * (size_t)c->w * c->ch);
    if (!tmp) return;
    for (int y = y0; y < y1; y++) blur1d(c->data + (size_t)y * c->w * c->ch, tmp, c->w, c->ch, c->r);
    free(tmp);
}

static void fblur_cols(void *p, int s0, int s1)
{
    fblur_ctx *c = (fblur_ctx *)p;
    const int ch = c->ch;
    float *col = (float *)malloc(sizeof(float) * (size_t)c->h * ch * (STRIP + 1));
    if (!col) return;
    float *tmp = col + (size_t)c->h * ch * STRIP;
    for (int s = s0; s < s1; s++) {
        int x0 = s * STRIP, x1 = x0 + STRIP > c->w ? c->w : x0 + STRIP, sw = x1 - x0;
        for (int y = 0; y < c->h; y++) {
            const float *src = c->data + ((size_t)y * c->w + x0) * ch;
            for (int x = 0; x < sw; x++)
                for (int k = 0; k < ch; k++) col[((size_t)x * c->h + y) * ch + k] = src[x * ch + k];
        }
        for (int x = 0; x < sw; x++) blur1d(col + (size_t)x * c->h * ch, tmp, c->h, ch, c->r);
        for (int y = 0; y < c->h; y++) {
            float *dst = c->data + ((size_t)y * c->w + x0) * ch;
            for (int x = 0; x < sw; x++)
                for (int k = 0; k < ch; k++) dst[x * ch + k] = col[((size_t)x * c->h + y) * ch + k];
        }
    }
    free(col);
}

int ie_blur_f32(float *data, int w, int h, int ch, float sigma)
{
    if (!data || w <= 0 || h <= 0 || ch < 1 || ch > 4) return IE_EINVAL;
    if (sigma <= 0.1f) return IE_OK;
    fblur_ctx c;
    c.data = data;
    c.w = w;
    c.h = h;
    c.ch = ch;
    gauss_boxes(sigma, c.r);
    ie_parallel_for(h, 8, fblur_rows, &c);
    ie_parallel_for((w + STRIP - 1) / STRIP, 1, fblur_cols, &c);
    return IE_OK;
}

/* ---- RGBA8 ------------------------------------------------------------- */

/* Rows go to a 16-bit premultiplied intermediate (value * 256) so the column
 * pass does not lose precision, especially in semi-transparent areas. */
typedef struct {
    ie_image *img;
    uint16_t *mid;
    int alpha;
    int r[3];
} blur_ctx;

static void blur_rows(void *p, int y0, int y1)
{
    blur_ctx *c = (blur_ctx *)p;
    const int w = c->img->w;
    float *a = (float *)malloc(sizeof(float) * (size_t)w * 8);
    if (!a) return;
    float *b = a + (size_t)w * 4;
    for (int y = y0; y < y1; y++) {
        const uint8_t *s = IE_ROW(c->img, y);
        for (int x = 0; x < w; x++, s += 4) {
            float k = c->alpha ? s[3] / 255.0f : 1.0f;
            a[x * 4] = s[0] * k;
            a[x * 4 + 1] = s[1] * k;
            a[x * 4 + 2] = s[2] * k;
            a[x * 4 + 3] = s[3];
        }
        blur1d(a, b, w, 4, c->r);
        uint16_t *m = c->mid + (size_t)y * w * 4;
        for (int i = 0; i < w * 4; i++) m[i] = (uint16_t)lrintf(ie_clampf(a[i], 0, 255) * 256.0f);
    }
    free(a);
}

static void blur_cols(void *p, int s0, int s1)
{
    blur_ctx *c = (blur_ctx *)p;
    const int w = c->img->w, h = c->img->h;
    float *col = (float *)malloc(sizeof(float) * (size_t)h * 4 * (STRIP + 1));
    if (!col) return;
    float *tmp = col + (size_t)h * 4 * STRIP;
    for (int s = s0; s < s1; s++) {
        int x0 = s * STRIP, x1 = x0 + STRIP > w ? w : x0 + STRIP, sw = x1 - x0;
        for (int y = 0; y < h; y++) {
            const uint16_t *m = c->mid + ((size_t)y * w + x0) * 4;
            for (int x = 0; x < sw; x++)
                for (int k = 0; k < 4; k++) col[((size_t)x * h + y) * 4 + k] = m[x * 4 + k] * (1 / 256.0f);
        }
        for (int x = 0; x < sw; x++) blur1d(col + (size_t)x * h * 4, tmp, h, 4, c->r);
        for (int y = 0; y < h; y++) {
            uint8_t *d = IE_ROW(c->img, y) + (size_t)x0 * 4;
            for (int x = 0; x < sw; x++, d += 4) {
                const float *v = col + ((size_t)x * h + y) * 4;
                float al = ie_clampf(v[3], 0, 255);
                float k = c->alpha ? (al > 0.001f ? 255.0f / al : 0) : 1.0f;
                d[0] = (uint8_t)lrintf(ie_clampf(v[0] * k, 0, 255));
                d[1] = (uint8_t)lrintf(ie_clampf(v[1] * k, 0, 255));
                d[2] = (uint8_t)lrintf(ie_clampf(v[2] * k, 0, 255));
                d[3] = (uint8_t)lrintf(al);
            }
        }
    }
    free(col);
}

int ie_blur(ie_image *img, float sigma)
{
    if (!img) return IE_EINVAL;
    if (sigma <= 0.1f) return IE_OK;
    blur_ctx c;
    c.img = img;
    c.alpha = ie_has_alpha(img);
    c.mid = (uint16_t *)malloc((size_t)img->w * img->h * 4 * sizeof(uint16_t));
    if (!c.mid) return IE_ENOMEM;
    gauss_boxes(sigma, c.r);
    ie_parallel_for(img->h, 8, blur_rows, &c);
    ie_parallel_for((img->w + STRIP - 1) / STRIP, 1, blur_cols, &c);
    free(c.mid);
    return IE_OK;
}

/* ---- sharpen ----------------------------------------------------------- */

typedef struct {
    ie_image *img;
    const ie_image *blurred;
    float amount;
    int threshold;
} sharp_ctx;

static void sharpen_rows(void *p, int y0, int y1)
{
    sharp_ctx *c = (sharp_ctx *)p;
    const int w = c->img->w;
    for (int y = y0; y < y1; y++) {
        uint8_t *s = IE_ROW(c->img, y);
        const uint8_t *b = IE_ROW(c->blurred, y);
        for (int x = 0; x < w * 4; x++) {
            if ((x & 3) == 3) continue;
            int d = s[x] - b[x];
            if (abs(d) <= c->threshold) continue;
            s[x] = (uint8_t)ie_clamp255((int)lrintf(s[x] + c->amount * d));
        }
    }
}

int ie_sharpen(ie_image *img, float amount, float sigma, int threshold)
{
    if (!img) return IE_EINVAL;
    if (amount <= 0) return IE_OK;
    if (sigma <= 0) sigma = 1.0f;
    ie_image *bl = ie_clone(img);
    if (!bl) return IE_ENOMEM;
    int r = ie_blur(bl, sigma);
    if (r == IE_OK) {
        sharp_ctx c = {img, bl, amount, threshold < 0 ? 0 : threshold};
        ie_parallel_for(img->h, 16, sharpen_rows, &c);
    }
    ie_free(bl);
    return r;
}

/* ---- box mean ---------------------------------------------------------- */

typedef struct {
    const float *src;
    float *tmp, *dst;
    int w, h, r;
} box_ctx;

static void box_rows(void *p, int y0, int y1)
{
    box_ctx *c = (box_ctx *)p;
    const int w = c->w, r = c->r;
    for (int y = y0; y < y1; y++) {
        const float *s = c->src + (size_t)y * w;
        float *d = c->tmp + (size_t)y * w;
        double acc = 0;
        for (int x = 0; x <= r && x < w; x++) acc += s[x];
        for (int x = 0; x < w; x++) {
            int lo = x - r < 0 ? 0 : x - r, hi = x + r >= w ? w - 1 : x + r;
            d[x] = (float)(acc / (hi - lo + 1));
            if (x + r + 1 < w) acc += s[x + r + 1];
            if (x - r >= 0) acc -= s[x - r];
        }
    }
}

static void box_cols(void *p, int s0, int s1)
{
    box_ctx *c = (box_ctx *)p;
    const int w = c->w, h = c->h, r = c->r;
    double acc[64];
    for (int s = s0; s < s1; s++) {
        int x0 = s * 64, x1 = x0 + 64 > w ? w : x0 + 64, sw = x1 - x0;
        for (int x = 0; x < sw; x++) acc[x] = 0;
        for (int y = 0; y <= r && y < h; y++)
            for (int x = 0; x < sw; x++) acc[x] += c->tmp[(size_t)y * w + x0 + x];
        for (int y = 0; y < h; y++) {
            int lo = y - r < 0 ? 0 : y - r, hi = y + r >= h ? h - 1 : y + r;
            double inv = 1.0 / (hi - lo + 1);
            float *d = c->dst + (size_t)y * w + x0;
            for (int x = 0; x < sw; x++) d[x] = (float)(acc[x] * inv);
            if (y + r + 1 < h) {
                const float *a = c->tmp + (size_t)(y + r + 1) * w + x0;
                for (int x = 0; x < sw; x++) acc[x] += a[x];
            }
            if (y - r >= 0) {
                const float *b = c->tmp + (size_t)(y - r) * w + x0;
                for (int x = 0; x < sw; x++) acc[x] -= b[x];
            }
        }
    }
}

int ie_box_mean(const float *src, float *dst, int w, int h, int r)
{
    if (r <= 0) {
        if (dst != src) memcpy(dst, src, sizeof(float) * (size_t)w * h);
        return IE_OK;
    }
    box_ctx c;
    c.src = src;
    c.dst = dst;
    c.w = w;
    c.h = h;
    c.r = r;
    c.tmp = (float *)malloc(sizeof(float) * (size_t)w * h);
    if (!c.tmp) return IE_ENOMEM;
    ie_parallel_for(h, 16, box_rows, &c);
    ie_parallel_for((w + 63) / 64, 1, box_cols, &c);
    free(c.tmp);
    return IE_OK;
}
