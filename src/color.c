/* Per-pixel color operations. Each is one multithreaded pass over the rows. */
#include "imgedit.h"
#include "parallel.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define ROW(img, y) ((img)->px + (size_t)(y) * (size_t)(img)->w * 4)
#define GRAIN 16

static inline int clamp255(int v) { return v < 0 ? 0 : v > 255 ? 255 : v; }
static inline uint8_t luma(const uint8_t *p) { return (uint8_t)((54 * p[0] + 183 * p[1] + 19 * p[2] + 128) >> 8); }

typedef struct {
    ie_image *img;
    const uint8_t *lut; /* 4 x 256 (r, g, b, a) */
    int mat[9];         /* 3x3, Q12 fixed point */
    int mode, level;
    uint32_t color;
} px_ctx;

/* ---- LUT / matrix machinery -------------------------------------------- */

static void lut_rows(void *p, int y0, int y1)
{
    const px_ctx *c = (const px_ctx *)p;
    const uint8_t *lr = c->lut, *lg = c->lut + 256, *lb = c->lut + 512, *la = c->lut + 768;
    const int w = c->img->w;
    for (int y = y0; y < y1; y++) {
        uint8_t *s = ROW(c->img, y);
        for (int x = 0; x < w; x++, s += 4) {
            s[0] = lr[s[0]];
            s[1] = lg[s[1]];
            s[2] = lb[s[2]];
            s[3] = la[s[3]];
        }
    }
}

static int apply_lut(ie_image *img, const uint8_t *lut)
{
    px_ctx c;
    c.img = img;
    c.lut = lut;
    ie_parallel_for(img->h, GRAIN, lut_rows, &c);
    return IE_OK;
}

static void identity_lut(uint8_t *lut)
{
    for (int k = 0; k < 4; k++)
        for (int i = 0; i < 256; i++) lut[k * 256 + i] = (uint8_t)i;
}

static void mat_rows(void *p, int y0, int y1)
{
    const px_ctx *c = (const px_ctx *)p;
    const int *m = c->mat;
    const int w = c->img->w;
    for (int y = y0; y < y1; y++) {
        uint8_t *s = ROW(c->img, y);
        for (int x = 0; x < w; x++, s += 4) {
            int r = s[0], g = s[1], b = s[2];
            s[0] = (uint8_t)clamp255((m[0] * r + m[1] * g + m[2] * b + 2048) >> 12);
            s[1] = (uint8_t)clamp255((m[3] * r + m[4] * g + m[5] * b + 2048) >> 12);
            s[2] = (uint8_t)clamp255((m[6] * r + m[7] * g + m[8] * b + 2048) >> 12);
        }
    }
}

static int apply_matrix(ie_image *img, const float *m)
{
    px_ctx c;
    c.img = img;
    for (int i = 0; i < 9; i++) c.mat[i] = (int)lrintf(m[i] * 4096.0f);
    ie_parallel_for(img->h, GRAIN, mat_rows, &c);
    return IE_OK;
}

static void mat_mul(const float *a, const float *b, float *out)
{
    float t[9];
    for (int r = 0; r < 3; r++)
        for (int k = 0; k < 3; k++) t[r * 3 + k] = a[r * 3] * b[k] + a[r * 3 + 1] * b[3 + k] + a[r * 3 + 2] * b[6 + k];
    memcpy(out, t, sizeof t);
}

/* ---- invert ------------------------------------------------------------ */

static void invert_rows(void *p, int y0, int y1)
{
    const px_ctx *c = (const px_ctx *)p;
    const int w = c->img->w, mode = c->mode;
    for (int y = y0; y < y1; y++) {
        uint8_t *s = ROW(c->img, y);
        if (mode & IE_INVERT_LIGHTNESS) {
            /* HSL lightness L -> 1-L with hue and saturation unchanged works out
             * to shifting every channel by 255 - max - min. */
            for (int x = 0; x < w; x++, s += 4) {
                int r = s[0], g = s[1], b = s[2];
                int mx = r > g ? (r > b ? r : b) : (g > b ? g : b);
                int mn = r < g ? (r < b ? r : b) : (g < b ? g : b);
                int k = 255 - mx - mn;
                s[0] = (uint8_t)(r + k);
                s[1] = (uint8_t)(g + k);
                s[2] = (uint8_t)(b + k);
            }
            s = ROW(c->img, y);
        } else if (mode & IE_INVERT_RGB) {
            for (int x = 0; x < w; x++, s += 4) {
                s[0] = (uint8_t)(255 - s[0]);
                s[1] = (uint8_t)(255 - s[1]);
                s[2] = (uint8_t)(255 - s[2]);
            }
            s = ROW(c->img, y);
        }
        if (mode & IE_INVERT_ALPHA)
            for (int x = 0; x < w; x++) s[x * 4 + 3] = (uint8_t)(255 - s[x * 4 + 3]);
    }
}

int ie_invert(ie_image *img, int mode)
{
    if (!img || !mode) return IE_EINVAL;
    px_ctx c;
    c.img = img;
    c.mode = mode;
    ie_parallel_for(img->h, GRAIN, invert_rows, &c);
    return IE_OK;
}

/* ---- grayscale, sepia --------------------------------------------------- */

static void gray_rows(void *p, int y0, int y1)
{
    const px_ctx *c = (const px_ctx *)p;
    const int w = c->img->w;
    for (int y = y0; y < y1; y++) {
        uint8_t *s = ROW(c->img, y);
        for (int x = 0; x < w; x++, s += 4) s[0] = s[1] = s[2] = luma(s);
    }
}

int ie_grayscale(ie_image *img)
{
    if (!img) return IE_EINVAL;
    px_ctx c;
    c.img = img;
    ie_parallel_for(img->h, GRAIN, gray_rows, &c);
    return IE_OK;
}

int ie_sepia(ie_image *img, float amount)
{
    if (!img) return IE_EINVAL;
    if (amount <= 0) return IE_OK;
    if (amount > 1) amount = 1;
    static const float S[9] = {0.393f, 0.769f, 0.189f, 0.349f, 0.686f, 0.168f, 0.272f, 0.534f, 0.131f};
    float m[9];
    for (int i = 0; i < 9; i++) m[i] = (i % 4 == 0 ? 1 - amount : 0) + amount * S[i];
    return apply_matrix(img, m);
}

/* ---- adjustments -------------------------------------------------------- */

void ie_adjust_init(ie_adjust *a)
{
    memset(a, 0, sizeof *a);
    a->gamma = 1;
}

int ie_adjust_is_identity(const ie_adjust *a)
{
    return a->exposure == 0 && a->brightness == 0 && a->contrast == 0 && (a->gamma == 1 || a->gamma <= 0) &&
           a->saturation == 0 && fmodf(a->hue, 360.0f) == 0 && a->temperature == 0 && a->tint == 0;
}

static float srgb_to_linear(float v) { return v <= 0.04045f ? v / 12.92f : powf((v + 0.055f) / 1.055f, 2.4f); }
static float linear_to_srgb(float v) { return v <= 0.0031308f ? v * 12.92f : 1.055f * powf(v, 1 / 2.4f) - 0.055f; }

int ie_apply_adjust(ie_image *img, const ie_adjust *a)
{
    if (!img || !a) return IE_EINVAL;
    if (ie_adjust_is_identity(a)) return IE_OK;

    /* Tone: one LUT per channel. Exposure and white balance act on linear
     * light, brightness/contrast/gamma on the encoded values. */
    float t = a->temperature, m = a->tint;
    float gain[3] = {1 + 0.3f * t, 1 - 0.3f * m, 1 - 0.3f * t};
    float ex = powf(2.0f, a->exposure);
    float con = a->contrast < -1 ? -1 : a->contrast > 0.99f ? 0.99f : a->contrast;
    float cf = tanf((con + 1) * 0.78539816f);
    float inv_gamma = a->gamma > 0 ? 1 / a->gamma : 1;
    uint8_t lut[1024];
    identity_lut(lut);
    for (int ch = 0; ch < 3; ch++) {
        float g = ex * gain[ch];
        for (int i = 0; i < 256; i++) {
            float v = i / 255.0f;
            if (g != 1) {
                v = srgb_to_linear(v) * g;
                v = linear_to_srgb(v > 1 ? 1 : v);
            }
            v += a->brightness;
            v = (v - 0.5f) * cf + 0.5f;
            v = v < 0 ? 0 : v > 1 ? 1 : v;
            if (inv_gamma != 1) v = powf(v, inv_gamma);
            lut[ch * 256 + i] = (uint8_t)clamp255((int)lrintf(v * 255));
        }
    }
    apply_lut(img, lut);

    /* Color: saturation and hue rotation as one 3x3 matrix around the luma axis. */
    if (a->saturation != 0 || fmodf(a->hue, 360.0f) != 0) {
        float s = 1 + (a->saturation < -1 ? -1 : a->saturation);
        float S[9] = {0.2126f + 0.7874f * s, 0.7152f - 0.7152f * s, 0.0722f - 0.0722f * s,
                      0.2126f - 0.2126f * s, 0.7152f + 0.2848f * s, 0.0722f - 0.0722f * s,
                      0.2126f - 0.2126f * s, 0.7152f - 0.7152f * s, 0.0722f + 0.9278f * s};
        float h = a->hue * 0.017453293f, co = cosf(h), si = sinf(h);
        float H[9] = {0.213f + co * 0.787f - si * 0.213f, 0.715f - co * 0.715f - si * 0.715f,
                      0.072f - co * 0.072f + si * 0.928f, 0.213f - co * 0.213f + si * 0.143f,
                      0.715f + co * 0.285f + si * 0.140f, 0.072f - co * 0.072f - si * 0.283f,
                      0.213f - co * 0.213f - si * 0.787f, 0.715f - co * 0.715f + si * 0.715f,
                      0.072f + co * 0.928f + si * 0.072f};
        float M[9];
        mat_mul(S, H, M);
        apply_matrix(img, M);
    }
    return IE_OK;
}

/* ---- levels ------------------------------------------------------------ */

#define SLICES 64

typedef struct {
    const ie_image *img;
    uint32_t (*hist)[4][256]; /* per slice: r, g, b, luma */
    int rows_per_slice;
} hist_ctx;

static void hist_slices(void *p, int s0, int s1)
{
    hist_ctx *c = (hist_ctx *)p;
    const int w = c->img->w;
    for (int s = s0; s < s1; s++) {
        uint32_t(*h)[256] = c->hist[s];
        int y0 = s * c->rows_per_slice, y1 = y0 + c->rows_per_slice;
        if (y1 > c->img->h) y1 = c->img->h;
        for (int y = y0; y < y1; y++) {
            const uint8_t *px = ROW(c->img, y);
            for (int x = 0; x < w; x++, px += 4) {
                if (px[3] == 0) continue;
                h[0][px[0]]++;
                h[1][px[1]]++;
                h[2][px[2]]++;
                h[3][luma(px)]++;
            }
        }
    }
}

static void find_range(const uint32_t *h, float clip, int *lo, int *hi)
{
    uint64_t total = 0;
    for (int i = 0; i < 256; i++) total += h[i];
    uint64_t cut = (uint64_t)(total * (double)clip), acc = 0;
    *lo = 0;
    *hi = 255;
    for (int i = 0; i < 256; i++) {
        acc += h[i];
        if (acc > cut) {
            *lo = i;
            break;
        }
    }
    acc = 0;
    for (int i = 255; i >= 0; i--) {
        acc += h[i];
        if (acc > cut) {
            *hi = i;
            break;
        }
    }
}

int ie_auto_levels(ie_image *img, float clip, int per_channel)
{
    if (!img) return IE_EINVAL;
    if (clip < 0) clip = 0;
    if (clip > 0.2f) clip = 0.2f;
    hist_ctx c;
    c.img = img;
    c.rows_per_slice = (img->h + SLICES - 1) / SLICES;
    int slices = (img->h + c.rows_per_slice - 1) / c.rows_per_slice;
    c.hist = (uint32_t(*)[4][256])calloc((size_t)slices, sizeof *c.hist);
    if (!c.hist) return IE_ENOMEM;
    ie_parallel_for(slices, 1, hist_slices, &c);
    uint32_t h[4][256] = {{0}};
    for (int s = 0; s < slices; s++)
        for (int k = 0; k < 4; k++)
            for (int i = 0; i < 256; i++) h[k][i] += c.hist[s][k][i];
    free(c.hist);

    uint8_t lut[1024];
    identity_lut(lut);
    for (int ch = 0; ch < 3; ch++) {
        int lo, hi;
        find_range(per_channel ? h[ch] : h[3], clip, &lo, &hi);
        if (hi - lo < 2) continue;
        for (int i = 0; i < 256; i++) lut[ch * 256 + i] = (uint8_t)clamp255(((i - lo) * 255 + (hi - lo) / 2) / (hi - lo));
    }
    return apply_lut(img, lut);
}

static void threshold_rows(void *p, int y0, int y1)
{
    const px_ctx *c = (const px_ctx *)p;
    const int w = c->img->w;
    for (int y = y0; y < y1; y++) {
        uint8_t *s = ROW(c->img, y);
        for (int x = 0; x < w; x++, s += 4) s[0] = s[1] = s[2] = luma(s) >= c->level ? 255 : 0;
    }
}

int ie_threshold(ie_image *img, int level)
{
    if (!img) return IE_EINVAL;
    px_ctx c;
    c.img = img;
    c.level = level;
    ie_parallel_for(img->h, GRAIN, threshold_rows, &c);
    return IE_OK;
}

int ie_posterize(ie_image *img, int levels)
{
    if (!img || levels < 2) return IE_EINVAL;
    if (levels > 255) return IE_OK;
    uint8_t lut[1024];
    identity_lut(lut);
    for (int i = 0; i < 256; i++) {
        int q = (i * (levels - 1) + 127) / 255;
        lut[i] = lut[256 + i] = lut[512 + i] = (uint8_t)((q * 255 + (levels - 1) / 2) / (levels - 1));
    }
    return apply_lut(img, lut);
}

/* ---- alpha ------------------------------------------------------------- */

static void flatten_rows(void *p, int y0, int y1)
{
    const px_ctx *c = (const px_ctx *)p;
    const int w = c->img->w;
    const unsigned br = (c->color >> 16) & 255, bg = (c->color >> 8) & 255, bb = c->color & 255;
    for (int y = y0; y < y1; y++) {
        uint8_t *s = ROW(c->img, y);
        for (int x = 0; x < w; x++, s += 4) {
            unsigned a = s[3], na = 255 - a;
            s[0] = (uint8_t)((s[0] * a + br * na + 127) / 255);
            s[1] = (uint8_t)((s[1] * a + bg * na + 127) / 255);
            s[2] = (uint8_t)((s[2] * a + bb * na + 127) / 255);
            s[3] = 255;
        }
    }
}

int ie_flatten(ie_image *img, uint32_t rgb)
{
    if (!img) return IE_EINVAL;
    px_ctx c;
    c.img = img;
    c.color = rgb;
    ie_parallel_for(img->h, GRAIN, flatten_rows, &c);
    return IE_OK;
}

static void alpha_gray_rows(void *p, int y0, int y1)
{
    const px_ctx *c = (const px_ctx *)p;
    const int w = c->img->w;
    for (int y = y0; y < y1; y++) {
        uint8_t *s = ROW(c->img, y);
        for (int x = 0; x < w; x++, s += 4) {
            s[0] = s[1] = s[2] = s[3];
            s[3] = 255;
        }
    }
}

int ie_alpha_to_gray(ie_image *img)
{
    if (!img) return IE_EINVAL;
    px_ctx c;
    c.img = img;
    ie_parallel_for(img->h, GRAIN, alpha_gray_rows, &c);
    return IE_OK;
}

int ie_alpha_levels(ie_image *img, int lo, int hi)
{
    if (!img) return IE_EINVAL;
    lo = clamp255(lo);
    hi = clamp255(hi);
    if (hi <= lo) hi = lo + 1;
    uint8_t lut[1024];
    identity_lut(lut);
    for (int i = 0; i < 256; i++)
        lut[768 + i] = (uint8_t)(i <= lo ? 0 : i >= hi ? 255 : clamp255(((i - lo) * 255 + (hi - lo) / 2) / (hi - lo)));
    return apply_lut(img, lut);
}
