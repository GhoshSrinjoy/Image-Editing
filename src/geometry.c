/* Resizing, cropping, rotation and other geometric operations. */
#include "internal.h"

#include <math.h>
#include <string.h>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../third_party/stb/stb_image_resize2.h"

/* ---- resize ------------------------------------------------------------ */

static void resize_splits(void *p, int s0, int s1) { stbir_resize_extended_split((STBIR_RESIZE *)p, s0, s1 - s0); }

static int run_resize(STBIR_RESIZE *r)
{
    int splits = stbir_build_samplers_with_splits(r, ie_get_threads());
    if (!splits) return IE_ENOMEM;
    ie_parallel_for(splits, 1, resize_splits, r);
    stbir_free_samplers(r);
    return IE_OK;
}

int ie_resize_rgba(const uint8_t *src, int sw, int sh, uint8_t *dst, int dw, int dh, int srgb)
{
    STBIR_RESIZE r;
    stbir_resize_init(&r, src, sw, sh, 0, dst, dw, dh, 0, STBIR_RGBA, srgb ? STBIR_TYPE_UINT8_SRGB : STBIR_TYPE_UINT8);
    return run_resize(&r);
}

int ie_resize_f32(const float *src, int sw, int sh, float *dst, int dw, int dh, int channels)
{
    stbir_pixel_layout layout = channels == 1 ? STBIR_1CHANNEL : channels == 3 ? STBIR_RGB : STBIR_4CHANNEL;
    if (channels != 1 && channels != 3 && channels != 4) return IE_EINVAL;
    STBIR_RESIZE r;
    stbir_resize_init(&r, src, sw, sh, 0, dst, dw, dh, 0, layout, STBIR_TYPE_FLOAT);
    return run_resize(&r);
}

int ie_resize(ie_image *img, int w, int h)
{
    if (!img || w <= 0 || h <= 0 || (size_t)w > ((size_t)1 << 30) / (size_t)h) return IE_EINVAL;
    if (w == img->w && h == img->h) return IE_OK;
    uint8_t *dst = (uint8_t *)malloc((size_t)w * h * 4);
    if (!dst) return IE_ENOMEM;
    int r = ie_resize_rgba(img->px, img->w, img->h, dst, w, h, 1);
    if (r) {
        free(dst);
        return r;
    }
    ie_replace(img, dst, w, h);
    return IE_OK;
}

/* ---- crop / pad -------------------------------------------------------- */

int ie_crop(ie_image *img, int x, int y, int w, int h)
{
    if (!img) return IE_EINVAL;
    if (x < 0) w += x, x = 0;
    if (y < 0) h += y, y = 0;
    if (x + w > img->w) w = img->w - x;
    if (y + h > img->h) h = img->h - y;
    if (w <= 0 || h <= 0) return IE_EINVAL;
    if (x == 0 && y == 0 && w == img->w && h == img->h) return IE_OK;
    uint8_t *dst = (uint8_t *)malloc((size_t)w * h * 4);
    if (!dst) return IE_ENOMEM;
    for (int r = 0; r < h; r++) memcpy(dst + (size_t)r * w * 4, IE_ROW(img, y + r) + (size_t)x * 4, (size_t)w * 4);
    ie_replace(img, dst, w, h);
    return IE_OK;
}

static void put_rgba(uint8_t *p, uint32_t rgba)
{
    p[0] = (uint8_t)(rgba >> 24);
    p[1] = (uint8_t)(rgba >> 16);
    p[2] = (uint8_t)(rgba >> 8);
    p[3] = (uint8_t)rgba;
}

int ie_pad(ie_image *img, int top, int right, int bottom, int left, uint32_t fill_rgba)
{
    if (!img || top < 0 || right < 0 || bottom < 0 || left < 0) return IE_EINVAL;
    int w = img->w + left + right, h = img->h + top + bottom;
    if (w == img->w && h == img->h) return IE_OK;
    ie_image *out = ie_new(w, h);
    if (!out) return IE_ENOMEM;
    if (fill_rgba) {
        put_rgba(out->px, fill_rgba);
        for (size_t i = 1; i < (size_t)w * h; i++) memcpy(out->px + i * 4, out->px, 4);
    }
    for (int r = 0; r < img->h; r++)
        memcpy(IE_ROW(out, top + r) + (size_t)left * 4, IE_ROW(img, r), (size_t)img->w * 4);
    ie_swap(img, out);
    ie_free(out);
    return IE_OK;
}

/* ---- flips, quarter turns, transpose ------------------------------------ */

typedef struct {
    const ie_image *src;
    uint32_t *dst;
    ie_image *img;
    int mode;
} geo_ctx;

static void flip_rows(void *p, int y0, int y1)
{
    geo_ctx *c = (geo_ctx *)p;
    ie_image *img = c->img;
    const int w = img->w;
    for (int y = y0; y < y1; y++) {
        if (c->mode) { /* horizontal: mirror each row */
            uint32_t *row = (uint32_t *)(void *)IE_ROW(img, y);
            for (int a = 0, b = w - 1; a < b; a++, b--) {
                uint32_t t = row[a];
                row[a] = row[b];
                row[b] = t;
            }
        } else { /* vertical: swap row y with its mirror (y < h/2 only) */
            uint8_t *a = IE_ROW(img, y), *b = IE_ROW(img, img->h - 1 - y);
            for (int x = 0; x < w * 4; x++) {
                uint8_t t = a[x];
                a[x] = b[x];
                b[x] = t;
            }
        }
    }
}

int ie_flip(ie_image *img, int horizontal)
{
    if (!img) return IE_EINVAL;
    geo_ctx c;
    c.img = img;
    c.mode = horizontal;
    ie_parallel_for(horizontal ? img->h : img->h / 2, 16, flip_rows, &c);
    return IE_OK;
}

/* mode 1: 90 cw, 3: 90 ccw, 4: transpose. Works in 64x64 tiles for cache reuse. */
#define TILE 64
static void turn_tiles(void *p, int t0, int t1)
{
    geo_ctx *c = (geo_ctx *)p;
    const int W = c->src->w, H = c->src->h;
    const uint32_t *src = (const uint32_t *)(const void *)c->src->px;
    uint32_t *dst = c->dst;
    for (int ty = t0; ty < t1; ty++) {
        int y0 = ty * TILE, y1 = y0 + TILE > H ? H : y0 + TILE;
        for (int x0 = 0; x0 < W; x0 += TILE) {
            int x1 = x0 + TILE > W ? W : x0 + TILE;
            for (int y = y0; y < y1; y++)
                for (int x = x0; x < x1; x++) {
                    size_t d = c->mode == 1   ? (size_t)x * H + (H - 1 - y)
                               : c->mode == 3 ? (size_t)(W - 1 - x) * H + y
                                              : (size_t)x * H + y;
                    dst[d] = src[(size_t)y * W + x];
                }
        }
    }
}

static int turn(ie_image *img, int mode)
{
    uint32_t *dst = (uint32_t *)malloc((size_t)img->w * img->h * 4);
    if (!dst) return IE_ENOMEM;
    geo_ctx c;
    c.src = img;
    c.dst = dst;
    c.mode = mode;
    ie_parallel_for((img->h + TILE - 1) / TILE, 1, turn_tiles, &c);
    ie_replace(img, (uint8_t *)dst, img->h, img->w);
    return IE_OK;
}

static void reverse_pixels(void *p, int b0, int b1)
{
    geo_ctx *c = (geo_ctx *)p;
    uint32_t *px = (uint32_t *)(void *)c->img->px;
    size_t n = (size_t)c->img->w * c->img->h;
    for (size_t i = (size_t)b0 * 4096; i < (size_t)b1 * 4096 && i < n / 2; i++) {
        uint32_t t = px[i];
        px[i] = px[n - 1 - i];
        px[n - 1 - i] = t;
    }
}

int ie_rotate90(ie_image *img, int turns)
{
    if (!img) return IE_EINVAL;
    turns = ((turns % 4) + 4) % 4;
    if (turns == 0) return IE_OK;
    if (turns == 2) {
        geo_ctx c;
        c.img = img;
        size_t half = (size_t)img->w * img->h / 2;
        ie_parallel_for((int)((half + 4095) / 4096), 4, reverse_pixels, &c);
        return IE_OK;
    }
    return turn(img, turns);
}

int ie_transpose(ie_image *img)
{
    if (!img) return IE_EINVAL;
    return turn(img, 4);
}

/* ---- arbitrary rotation ----------------------------------------------- */

typedef struct {
    const ie_image *src;
    ie_image *dst;
    float cs, sn, cx, cy, ox, oy;
    float fill[4];      /* premultiplied, for edge taps */
    uint8_t fill_px[4]; /* straight, for pixels entirely outside */
} rot_ctx;

static inline void cubic_weights(float t, float *w)
{
    float t2 = t * t, t3 = t2 * t;
    w[0] = 0.5f * (-t3 + 2 * t2 - t);
    w[1] = 0.5f * (3 * t3 - 5 * t2 + 2);
    w[2] = 0.5f * (-3 * t3 + 4 * t2 + t);
    w[3] = 0.5f * (t3 - t2);
}

static void rotate_rows(void *p, int y0, int y1)
{
    const rot_ctx *c = (const rot_ctx *)p;
    const ie_image *s = c->src;
    for (int y = y0; y < y1; y++) {
        uint8_t *out = IE_ROW(c->dst, y);
        float dy = y + 0.5f - c->oy;
        for (int x = 0; x < c->dst->w; x++, out += 4) {
            float dx = x + 0.5f - c->ox;
            float sx = c->cx + dx * c->cs + dy * c->sn - 0.5f;
            float sy = c->cy - dx * c->sn + dy * c->cs - 0.5f;
            int ix = (int)floorf(sx), iy = (int)floorf(sy);
            if (ix < -2 || iy < -2 || ix > s->w || iy > s->h) {
                memcpy(out, c->fill_px, 4);
                continue;
            }
            float wx[4], wy[4], acc[4] = {0, 0, 0, 0};
            cubic_weights(sx - ix, wx);
            cubic_weights(sy - iy, wy);
            for (int j = 0; j < 4; j++) {
                int yy = iy - 1 + j;
                float row[4] = {0, 0, 0, 0};
                for (int i = 0; i < 4; i++) {
                    int xx = ix - 1 + i;
                    float v[4];
                    if (xx < 0 || yy < 0 || xx >= s->w || yy >= s->h) {
                        memcpy(v, c->fill, sizeof v);
                    } else {
                        const uint8_t *q = IE_ROW(s, yy) + (size_t)xx * 4;
                        float a = q[3];
                        v[0] = q[0] * a * (1 / 255.0f);
                        v[1] = q[1] * a * (1 / 255.0f);
                        v[2] = q[2] * a * (1 / 255.0f);
                        v[3] = a;
                    }
                    for (int k = 0; k < 4; k++) row[k] += wx[i] * v[k];
                }
                for (int k = 0; k < 4; k++) acc[k] += wy[j] * row[k];
            }
            float a = ie_clampf(acc[3], 0, 255);
            out[3] = (uint8_t)lrintf(a);
            for (int k = 0; k < 3; k++)
                out[k] = a > 0 ? (uint8_t)lrintf(ie_clampf(acc[k], 0, a) * 255.0f / a) : 0;
        }
    }
}

int ie_rotate(ie_image *img, float degrees, uint32_t fill_rgba)
{
    if (!img) return IE_EINVAL;
    float q = degrees / 90.0f;
    if (fabsf(q - roundf(q)) < 1e-6f) return ie_rotate90(img, (int)roundf(q));

    double th = degrees * 3.14159265358979323846 / 180.0;
    double cs = cos(th), sn = sin(th);
    int w = (int)ceil(fabs(img->w * cs) + fabs(img->h * sn) - 1e-6);
    int h = (int)ceil(fabs(img->w * sn) + fabs(img->h * cs) - 1e-6);
    ie_image *out = ie_new(w, h);
    if (!out) return IE_ENOMEM;
    rot_ctx c;
    c.src = img;
    c.dst = out;
    c.cs = (float)cs;
    c.sn = (float)sn;
    c.cx = img->w * 0.5f;
    c.cy = img->h * 0.5f;
    c.ox = w * 0.5f;
    c.oy = h * 0.5f;
    float fa = (float)(fill_rgba & 255);
    c.fill[0] = (float)(fill_rgba >> 24) * fa / 255.0f;
    c.fill[1] = (float)((fill_rgba >> 16) & 255) * fa / 255.0f;
    c.fill[2] = (float)((fill_rgba >> 8) & 255) * fa / 255.0f;
    c.fill[3] = fa;
    put_rgba(c.fill_px, fill_rgba);
    ie_parallel_for(h, 8, rotate_rows, &c);
    ie_swap(img, out);
    ie_free(out);
    return IE_OK;
}

/* ---- trim -------------------------------------------------------------- */

typedef struct {
    const ie_image *img;
    int alpha_mode, tol;
    uint8_t ref[4];
    int *minx, *maxx; /* per row, -1 when the row is empty */
} trim_ctx;

static void trim_rows(void *p, int y0, int y1)
{
    trim_ctx *c = (trim_ctx *)p;
    const int w = c->img->w;
    for (int y = y0; y < y1; y++) {
        const uint8_t *s = IE_ROW(c->img, y);
        int lo = -1, hi = -1;
        for (int x = 0; x < w; x++, s += 4) {
            int content;
            if (c->alpha_mode) {
                content = s[3] > c->tol;
            } else {
                int d0 = abs(s[0] - c->ref[0]), d1 = abs(s[1] - c->ref[1]), d2 = abs(s[2] - c->ref[2]);
                content = d0 > c->tol || d1 > c->tol || d2 > c->tol;
            }
            if (content) {
                if (lo < 0) lo = x;
                hi = x;
            }
        }
        c->minx[y] = lo;
        c->maxx[y] = hi;
    }
}

int ie_trim(ie_image *img, int tolerance)
{
    if (!img) return IE_EINVAL;
    trim_ctx c;
    c.img = img;
    c.tol = ie_clamp255(tolerance);
    const uint8_t *corner[4] = {img->px, IE_ROW(img, 0) + (size_t)(img->w - 1) * 4, IE_ROW(img, img->h - 1),
                                IE_ROW(img, img->h - 1) + (size_t)(img->w - 1) * 4};
    c.alpha_mode = 0;
    for (int i = 0; i < 4; i++)
        if (corner[i][3] <= c.tol) c.alpha_mode = 1;
    memcpy(c.ref, img->px, 4);
    c.minx = (int *)malloc(sizeof(int) * 2 * (size_t)img->h);
    if (!c.minx) return IE_ENOMEM;
    c.maxx = c.minx + img->h;
    ie_parallel_for(img->h, 32, trim_rows, &c);
    int x0 = img->w, x1 = -1, y0 = -1, y1 = -1;
    for (int y = 0; y < img->h; y++) {
        if (c.minx[y] < 0) continue;
        if (y0 < 0) y0 = y;
        y1 = y;
        if (c.minx[y] < x0) x0 = c.minx[y];
        if (c.maxx[y] > x1) x1 = c.maxx[y];
    }
    free(c.minx);
    if (y0 < 0) return IE_OK; /* nothing but background: leave as is */
    return ie_crop(img, x0, y0, x1 - x0 + 1, y1 - y0 + 1);
}
