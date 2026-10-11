/*
 * Background removal building blocks: model pre/post-processing, matte
 * upsampling with a guided filter, alpha clean-up, foreground color
 * estimation and compositing onto a new background.
 */
#include "internal.h"

#include <float.h>
#include <math.h>
#include <string.h>

/* ---- model input / output ---------------------------------------------- */

int ie_nn_input(const ie_image *img, int tw, int th, const float mean[3], const float std[3], float scale, float *out)
{
    if (!img || tw <= 0 || th <= 0 || !out) return IE_EINVAL;
    uint8_t *tmp = (uint8_t *)malloc((size_t)tw * th * 4);
    if (!tmp) return IE_ENOMEM;
    int r = ie_resize_rgba(img->px, img->w, img->h, tmp, tw, th, 0);
    if (r) {
        free(tmp);
        return r;
    }
    const size_t plane = (size_t)tw * th;
    for (size_t i = 0; i < plane; i++) {
        const uint8_t *p = tmp + i * 4;
        float a = p[3] / 255.0f;
        for (int c = 0; c < 3; c++) {
            float v = p[c] * a + 255.0f * (1 - a); /* transparent areas read as white */
            out[c * plane + i] = (v * scale - mean[c]) / std[c];
        }
    }
    free(tmp);
    return IE_OK;
}

void ie_nn_output(float *v, size_t n, int minmax)
{
    if (!v || !n) return;
    float mn = FLT_MAX, mx = -FLT_MAX;
    for (size_t i = 0; i < n; i++) {
        if (v[i] < mn) mn = v[i];
        if (v[i] > mx) mx = v[i];
    }
    if (mn < -0.01f || mx > 1.01f) { /* logits */
        mn = FLT_MAX;
        mx = -FLT_MAX;
        for (size_t i = 0; i < n; i++) {
            v[i] = 1.0f / (1.0f + expf(-v[i]));
            if (v[i] < mn) mn = v[i];
            if (v[i] > mx) mx = v[i];
        }
    }
    /* Stretch only when the prediction has both background and subject;
     * stretching an all-subject or all-background matte would invent edges. */
    int stretch = minmax && mn < 0.25f && mx > 0.5f && mx - mn > 0.05f;
    float k = stretch ? 1.0f / (mx - mn) : 1.0f, o = stretch ? mn : 0.0f;
    for (size_t i = 0; i < n; i++) v[i] = ie_clampf((v[i] - o) * k, 0, 1);
}

/* ---- guided-filter matte upsampling ------------------------------------ */

typedef struct {
    float *pl[17];
    int n;
    float eps;
} gf_ctx;

/* Planes: 0-2 I, 3 p, 4-6 mean I, 7 mean p, 8-10 mean Ip, 11-16 mean II.
 * Writes a (3 planes) to 8-10 and b to 7. */
static void gf_coeffs(void *p, int i0, int i1)
{
    gf_ctx *c = (gf_ctx *)p;
    float **P = c->pl;
    const float eps = c->eps;
    for (int i = i0; i < i1; i++) {
        float mr = P[4][i], mg = P[5][i], mb = P[6][i], mp = P[7][i];
        float cr = P[8][i] - mr * mp, cg = P[9][i] - mg * mp, cb = P[10][i] - mb * mp;
        float srr = P[11][i] - mr * mr + eps, srg = P[12][i] - mr * mg, srb = P[13][i] - mr * mb;
        float sgg = P[14][i] - mg * mg + eps, sgb = P[15][i] - mg * mb, sbb = P[16][i] - mb * mb + eps;
        float irr = sgg * sbb - sgb * sgb, irg = sgb * srb - srg * sbb, irb = srg * sgb - sgg * srb;
        float igg = srr * sbb - srb * srb, igb = srb * srg - srr * sgb, ibb = srr * sgg - srg * srg;
        float det = srr * irr + srg * irg + srb * irb;
        float inv = fabsf(det) > 1e-12f ? 1.0f / det : 0.0f;
        float ar = (irr * cr + irg * cg + irb * cb) * inv;
        float ag = (irg * cr + igg * cg + igb * cb) * inv;
        float ab = (irb * cr + igb * cg + ibb * cb) * inv;
        P[8][i] = ar;
        P[9][i] = ag;
        P[10][i] = ab;
        P[7][i] = mp - ar * mr - ag * mg - ab * mb;
    }
}

typedef struct {
    ie_image *img;
    const float *ar, *ag, *ab, *b; /* ar == NULL: plain bilinear of b */
    int gw, gh;
    const int *x0, *x1;
    const float *fx;
} up_ctx;

static void upsample_rows(void *p, int y0, int y1)
{
    const up_ctx *c = (const up_ctx *)p;
    const int W = c->img->w, H = c->img->h, gw = c->gw, gh = c->gh;
    for (int y = y0; y < y1; y++) {
        float v = (y + 0.5f) * gh / H - 0.5f;
        if (v < 0) v = 0;
        int r0 = (int)v;
        if (r0 > gh - 1) r0 = gh - 1;
        int r1 = r0 + 1 < gh ? r0 + 1 : gh - 1;
        float fy = v - r0;
        const size_t o0 = (size_t)r0 * gw, o1 = (size_t)r1 * gw;
        uint8_t *s = IE_ROW(c->img, y);
        for (int x = 0; x < W; x++, s += 4) {
            const int a = c->x0[x], b = c->x1[x];
            const float fx = c->fx[x];
            const float w00 = (1 - fx) * (1 - fy), w01 = fx * (1 - fy), w10 = (1 - fx) * fy, w11 = fx * fy;
#define BL(P) ((P)[o0 + a] * w00 + (P)[o0 + b] * w01 + (P)[o1 + a] * w10 + (P)[o1 + b] * w11)
            float q = BL(c->b);
            if (c->ar) q += BL(c->ar) * (s[0] * (1 / 255.0f)) + BL(c->ag) * (s[1] * (1 / 255.0f)) + BL(c->ab) * (s[2] * (1 / 255.0f));
#undef BL
            q = ie_clampf(q, 0, 1);
            s[3] = (uint8_t)lrintf(q * s[3]);
        }
    }
}

int ie_matte_apply(ie_image *img, const float *matte, int mw, int mh, int radius, float eps)
{
    if (!img || !matte || mw <= 0 || mh <= 0) return IE_EINVAL;
    const int W = img->w, H = img->h;
    /* Work at the matte's resolution, or the image's if that is smaller. */
    int gw = mw, gh = mh;
    if ((size_t)W * H <= (size_t)mw * mh) gw = W, gh = H;
    const size_t n = (size_t)gw * gh;
    /* When the matte is stretched over a much larger photo the low-resolution
     * guide is blurrier, so wider windows are needed to lock onto edges; but a
     * window must stay small next to the grid itself. */
    if (radius > 0) {
        double s = sqrt((double)W * H / (double)n);
        if (s > 1) radius = (int)lrint(radius * pow(s < 6 ? s : 6, 0.75));
        int cap = (gw < gh ? gw : gh) / 64;
        if (radius > (cap > 1 ? cap : 1)) radius = cap > 1 ? cap : 1;
    }
    int err = IE_ENOMEM;
    float *mem = NULL;
    uint8_t *small = NULL;
    int *xi = NULL;

    int planes = radius > 0 ? 17 : 1;
    mem = (float *)malloc(sizeof(float) * n * planes);
    xi = (int *)malloc(sizeof(int) * 2 * (size_t)W + sizeof(float) * (size_t)W);
    if (!mem || !xi) goto done;
    float *P[17];
    for (int k = 0; k < planes; k++) P[k] = mem + n * k;
    float *pmat = radius > 0 ? P[3] : P[0];

    if (gw == mw && gh == mh) memcpy(pmat, matte, sizeof(float) * n);
    else if ((err = ie_resize_f32(matte, mw, mh, pmat, gw, gh, 1)) != IE_OK) goto done;

    up_ctx u;
    memset(&u, 0, sizeof u);
    if (radius > 0) {
        const uint8_t *g = img->px;
        if (gw != W || gh != H) {
            small = (uint8_t *)malloc(n * 4);
            if (!small) {
                err = IE_ENOMEM;
                goto done;
            }
            if ((err = ie_resize_rgba(img->px, W, H, small, gw, gh, 0)) != IE_OK) goto done;
            g = small;
        }
        for (size_t i = 0; i < n; i++) {
            float r = g[i * 4] / 255.0f, gg = g[i * 4 + 1] / 255.0f, b = g[i * 4 + 2] / 255.0f, pv = P[3][i];
            P[0][i] = r, P[1][i] = gg, P[2][i] = b;
            P[8][i] = r * pv, P[9][i] = gg * pv, P[10][i] = b * pv;
            P[11][i] = r * r, P[12][i] = r * gg, P[13][i] = r * b;
            P[14][i] = gg * gg, P[15][i] = gg * b, P[16][i] = b * b;
        }
        static const int src_of[13] = {0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        static const int dst_of[13] = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        for (int k = 0; k < 13; k++)
            if ((err = ie_box_mean(P[src_of[k]], P[dst_of[k]], gw, gh, radius)) != IE_OK) goto done;
        gf_ctx gc;
        memcpy(gc.pl, P, sizeof P);
        gc.n = (int)n;
        gc.eps = eps > 0 ? eps : 1e-4f;
        ie_parallel_for((int)n, 4096, gf_coeffs, &gc);
        for (int k = 7; k <= 10; k++)
            if ((err = ie_box_mean(P[k], P[k], gw, gh, radius)) != IE_OK) goto done;
        u.ar = P[8];
        u.ag = P[9];
        u.ab = P[10];
        u.b = P[7];
    } else {
        u.b = P[0];
    }

    int *x0 = xi, *x1 = xi + W;
    float *fx = (float *)(xi + 2 * (size_t)W);
    for (int x = 0; x < W; x++) {
        float t = (x + 0.5f) * gw / W - 0.5f;
        if (t < 0) t = 0;
        int a = (int)t;
        if (a > gw - 1) a = gw - 1;
        x0[x] = a;
        x1[x] = a + 1 < gw ? a + 1 : gw - 1;
        fx[x] = t - a;
    }
    u.img = img;
    u.gw = gw;
    u.gh = gh;
    u.x0 = x0;
    u.x1 = x1;
    u.fx = fx;
    ie_parallel_for(H, 8, upsample_rows, &u);
    err = IE_OK;
done:
    free(mem);
    free(small);
    free(xi);
    return err;
}

/* ---- alpha morphology, feathering, islands ----------------------------- */

/* van Herk / Gil-Werman running max (or min) over a 2r+1 window, O(1) per
 * sample regardless of r. g and h need n + 2r bytes each. */
static void minmax1d(const uint8_t *in, uint8_t *out, int n, int r, int is_max, uint8_t *g, uint8_t *h)
{
    const int k = 2 * r + 1, m = n + 2 * r;
    const uint8_t neutral = is_max ? 0 : 255;
#define PAD(i) ((i) >= r && (i) < n + r ? in[(i) - r] : neutral)
#define OP(a, b) (is_max ? ((a) > (b) ? (a) : (b)) : ((a) < (b) ? (a) : (b)))
    for (int i = 0; i < m; i++) g[i] = i % k == 0 ? PAD(i) : OP(g[i - 1], PAD(i));
    for (int i = m - 1; i >= 0; i--) h[i] = (i == m - 1 || (i + 1) % k == 0) ? PAD(i) : OP(h[i + 1], PAD(i));
    for (int j = 0; j < n; j++) out[j] = OP(h[j], g[j + 2 * r]);
#undef PAD
#undef OP
}

typedef struct {
    ie_image *img;
    uint8_t *plane; /* optional separate plane (w*h) instead of the alpha channel */
    int r, is_max;
} morph_ctx;

static inline uint8_t *morph_px(const morph_ctx *c, size_t i)
{
    return c->plane ? c->plane + i : c->img->px + i * 4 + 3;
}

static void morph_rows(void *p, int y0, int y1)
{
    morph_ctx *c = (morph_ctx *)p;
    const int w = c->img->w, r = c->r;
    uint8_t *buf = (uint8_t *)malloc((size_t)w * 2 + (size_t)(w + 2 * r) * 2);
    if (!buf) return;
    uint8_t *in = buf, *out = buf + w, *g = buf + 2 * (size_t)w, *h = g + w + 2 * r;
    for (int y = y0; y < y1; y++) {
        size_t base = (size_t)y * w;
        for (int x = 0; x < w; x++) in[x] = *morph_px(c, base + x);
        minmax1d(in, out, w, r, c->is_max, g, h);
        for (int x = 0; x < w; x++) *morph_px(c, base + x) = out[x];
    }
    free(buf);
}

static void morph_cols(void *p, int x0, int x1)
{
    morph_ctx *c = (morph_ctx *)p;
    const int w = c->img->w, H = c->img->h, r = c->r;
    uint8_t *buf = (uint8_t *)malloc((size_t)H * 2 + (size_t)(H + 2 * r) * 2);
    if (!buf) return;
    uint8_t *in = buf, *out = buf + H, *g = buf + 2 * (size_t)H, *h = g + H + 2 * r;
    for (int x = x0; x < x1; x++) {
        for (int y = 0; y < H; y++) in[y] = *morph_px(c, (size_t)y * w + x);
        minmax1d(in, out, H, r, c->is_max, g, h);
        for (int y = 0; y < H; y++) *morph_px(c, (size_t)y * w + x) = out[y];
    }
    free(buf);
}

static void morph(ie_image *img, uint8_t *plane, int radius)
{
    morph_ctx c;
    c.img = img;
    c.plane = plane;
    c.r = radius < 0 ? -radius : radius;
    c.is_max = radius > 0;
    ie_parallel_for(img->h, 16, morph_rows, &c);
    ie_parallel_for(img->w, 16, morph_cols, &c);
}

int ie_alpha_grow(ie_image *img, int radius)
{
    if (!img) return IE_EINVAL;
    if (radius) morph(img, NULL, radius);
    return IE_OK;
}

int ie_alpha_feather(ie_image *img, float sigma)
{
    if (!img) return IE_EINVAL;
    if (sigma <= 0.1f) return IE_OK;
    size_t n = (size_t)img->w * img->h;
    float *a = (float *)malloc(sizeof(float) * n);
    if (!a) return IE_ENOMEM;
    for (size_t i = 0; i < n; i++) a[i] = img->px[i * 4 + 3];
    int r = ie_blur_f32(a, img->w, img->h, 1, sigma);
    for (size_t i = 0; r == IE_OK && i < n; i++) img->px[i * 4 + 3] = (uint8_t)lrintf(ie_clampf(a[i], 0, 255));
    free(a);
    return r;
}

/* Drop 8-connected islands of the solid part (value >= 128) of a mask that are
 * smaller than min_area times the largest one; soft edges around surviving
 * islands are kept. `a` is the mask, `stride` bytes apart (4 for an alpha
 * channel); img only supplies the size. */
static int clean_islands(ie_image *img, uint8_t *a, size_t stride, float min_area)
{
    const int W = img->w, H = img->h;
    const size_t n = (size_t)W * H;
    int32_t *label = (int32_t *)calloc(n, sizeof(int32_t));
    size_t cap = 1 << 16, nlabels = 0;
    size_t *area = (size_t *)malloc(sizeof(size_t) * cap);
    size_t scap = 1 << 16;
    int32_t *stack = (int32_t *)malloc(sizeof(int32_t) * scap);
    uint8_t *keep = (uint8_t *)malloc(n);
    int err = IE_ENOMEM;
    if (!label || !area || !stack || !keep) goto done;

    /* Label 8-connected components of the solid (alpha >= 128) part. */
    for (size_t start = 0; start < n; start++) {
        if (label[start] || a[start * stride] < 128) continue;
        if (nlabels + 1 >= cap) {
            size_t *t = (size_t *)realloc(area, sizeof(size_t) * cap * 2);
            if (!t) goto done;
            area = t;
            cap *= 2;
        }
        int32_t id = (int32_t)++nlabels;
        size_t sp = 0, count = 0;
        stack[sp++] = (int32_t)start;
        label[start] = id;
        while (sp) {
            int32_t i = stack[--sp];
            count++;
            int x = i % W, y = i / W;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = x + dx, ny = y + dy;
                    if ((dx | dy) == 0 || nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
                    size_t j = (size_t)ny * W + nx;
                    if (label[j] || a[j * stride] < 128) continue;
                    label[j] = id;
                    if (sp == scap) {
                        int32_t *t = (int32_t *)realloc(stack, sizeof(int32_t) * scap * 2);
                        if (!t) goto done;
                        stack = t;
                        scap *= 2;
                    }
                    stack[sp++] = (int32_t)j;
                }
        }
        area[id] = count;
    }
    size_t largest = 0;
    for (size_t k = 1; k <= nlabels; k++)
        if (area[k] > largest) largest = area[k];
    double min_px = (double)largest * (min_area > 0 ? min_area : 0.01);
    for (size_t i = 0; i < n; i++) keep[i] = label[i] && area[label[i]] >= min_px ? 255 : 0;

    /* Keep the soft edge around surviving components, drop everything else. */
    int grow = 2 + (W > H ? W : H) / 500;
    morph(img, keep, grow);
    for (size_t i = 0; i < n; i++)
        if (!keep[i]) a[i * stride] = 0;
    err = IE_OK;
done:
    free(label);
    free(area);
    free(stack);
    free(keep);
    return err;
}

int ie_alpha_clean(ie_image *img, float min_area)
{
    if (!img) return IE_EINVAL;
    return clean_islands(img, img->px + 3, 4, min_area);
}

/* ---- foreground color estimation ---------------------------------------- */

/*
 * Blur-fusion foreground estimation (Forte & Pitie, "Approximate Fast
 * Foreground Colour Estimation", ICIP 2021): two passes of
 *   F = blur(F*a)/blur(a) + a * (I - a*blur(F*a)/blur(a) - (1-a)*blur(B*(1-a))/blur(1-a))
 * with a large then a small box radius. The large pass is smooth, so it runs on
 * a downsampled grid. The small pass only matters near semi-transparent pixels,
 * so it runs per tile and skips tiles without any.
 */

typedef struct {
    const ie_image *img;
    int s, lw, lh;
    float *pl; /* 7 planes: a, F*a (3), B*(1-a) (3); later F1 (3) and B1 (3) */
} dc_low;

static void dc_downsample(void *p, int y0, int y1)
{
    dc_low *c = (dc_low *)p;
    const ie_image *img = c->img;
    const size_t n = (size_t)c->lw * c->lh;
    for (int ly = y0; ly < y1; ly++)
        for (int lx = 0; lx < c->lw; lx++) {
            double acc[7] = {0, 0, 0, 0, 0, 0, 0};
            int cnt = 0;
            for (int y = ly * c->s; y < (ly + 1) * c->s && y < img->h; y++) {
                const uint8_t *q = IE_ROW(img, y) + (size_t)lx * c->s * 4;
                for (int x = lx * c->s; x < (lx + 1) * c->s && x < img->w; x++, q += 4) {
                    double a = q[3] / 255.0, r = q[0] / 255.0, g = q[1] / 255.0, b = q[2] / 255.0;
                    acc[0] += a;
                    acc[1] += r * a, acc[2] += g * a, acc[3] += b * a;
                    acc[4] += r * (1 - a), acc[5] += g * (1 - a), acc[6] += b * (1 - a);
                    cnt++;
                }
            }
            size_t i = (size_t)ly * c->lw + lx;
            for (int k = 0; k < 7; k++) c->pl[k * n + i] = (float)(acc[k] / cnt);
        }
}

typedef struct {
    const ie_image *img;
    uint8_t *out;
    const float *F1, *B1; /* low-res, 3 planes each */
    int lw, lh, s, r, tile, tiles_x;
} dc_tile;

static void dc_tiles(void *p, int t0, int t1)
{
    const dc_tile *c = (const dc_tile *)p;
    const ie_image *img = c->img;
    const int W = img->w, H = img->h, r = c->r;
    const size_t ln = (size_t)c->lw * c->lh;
    const int ext = c->tile + 2 * r;
    double *sat = (double *)malloc(sizeof(double) * 7 * (size_t)(ext + 1) * (ext + 1));
    float *loc = (float *)malloc(sizeof(float) * 4 * (size_t)ext * ext); /* I (3) + a */
    if (!sat || !loc) goto out;

    for (int t = t0; t < t1; t++) {
        int x0 = (t % c->tiles_x) * c->tile, y0 = (t / c->tiles_x) * c->tile;
        int x1 = x0 + c->tile > W ? W : x0 + c->tile, y1 = y0 + c->tile > H ? H : y0 + c->tile;
        int ex0 = x0 - r < 0 ? 0 : x0 - r, ey0 = y0 - r < 0 ? 0 : y0 - r;
        int ex1 = x1 + r > W ? W : x1 + r, ey1 = y1 + r > H ? H : y1 + r;
        int ew = ex1 - ex0, eh = ey1 - ey0;

        int partial = 0;
        for (int y = y0; y < y1 && !partial; y++) {
            const uint8_t *q = IE_ROW(img, y) + (size_t)x0 * 4 + 3;
            for (int x = x0; x < x1; x++, q += 4)
                if (*q > 0 && *q < 255) {
                    partial = 1;
                    break;
                }
        }
        if (!partial) continue;

        /* Summed-area tables of a, F1*a and B1*(1-a) over the extended tile. */
        const size_t sw = (size_t)ew + 1, plane = sw * (eh + 1);
        for (int k = 0; k < 7; k++) memset(sat + k * plane, 0, sizeof(double) * sw);
        for (int y = 0; y < eh; y++) {
            double run[7] = {0, 0, 0, 0, 0, 0, 0};
            const uint8_t *q = IE_ROW(img, ey0 + y) + (size_t)ex0 * 4;
            float ly = (ey0 + y + 0.5f) / c->s - 0.5f;
            if (ly < 0) ly = 0;
            int ly0 = (int)ly;
            if (ly0 > c->lh - 1) ly0 = c->lh - 1;
            int ly1 = ly0 + 1 < c->lh ? ly0 + 1 : c->lh - 1;
            float fy = ly - ly0;
            for (int k = 0; k < 7; k++) sat[k * plane + (size_t)(y + 1) * sw] = 0;
            for (int x = 0; x < ew; x++, q += 4) {
                float a = q[3] / 255.0f, I[3] = {q[0] / 255.0f, q[1] / 255.0f, q[2] / 255.0f};
                float *L = loc + ((size_t)y * ew + x) * 4;
                L[0] = I[0], L[1] = I[1], L[2] = I[2], L[3] = a;
                float lx = (ex0 + x + 0.5f) / c->s - 0.5f;
                if (lx < 0) lx = 0;
                int lx0 = (int)lx;
                if (lx0 > c->lw - 1) lx0 = c->lw - 1;
                int lx1 = lx0 + 1 < c->lw ? lx0 + 1 : c->lw - 1;
                float fx = lx - lx0;
                size_t i00 = (size_t)ly0 * c->lw + lx0, i01 = (size_t)ly0 * c->lw + lx1;
                size_t i10 = (size_t)ly1 * c->lw + lx0, i11 = (size_t)ly1 * c->lw + lx1;
                float w00 = (1 - fx) * (1 - fy), w01 = fx * (1 - fy), w10 = (1 - fx) * fy, w11 = fx * fy;
                double v[7];
                v[0] = a;
                for (int k = 0; k < 3; k++) {
                    const float *F = c->F1 + k * ln, *B = c->B1 + k * ln;
                    float fh = F[i00] * w00 + F[i01] * w01 + F[i10] * w10 + F[i11] * w11;
                    float bh = B[i00] * w00 + B[i01] * w01 + B[i10] * w10 + B[i11] * w11;
                    float f1 = ie_clampf(fh + a * (I[k] - a * fh - (1 - a) * bh), 0, 1);
                    v[1 + k] = f1 * a;
                    v[4 + k] = bh * (1 - a);
                }
                for (int k = 0; k < 7; k++) {
                    run[k] += v[k];
                    sat[k * plane + (size_t)(y + 1) * sw + x + 1] = sat[k * plane + (size_t)y * sw + x + 1] + run[k];
                }
            }
        }

        for (int y = y0; y < y1; y++) {
            uint8_t *o = c->out + ((size_t)y * W + x0) * 4;
            int ya = (y - r < ey0 ? ey0 : y - r) - ey0, yb = (y + r >= ey1 ? ey1 - 1 : y + r) - ey0 + 1;
            for (int x = x0; x < x1; x++, o += 4) {
                const float *L = loc + ((size_t)(y - ey0) * ew + (x - ex0)) * 4;
                float a = L[3];
                if (a <= 0 || a >= 1) continue;
                int xa = (x - r < ex0 ? ex0 : x - r) - ex0, xb = (x + r >= ex1 ? ex1 - 1 : x + r) - ex0 + 1;
                double cnt = (double)(xb - xa) * (yb - ya);
                double m[7];
                for (int k = 0; k < 7; k++) {
                    const double *S = sat + k * plane;
                    m[k] = (S[(size_t)yb * sw + xb] - S[(size_t)ya * sw + xb] - S[(size_t)yb * sw + xa] +
                            S[(size_t)ya * sw + xa]) / cnt;
                }
                for (int k = 0; k < 3; k++) {
                    double fh = m[1 + k] / (m[0] + 1e-5), bh = m[4 + k] / (1 - m[0] + 1e-5);
                    double f = fh + a * (L[k] - a * fh - (1 - a) * bh);
                    o[k] = (uint8_t)lrint(ie_clampf((float)f, 0, 1) * 255.0);
                }
            }
        }
    }
out:
    free(sat);
    free(loc);
}

int ie_decontaminate(ie_image *img)
{
    if (!img) return IE_EINVAL;
    const int W = img->w, H = img->h, longest = W > H ? W : H;
    const float scale = longest > 2000 ? longest / 2000.0f : 1.0f;
    const int r1 = (int)lrintf(90 * scale), r2 = (int)lrintf(6 * scale);

    dc_low lo;
    lo.img = img;
    lo.s = (int)ceil(sqrt((double)W * H / 1.0e6));
    if (lo.s < 1) lo.s = 1;
    lo.lw = (W + lo.s - 1) / lo.s;
    lo.lh = (H + lo.s - 1) / lo.s;
    const size_t ln = (size_t)lo.lw * lo.lh;
    lo.pl = (float *)malloc(sizeof(float) * ln * 7);
    uint8_t *out = (uint8_t *)malloc((size_t)W * H * 4);
    if (!lo.pl || !out) {
        free(lo.pl);
        free(out);
        return IE_ENOMEM;
    }
    ie_parallel_for(lo.lh, 4, dc_downsample, &lo);
    int rl = (int)lrintf((float)r1 / lo.s);
    for (int k = 0; k < 7; k++) ie_box_mean(lo.pl + k * ln, lo.pl + k * ln, lo.lw, lo.lh, rl < 1 ? 1 : rl);
    /* F1 = blur(I*a)/blur(a) into planes 1-3, B1 = blur(I*(1-a))/blur(1-a) into 4-6. */
    for (size_t i = 0; i < ln; i++) {
        float a = lo.pl[i];
        for (int k = 0; k < 3; k++) {
            lo.pl[(1 + k) * ln + i] /= a + 1e-5f;
            lo.pl[(4 + k) * ln + i] /= 1 - a + 1e-5f;
        }
    }

    memcpy(out, img->px, (size_t)W * H * 4);
    dc_tile t;
    t.img = img;
    t.out = out;
    t.F1 = lo.pl + ln;
    t.B1 = lo.pl + 4 * ln;
    t.lw = lo.lw;
    t.lh = lo.lh;
    t.s = lo.s;
    t.r = r2 < 1 ? 1 : r2;
    t.tile = 64;
    t.tiles_x = (W + t.tile - 1) / t.tile;
    int tiles = t.tiles_x * ((H + t.tile - 1) / t.tile);
    ie_parallel_for(tiles, 1, dc_tiles, &t);
    free(img->px);
    img->px = out;
    free(lo.pl);
    return IE_OK;
}

/* ---- color key --------------------------------------------------------- */

static void lab_init(float *lin)
{
    for (int i = 0; i < 256; i++) {
        float v = i / 255.0f;
        lin[i] = v <= 0.04045f ? v / 12.92f : powf((v + 0.055f) / 1.055f, 2.4f);
    }
}

static inline float lab_f(float t) { return t > 0.008856f ? cbrtf(t) : 7.787f * t + 16.0f / 116.0f; }

static void to_lab(const float *lin, const uint8_t *p, float *lab)
{
    float r = lin[p[0]], g = lin[p[1]], b = lin[p[2]];
    float x = (0.4124f * r + 0.3576f * g + 0.1805f * b) / 0.95047f;
    float y = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    float z = (0.0193f * r + 0.1192f * g + 0.9505f * b) / 1.08883f;
    float fx = lab_f(x), fy = lab_f(y), fz = lab_f(z);
    lab[0] = 116 * fy - 16;
    lab[1] = 500 * (fx - fy);
    lab[2] = 200 * (fy - fz);
}

typedef struct {
    ie_image *img;
    uint8_t *k; /* 0 = background, 255 = keep */
    float key[3], tol, soft;
    float lin[256]; /* sRGB -> linear */
} key_ctx;

static void key_rows(void *p, int y0, int y1)
{
    key_ctx *c = (key_ctx *)p;
    const int w = c->img->w;
    for (int y = y0; y < y1; y++) {
        const uint8_t *s = IE_ROW(c->img, y);
        uint8_t *k = c->k + (size_t)y * w;
        for (int x = 0; x < w; x++, s += 4) {
            float lab[3];
            to_lab(c->lin, s, lab);
            float d0 = lab[0] - c->key[0], d1 = lab[1] - c->key[1], d2 = lab[2] - c->key[2];
            float d = sqrtf(d0 * d0 + d1 * d1 + d2 * d2);
            float t = ie_clampf((d - c->tol) / c->soft, 0, 1);
            k[x] = (uint8_t)lrintf(t * t * (3 - 2 * t) * 255);
        }
    }
}

int ie_key_background(ie_image *img, int auto_color, uint32_t rgb, float tolerance, float softness, int connected)
{
    if (!img) return IE_EINVAL;
    const int W = img->w, H = img->h;
    const size_t n = (size_t)W * H;

    uint8_t key[4] = {(uint8_t)(rgb >> 16), (uint8_t)(rgb >> 8), (uint8_t)rgb, 255};
    if (auto_color) { /* median of the border pixels */
        uint32_t hist[3][256] = {{0}};
        size_t cnt = 0;
        for (int x = 0; x < W; x++)
            for (int e = 0; e < 2; e++) {
                const uint8_t *q = IE_ROW(img, e ? H - 1 : 0) + (size_t)x * 4;
                for (int k = 0; k < 3; k++) hist[k][q[k]]++;
                cnt++;
            }
        for (int y = 1; y < H - 1; y++)
            for (int e = 0; e < 2; e++) {
                const uint8_t *q = IE_ROW(img, y) + (size_t)(e ? W - 1 : 0) * 4;
                for (int k = 0; k < 3; k++) hist[k][q[k]]++;
                cnt++;
            }
        for (int k = 0; k < 3; k++) {
            size_t acc = 0;
            int v = 0;
            while (v < 255 && (acc += hist[k][v]) <= cnt / 2) v++;
            key[k] = (uint8_t)v;
        }
    }

    key_ctx c;
    c.img = img;
    c.k = (uint8_t *)malloc(n);
    if (!c.k) return IE_ENOMEM;
    lab_init(c.lin);
    to_lab(c.lin, key, c.key);
    c.tol = tolerance > 0 ? tolerance : 0;
    c.soft = softness > 0.5f ? softness : 0.5f;
    ie_parallel_for(H, 16, key_rows, &c);

    if (connected) { /* only background reachable from the border counts */
        uint8_t *seen = (uint8_t *)calloc(n, 1);
        size_t cap = (size_t)2 * (W + H) + 1024, sp = 0;
        int32_t *stack = (int32_t *)malloc(sizeof(int32_t) * cap);
        if (!seen || !stack) {
            free(seen);
            free(stack);
            free(c.k);
            return IE_ENOMEM;
        }
#define PUSH(i)                                                                            \
    do {                                                                                   \
        size_t i_ = (i);                                                                   \
        if (!seen[i_] && c.k[i_] < 255) {                                                  \
            seen[i_] = 1;                                                                  \
            if (sp == cap) {                                                               \
                int32_t *t_ = (int32_t *)realloc(stack, sizeof(int32_t) * cap * 2);        \
                if (!t_) break;                                                            \
                stack = t_;                                                                \
                cap *= 2;                                                                  \
            }                                                                              \
            stack[sp++] = (int32_t)i_;                                                     \
        }                                                                                  \
    } while (0)
        for (int x = 0; x < W; x++) {
            PUSH((size_t)x);
            PUSH((size_t)(H - 1) * W + x);
        }
        for (int y = 0; y < H; y++) {
            PUSH((size_t)y * W);
            PUSH((size_t)y * W + W - 1);
        }
        while (sp) {
            int32_t i = stack[--sp];
            int x = i % W, y = i / W;
            if (x > 0) PUSH((size_t)i - 1);
            if (x < W - 1) PUSH((size_t)i + 1);
            if (y > 0) PUSH((size_t)i - W);
            if (y < H - 1) PUSH((size_t)i + W);
        }
#undef PUSH
        for (size_t i = 0; i < n; i++)
            if (!seen[i]) c.k[i] = 255;
        free(seen);
        free(stack);
    }

    /* Noise right at the tolerance leaves specks; drop the small ones. */
    int r = clean_islands(img, c.k, 1, 0.01f);
    if (r == IE_OK)
        for (size_t i = 0; i < n; i++) img->px[i * 4 + 3] = (uint8_t)((img->px[i * 4 + 3] * c.k[i] + 127) / 255);
    free(c.k);
    return r;
}

/* ---- compositing ------------------------------------------------------- */

typedef struct {
    ie_image *img;
    const ie_image *bg;
} over_ctx;

/* img = img over bg (straight alpha). */
static void over_rows(void *p, int y0, int y1)
{
    over_ctx *c = (over_ctx *)p;
    const int w = c->img->w;
    for (int y = y0; y < y1; y++) {
        uint8_t *s = IE_ROW(c->img, y);
        const uint8_t *b = IE_ROW(c->bg, y);
        for (int x = 0; x < w; x++, s += 4, b += 4) {
            float a = s[3] / 255.0f, ba = b[3] / 255.0f * (1 - a), oa = a + ba;
            if (oa <= 0) {
                s[0] = s[1] = s[2] = s[3] = 0;
                continue;
            }
            for (int k = 0; k < 3; k++) s[k] = (uint8_t)lrintf((s[k] * a + b[k] * ba) / oa);
            s[3] = (uint8_t)lrintf(oa * 255);
        }
    }
}

static void clear_hidden(void *p, int y0, int y1)
{
    over_ctx *c = (over_ctx *)p;
    const int w = c->img->w;
    for (int y = y0; y < y1; y++) {
        uint8_t *s = IE_ROW(c->img, y);
        for (int x = 0; x < w; x++, s += 4)
            if (!s[3]) s[0] = s[1] = s[2] = 0;
    }
}

/* Scale bg to cover w x h (centered crop). */
static ie_image *cover(const ie_image *bg, int w, int h)
{
    double sc = (double)w / bg->w > (double)h / bg->h ? (double)w / bg->w : (double)h / bg->h;
    int sw = (int)ceil(bg->w * sc - 1e-9), sh = (int)ceil(bg->h * sc - 1e-9);
    if (sw < w) sw = w;
    if (sh < h) sh = h;
    ie_image *t = ie_clone(bg);
    if (!t) return NULL;
    if (ie_resize(t, sw, sh) || ie_crop(t, (sw - w) / 2, (sh - h) / 2, w, h)) {
        ie_free(t);
        return NULL;
    }
    return t;
}

int ie_finish_cutout(ie_image *img, const ie_bg_options *opt)
{
    if (!img) return IE_EINVAL;
    ie_bg_options def;
    memset(&def, 0, sizeof def);
    def.decontaminate = 1;
    if (!opt) opt = &def;

    ie_image *bg = NULL;
    if (opt->mode == IE_BG_BLUR) {
        /* Blur the background only: weight by (1 - alpha) so the subject does
         * not smear into it (normalized convolution via premultiplied blur). */
        bg = ie_clone(img);
        if (!bg) return IE_ENOMEM;
        ie_invert(bg, IE_INVERT_ALPHA);
        int r = ie_blur(bg, opt->blur > 0 ? opt->blur : 12.0f);
        if (r) {
            ie_free(bg);
            return r;
        }
        for (size_t i = 0; i < (size_t)bg->w * bg->h; i++) bg->px[i * 4 + 3] = 255;
    } else if (opt->mode == IE_BG_IMAGE) {
        if (!opt->image) return IE_EINVAL;
        bg = cover(opt->image, img->w, img->h);
        if (!bg) return IE_ENOMEM;
    }

    int r = opt->decontaminate ? ie_decontaminate(img) : IE_OK;
    if (r == IE_OK) {
        over_ctx c = {img, bg};
        switch (opt->mode) {
        case IE_BG_COLOR: r = ie_flatten(img, opt->color); break;
        case IE_BG_BLUR:
        case IE_BG_IMAGE: ie_parallel_for(img->h, 16, over_rows, &c); break;
        default: ie_parallel_for(img->h, 16, clear_hidden, &c); break;
        }
    }
    ie_free(bg);
    return r;
}
