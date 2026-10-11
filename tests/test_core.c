/* Unit tests for the imgedit core. Build with build.ps1 / make test. */
#include "../src/imgedit.h"
#include "../src/parallel.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures, checks;

#define CHECK(cond)                                                              \
    do {                                                                         \
        checks++;                                                                \
        if (!(cond)) {                                                           \
            failures++;                                                          \
            fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
        }                                                                        \
    } while (0)

static uint32_t rng = 12345;
static uint8_t rnd(void)
{
    rng = rng * 1664525u + 1013904223u;
    return (uint8_t)(rng >> 24);
}

static ie_image *make(int w, int h, int kind)
{
    ie_image *img = ie_new(w, h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            uint8_t *p = img->px + ((size_t)y * w + x) * 4;
            switch (kind) {
            case 0: /* noise with alpha */
                p[0] = rnd(), p[1] = rnd(), p[2] = rnd(), p[3] = rnd();
                break;
            case 1: /* smooth opaque gradient */
                p[0] = (uint8_t)(x * 255 / (w > 1 ? w - 1 : 1)), p[1] = (uint8_t)(y * 255 / (h > 1 ? h - 1 : 1)), p[2] = (uint8_t)((x + y) & 255), p[3] = 255;
                break;
            case 2: /* gray */
                p[0] = p[1] = p[2] = (uint8_t)((x * 7 + y * 3) & 255), p[3] = 255;
                break;
            default: /* gray with alpha */
                p[0] = p[1] = p[2] = (uint8_t)(x & 255), p[3] = (uint8_t)(y & 255);
            }
        }
    return img;
}

static int same(const ie_image *a, const ie_image *b)
{
    return a->w == b->w && a->h == b->h && memcmp(a->px, b->px, (size_t)a->w * a->h * 4) == 0;
}

static double psnr(const ie_image *a, const ie_image *b)
{
    double se = 0;
    size_t n = (size_t)a->w * a->h;
    for (size_t i = 0; i < n; i++)
        for (int c = 0; c < 3; c++) {
            double d = (double)a->px[i * 4 + c] - b->px[i * 4 + c];
            se += d * d;
        }
    double mse = se / (n * 3.0);
    return mse == 0 ? 99 : 10 * log10(255.0 * 255.0 / mse);
}

static ie_image *roundtrip(const ie_image *img, int fmt, int quality)
{
    uint8_t *buf;
    size_t len;
    ie_image *out = NULL;
    if (ie_encode(img, fmt, quality, &buf, &len) != IE_OK) return NULL;
    if (ie_decode(buf, len, &out) != IE_OK) out = NULL;
    ie_buf_free(buf);
    return out;
}

static void test_png(void)
{
    puts("png");
    static const int sizes[][2] = {{1, 1}, {7, 3}, {300, 200}, {3000, 2000}};
    for (int kind = 0; kind < 4; kind++)
        for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; s++) {
            ie_image *img = make(sizes[s][0], sizes[s][1], kind);
            for (int level = 0; level <= 9; level += 3) {
                if (sizes[s][0] == 3000 && level != 6) continue;
                ie_image *back = roundtrip(img, IE_FMT_PNG, level ? level : 0);
                uint8_t *buf;
                size_t len;
                if (level == 0) { /* level 0 = stored; exercise it explicitly */
                    ie_free(back);
                    back = NULL;
                    if (ie_png_encode(img, 0, &buf, &len) == IE_OK) {
                        ie_decode(buf, len, &back);
                        ie_buf_free(buf);
                    }
                }
                CHECK(back && same(img, back));
                ie_free(back);
            }
            ie_free(img);
        }
}

static void test_formats(void)
{
    puts("formats");
    ie_image *rgba = make(97, 61, 0), *rgb = make(97, 61, 1);
    int lossless_alpha[] = {IE_FMT_QOI, IE_FMT_TGA, IE_FMT_BMP, IE_FMT_PAM};
    for (int i = 0; i < 4; i++) {
        ie_image *b = roundtrip(rgba, lossless_alpha[i], 0);
        CHECK(b && same(rgba, b));
        ie_free(b);
    }
    ie_image *b = roundtrip(rgb, IE_FMT_PPM, 0);
    CHECK(b && same(rgb, b));
    ie_free(b);
    b = roundtrip(rgb, IE_FMT_JPEG, 95);
    CHECK(b && b->w == rgb->w && b->h == rgb->h && psnr(rgb, b) > 30);
    ie_free(b);
    CHECK(ie_format_from_name("a/b/photo.JPEG") == IE_FMT_JPEG);
    CHECK(ie_format_from_name(".png") == IE_FMT_PNG);
    CHECK(ie_format_from_name("qoi") == IE_FMT_QOI);
    CHECK(ie_format_from_name("dir.png/file") == IE_FMT_UNKNOWN);
    ie_free(rgba);
    ie_free(rgb);
}

/* A JPEG with an EXIF orientation tag must come back rotated. */
static void test_exif(void)
{
    puts("exif");
    ie_image *img = make(40, 20, 1);
    uint8_t *jpg;
    size_t len;
    CHECK(ie_encode(img, IE_FMT_JPEG, 90, &jpg, &len) == IE_OK);
    static const uint8_t app1[] = {
        0xFF, 0xE1, 0x00, 0x22, 'E', 'x', 'i', 'f', 0, 0,             /* APP1, length 34 */
        'I', 'I', 42, 0, 8, 0, 0, 0,                                 /* TIFF header, IFD at 8 */
        1, 0,                                                        /* one entry */
        0x12, 0x01, 3, 0, 1, 0, 0, 0, 6, 0, 0, 0,                    /* orientation = 6 */
        0, 0, 0, 0};                                                 /* next IFD */
    uint8_t *tagged = (uint8_t *)malloc(len + sizeof app1);
    memcpy(tagged, jpg, 2);
    memcpy(tagged + 2, app1, sizeof app1);
    memcpy(tagged + 2 + sizeof app1, jpg + 2, len - 2);
    ie_image *out = NULL;
    CHECK(ie_decode(tagged, len + sizeof app1, &out) == IE_OK);
    CHECK(out && out->w == 20 && out->h == 40);
    ie_free(out);
    free(tagged);
    ie_buf_free(jpg);
    ie_free(img);
}

static void test_geometry(void)
{
    puts("geometry");
    ie_image *img = make(123, 77, 0), *ref = ie_clone(img);
    for (int i = 0; i < 4; i++) ie_rotate90(img, 1);
    CHECK(same(img, ref));
    ie_rotate90(img, 1);
    CHECK(img->w == 77 && img->h == 123);
    ie_rotate90(img, -1);
    CHECK(same(img, ref));
    ie_flip(img, 1);
    ie_flip(img, 1);
    CHECK(same(img, ref));
    ie_transpose(img);
    ie_transpose(img);
    CHECK(same(img, ref));
    ie_image *a = ie_clone(img), *b = ie_clone(img);
    ie_rotate90(a, 2);
    ie_flip(b, 1);
    ie_flip(b, 0);
    CHECK(same(a, b));
    ie_free(a);
    ie_free(b);

    ie_pad(img, 3, 4, 5, 6, 0);
    CHECK(img->w == 123 + 10 && img->h == 77 + 8);
    ie_crop(img, 6, 3, 123, 77);
    CHECK(same(img, ref));

    /* trim: a transparent border goes away */
    ie_image *t = ie_new(50, 40);
    for (int y = 10; y < 25; y++)
        for (int x = 5; x < 30; x++) memset(t->px + ((size_t)y * 50 + x) * 4, 200, 4);
    ie_trim(t, 0);
    CHECK(t->w == 25 && t->h == 15);
    ie_free(t);

    ie_resize(img, 61, 30);
    CHECK(img->w == 61 && img->h == 30);
    ie_image *flat = ie_new(64, 64);
    for (size_t i = 0; i < 64 * 64; i++) flat->px[i * 4] = 10, flat->px[i * 4 + 1] = 120, flat->px[i * 4 + 2] = 230, flat->px[i * 4 + 3] = 255;
    ie_resize(flat, 100, 37);
    int flat_ok = 1;
    for (size_t i = 0; i < 100 * 37; i++)
        if (abs(flat->px[i * 4] - 10) > 1 || abs(flat->px[i * 4 + 1] - 120) > 1 || abs(flat->px[i * 4 + 2] - 230) > 1) flat_ok = 0;
    CHECK(flat_ok);
    ie_rotate(flat, 30, 0);
    CHECK(flat->w > 100 && flat->h > 37 && flat->px[3] == 0); /* corners transparent */
    ie_free(flat);
    ie_free(img);
    ie_free(ref);
}

static void test_color(void)
{
    puts("color");
    ie_image *img = make(64, 48, 0), *ref = ie_clone(img);
    ie_invert(img, IE_INVERT_RGB);
    CHECK(img->px[0] == 255 - ref->px[0] && img->px[3] == ref->px[3]);
    ie_invert(img, IE_INVERT_RGB);
    CHECK(same(img, ref));
    ie_invert(img, IE_INVERT_ALPHA);
    ie_invert(img, IE_INVERT_ALPHA);
    CHECK(same(img, ref));

    /* Lightness inversion keeps the hue: pure red stays red, white turns black. */
    ie_image *c = ie_new(3, 1);
    uint8_t px[12] = {255, 0, 0, 255, 255, 255, 255, 255, 40, 40, 40, 255};
    memcpy(c->px, px, 12);
    ie_invert(c, IE_INVERT_LIGHTNESS);
    CHECK(c->px[0] == 255 && c->px[1] == 0 && c->px[2] == 0);
    CHECK(c->px[4] == 0 && c->px[5] == 0 && c->px[6] == 0);
    CHECK(c->px[8] == 215);
    ie_free(c);

    ie_adjust a;
    ie_adjust_init(&a);
    CHECK(ie_adjust_is_identity(&a));
    ie_apply_adjust(img, &a);
    CHECK(same(img, ref));
    a.saturation = -1;
    ie_apply_adjust(img, &a);
    int gray = 1;
    for (size_t i = 0; i < 64 * 48; i++)
        if (abs(img->px[i * 4] - img->px[i * 4 + 1]) > 1 || abs(img->px[i * 4 + 1] - img->px[i * 4 + 2]) > 1) gray = 0;
    CHECK(gray);

    ie_image *dull = make(64, 64, 2);
    for (size_t i = 0; i < 64 * 64; i++)
        for (int k = 0; k < 3; k++) dull->px[i * 4 + k] = (uint8_t)(100 + dull->px[i * 4 + k] / 8);
    ie_auto_levels(dull, 0, 0);
    int lo = 255, hi = 0;
    for (size_t i = 0; i < 64 * 64; i++) {
        if (dull->px[i * 4] < lo) lo = dull->px[i * 4];
        if (dull->px[i * 4] > hi) hi = dull->px[i * 4];
    }
    CHECK(lo == 0 && hi == 255);
    ie_free(dull);

    ie_image *bw = make(32, 32, 1);
    ie_threshold(bw, 128);
    int binary = 1;
    for (size_t i = 0; i < 32 * 32 * 4; i++)
        if ((i & 3) != 3 && bw->px[i] != 0 && bw->px[i] != 255) binary = 0;
    CHECK(binary);
    ie_free(bw);

    ie_image *f = make(16, 16, 0);
    ie_flatten(f, 0x102030);
    CHECK(!ie_has_alpha(f));
    ie_free(f);
    ie_free(img);
    ie_free(ref);
}

static void test_filters(void)
{
    puts("filters");
    ie_image *flat = ie_new(80, 60);
    for (size_t i = 0; i < 80 * 60; i++) memset(flat->px + i * 4, 77, 3), flat->px[i * 4 + 3] = 255;
    ie_image *ref = ie_clone(flat);
    ie_blur(flat, 5);
    CHECK(same(flat, ref)); /* blurring a flat image changes nothing */
    ie_sharpen(flat, 1, 1, 0);
    CHECK(same(flat, ref));
    ie_free(ref);

    /* A blurred step edge becomes a monotone ramp. */
    for (int y = 0; y < 60; y++)
        for (int x = 0; x < 80; x++) memset(flat->px + ((size_t)y * 80 + x) * 4, x < 40 ? 0 : 255, 3);
    ie_blur(flat, 3);
    const uint8_t *row = flat->px + 30 * 80 * 4;
    int mono = 1;
    for (int x = 1; x < 80; x++)
        if (row[x * 4] < row[(x - 1) * 4]) mono = 0;
    CHECK(mono && row[0] < 5 && row[79 * 4] > 250 && row[38 * 4] > 60 && row[41 * 4] < 195);
    ie_free(flat);
}

/* Synthetic scene: red disc on a green background. */
static ie_image *disc_scene(int w, int h, float *matte, int mw, int mh)
{
    ie_image *img = ie_new(w, h);
    float cx = w / 2.0f, cy = h / 2.0f, r = h / 4.0f;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            float d = sqrtf((x + 0.5f - cx) * (x + 0.5f - cx) + (y + 0.5f - cy) * (y + 0.5f - cy));
            float a = d < r - 1 ? 1 : d > r + 1 ? 0 : (r + 1 - d) / 2;
            uint8_t *p = img->px + ((size_t)y * w + x) * 4;
            p[0] = (uint8_t)lrintf(220 * a + 30 * (1 - a));
            p[1] = (uint8_t)lrintf(40 * a + 200 * (1 - a));
            p[2] = 40;
            p[3] = 255;
        }
    for (int y = 0; y < mh; y++)
        for (int x = 0; x < mw; x++) {
            float fx = (x + 0.5f) * w / mw, fy = (y + 0.5f) * h / mh;
            float d = sqrtf((fx - cx) * (fx - cx) + (fy - cy) * (fy - cy));
            matte[y * mw + x] = d < r ? 0.95f : 0.05f; /* blocky, slightly off like a model */
        }
    return img;
}

static void test_matte(void)
{
    puts("matte");
    float *m = (float *)malloc(sizeof(float) * 64 * 64);
    const int sizes[][2] = {{400, 300}, {40, 30}}; /* upsampling and downsampling paths */
    for (int s = 0; s < 2; s++)
        for (int radius = 0; radius <= 4; radius += 4) {
            int w = sizes[s][0], h = sizes[s][1];
            ie_image *img = disc_scene(w, h, m, 64, 64);
            ie_nn_output(m, 64 * 64, 1);
            CHECK(ie_matte_apply(img, m, 64, 64, radius, 1e-4f) == IE_OK);
            const uint8_t *center = img->px + ((size_t)(h / 2) * w + w / 2) * 4;
            const uint8_t *corner = img->px;
            CHECK(center[3] > 240 && corner[3] < 15);
            ie_free(img);
        }

    /* Logits get a sigmoid, then the stretch maps the range to 0..1. */
    float v[4] = {-6, -2, 2, 6};
    ie_nn_output(v, 4, 1);
    CHECK(v[0] == 0 && v[3] == 1 && v[1] > 0 && v[1] < 0.5f && v[2] > 0.5f);
    float allfg[3] = {0.9f, 0.95f, 1.0f}; /* no background: must not be stretched */
    ie_nn_output(allfg, 3, 1);
    CHECK(allfg[0] > 0.85f);

    /* Decontamination pulls edge colours toward the subject colour. */
    ie_image *img = disc_scene(300, 300, m, 64, 64);
    for (int y = 0; y < 300; y++)
        for (int x = 0; x < 300; x++) {
            float d = sqrtf((x + 0.5f - 150) * (x + 0.5f - 150) + (y + 0.5f - 150) * (y + 0.5f - 150));
            uint8_t *p = img->px + ((size_t)y * 300 + x) * 4;
            float a = d < 74 ? 1 : d > 76 ? 0 : (76 - d) / 2;
            p[3] = (uint8_t)lrintf(a * 255);
        }
    ie_image *before = ie_clone(img);
    ie_decontaminate(img);
    double green_before = 0, green_after = 0;
    int n = 0;
    for (size_t i = 0; i < 300 * 300; i++) {
        uint8_t a = img->px[i * 4 + 3];
        if (a > 40 && a < 215) {
            green_before += before->px[i * 4 + 1];
            green_after += img->px[i * 4 + 1];
            n++;
        }
    }
    CHECK(n > 0 && green_after < green_before * 0.7);
    ie_free(before);
    ie_free(img);
    free(m);
}

static void test_key_and_clean(void)
{
    puts("key / clean");
    ie_image *img = ie_new(200, 150);
    for (size_t i = 0; i < 200 * 150; i++) img->px[i * 4] = 250, img->px[i * 4 + 1] = 250, img->px[i * 4 + 2] = 250, img->px[i * 4 + 3] = 255;
    for (int y = 40; y < 110; y++)
        for (int x = 60; x < 140; x++) {
            uint8_t *p = img->px + ((size_t)y * 200 + x) * 4;
            p[0] = 30, p[1] = 60, p[2] = 200;
        }
    /* a white hole inside the subject stays (not connected to the border) */
    for (int y = 70; y < 80; y++)
        for (int x = 95; x < 105; x++) memset(img->px + ((size_t)y * 200 + x) * 4, 250, 3);
    CHECK(ie_key_background(img, 1, 0, 10, 8, 1) == IE_OK);
    CHECK(img->px[3] == 0);
    CHECK(img->px[((size_t)50 * 200 + 70) * 4 + 3] == 255);
    CHECK(img->px[((size_t)75 * 200 + 100) * 4 + 3] == 255);
    ie_free(img);

    ie_image *a = ie_new(100, 100);
    for (int y = 10; y < 90; y++)
        for (int x = 10; x < 60; x++) a->px[((size_t)y * 100 + x) * 4 + 3] = 255;
    a->px[((size_t)5 * 100 + 95) * 4 + 3] = 255; /* a stray speck */
    ie_alpha_clean(a, 0.01f);
    CHECK(a->px[((size_t)5 * 100 + 95) * 4 + 3] == 0);
    CHECK(a->px[((size_t)50 * 100 + 30) * 4 + 3] == 255);
    ie_alpha_grow(a, -5);
    CHECK(a->px[((size_t)50 * 100 + 12) * 4 + 3] == 0 && a->px[((size_t)50 * 100 + 30) * 4 + 3] == 255);
    ie_alpha_grow(a, 8);
    CHECK(a->px[((size_t)50 * 100 + 62) * 4 + 3] == 255);
    ie_free(a);
}

typedef struct {
    long long sum;
    int hits[1000];
} pf_ctx;

static void pf_body(void *p, int b, int e)
{
    pf_ctx *c = (pf_ctx *)p;
    for (int i = b; i < e; i++) c->hits[i]++;
}

static void test_parallel(void)
{
    puts("parallel");
    for (int round = 0; round < 50; round++) {
        pf_ctx c;
        memset(&c, 0, sizeof c);
        ie_parallel_for(1000, 1 + round % 7, pf_body, &c);
        int ok = 1;
        for (int i = 0; i < 1000; i++)
            if (c.hits[i] != 1) ok = 0;
        CHECK(ok);
    }
}

int main(void)
{
    test_png();
    test_formats();
    test_exif();
    test_geometry();
    test_color();
    test_filters();
    test_matte();
    test_key_and_clean();
    test_parallel();
    printf("%d checks, %d failed\n", checks, failures);
    return failures ? 1 : 0;
}
