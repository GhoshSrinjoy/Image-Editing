/* Image lifetime, decoding and encoding. */
#include "imgedit.h"
#include "parallel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STBI_NO_STDIO
#define STBI_FAILURE_USERMSG
#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb/stb_image.h"

#define STBI_WRITE_NO_STDIO
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb/stb_image_write.h"

#define QOI_NO_STDIO
#define QOI_IMPLEMENTATION
#include "../third_party/qoi/qoi.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

const char *ie_strerror(int err)
{
    switch (err) {
    case IE_OK: return "ok";
    case IE_ENOMEM: return "out of memory";
    case IE_EINVAL: return "invalid argument";
    case IE_EIO: return "file read/write failed";
    case IE_EFORMAT: return "unrecognised or corrupt image data";
    case IE_EUNSUPPORTED: return "unsupported format or feature";
    default: return "unknown error";
    }
}

/* ---- lifetime ---------------------------------------------------------- */

ie_image *ie_new(int w, int h)
{
    if (w <= 0 || h <= 0 || (size_t)w > ((size_t)1 << 30) / (size_t)h) return NULL;
    ie_image *img = (ie_image *)malloc(sizeof *img);
    if (!img) return NULL;
    img->px = (uint8_t *)calloc((size_t)w * h, 4);
    if (!img->px) {
        free(img);
        return NULL;
    }
    img->w = w;
    img->h = h;
    return img;
}

ie_image *ie_clone(const ie_image *src)
{
    ie_image *img = (ie_image *)malloc(sizeof *img);
    if (!img) return NULL;
    size_t n = (size_t)src->w * src->h * 4;
    img->px = (uint8_t *)malloc(n);
    if (!img->px) {
        free(img);
        return NULL;
    }
    memcpy(img->px, src->px, n);
    img->w = src->w;
    img->h = src->h;
    return img;
}

void ie_free(ie_image *img)
{
    if (!img) return;
    free(img->px);
    free(img);
}

void ie_swap(ie_image *a, ie_image *b)
{
    ie_image t = *a;
    *a = *b;
    *b = t;
}

typedef struct {
    const ie_image *img;
    volatile int found;
    int mode; /* 0 = alpha, 1 = gray */
} scan_ctx;

static void scan_rows(void *p, int y0, int y1)
{
    scan_ctx *c = (scan_ctx *)p;
    const int w = c->img->w;
    for (int y = y0; y < y1 && !c->found; y++) {
        const uint8_t *s = c->img->px + (size_t)y * w * 4;
        if (c->mode == 0) {
            for (int x = 0; x < w; x++)
                if (s[x * 4 + 3] != 255) {
                    c->found = 1;
                    return;
                }
        } else {
            for (int x = 0; x < w; x++)
                if (s[x * 4] != s[x * 4 + 1] || s[x * 4] != s[x * 4 + 2]) {
                    c->found = 1;
                    return;
                }
        }
    }
}

int ie_has_alpha(const ie_image *img)
{
    scan_ctx c = {img, 0, 0};
    ie_parallel_for(img->h, 64, scan_rows, &c);
    return c.found;
}

int ie_is_gray(const ie_image *img)
{
    scan_ctx c = {img, 0, 1};
    ie_parallel_for(img->h, 64, scan_rows, &c);
    return !c.found;
}

/* ---- format names ------------------------------------------------------ */

static int lower(int c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; }

static int ext_eq(const char *a, const char *b)
{
    while (*a && *b)
        if (lower(*a++) != *b++) return 0;
    return *a == 0 && *b == 0;
}

int ie_format_from_name(const char *name)
{
    if (!name) return IE_FMT_UNKNOWN;
    const char *dot = strrchr(name, '.');
    const char *e = dot ? dot + 1 : name;
    if (dot && (strchr(dot, '/') || strchr(dot, '\\'))) return IE_FMT_UNKNOWN;
    if (ext_eq(e, "png")) return IE_FMT_PNG;
    if (ext_eq(e, "jpg") || ext_eq(e, "jpeg") || ext_eq(e, "jpe") || ext_eq(e, "jfif")) return IE_FMT_JPEG;
    if (ext_eq(e, "bmp")) return IE_FMT_BMP;
    if (ext_eq(e, "tga")) return IE_FMT_TGA;
    if (ext_eq(e, "qoi")) return IE_FMT_QOI;
    if (ext_eq(e, "ppm") || ext_eq(e, "pnm")) return IE_FMT_PPM;
    if (ext_eq(e, "pgm")) return IE_FMT_PGM;
    if (ext_eq(e, "pam")) return IE_FMT_PAM;
    return IE_FMT_UNKNOWN;
}

const char *ie_format_ext(int fmt)
{
    switch (fmt) {
    case IE_FMT_PNG: return "png";
    case IE_FMT_JPEG: return "jpg";
    case IE_FMT_BMP: return "bmp";
    case IE_FMT_TGA: return "tga";
    case IE_FMT_QOI: return "qoi";
    case IE_FMT_PPM: return "ppm";
    case IE_FMT_PGM: return "pgm";
    case IE_FMT_PAM: return "pam";
    default: return NULL;
    }
}

int ie_format_has_alpha(int fmt)
{
    return fmt == IE_FMT_PNG || fmt == IE_FMT_BMP || fmt == IE_FMT_TGA || fmt == IE_FMT_QOI ||
           fmt == IE_FMT_PAM;
}

/* ---- EXIF orientation -------------------------------------------------- */

static int tiff_orientation(const uint8_t *t, size_t n)
{
    if (n < 8) return 1;
    int le;
    if (t[0] == 'I' && t[1] == 'I') le = 1;
    else if (t[0] == 'M' && t[1] == 'M') le = 0;
    else return 1;
#define RD16(o) (le ? (unsigned)(t[o] | t[(o) + 1] << 8) : (unsigned)(t[o] << 8 | t[(o) + 1]))
#define RD32(o) (le ? ((uint32_t)RD16(o) | (uint32_t)RD16((o) + 2) << 16) \
                    : ((uint32_t)RD16(o) << 16 | (uint32_t)RD16((o) + 2)))
    if (RD16(2) != 42) return 1;
    uint32_t ifd = RD32(4);
    if (ifd > n - 2) return 1;
    unsigned count = RD16(ifd);
    for (unsigned k = 0; k < count; k++) {
        size_t e = (size_t)ifd + 2 + 12 * (size_t)k;
        if (e + 12 > n) break;
        if (RD16(e) == 0x0112) {
            unsigned v = RD16(e + 8);
            return v >= 1 && v <= 8 ? (int)v : 1;
        }
    }
#undef RD16
#undef RD32
    return 1;
}

static int jpeg_orientation(const uint8_t *p, size_t n)
{
    if (n < 4 || p[0] != 0xFF || p[1] != 0xD8) return 1;
    size_t i = 2;
    while (i + 4 <= n) {
        if (p[i] != 0xFF) return 1;
        unsigned m = p[i + 1];
        if (m == 0xFF) {
            i++;
            continue;
        }
        if (m == 0x01 || (m >= 0xD0 && m <= 0xD8)) {
            i += 2;
            continue;
        }
        if (m == 0xDA || m == 0xD9) return 1;
        size_t len = (size_t)p[i + 2] << 8 | p[i + 3];
        if (len < 2 || i + 2 + len > n) return 1;
        if (m == 0xE1 && len >= 16 && memcmp(p + i + 4, "Exif\0\0", 6) == 0)
            return tiff_orientation(p + i + 10, len - 8);
        i += 2 + len;
    }
    return 1;
}

static int apply_orientation(ie_image *img, int o)
{
    switch (o) {
    case 2: return ie_flip(img, 1);
    case 3: return ie_rotate90(img, 2);
    case 4: return ie_flip(img, 0);
    case 5: return ie_transpose(img);
    case 6: return ie_rotate90(img, 1);
    case 7: {
        int r = ie_rotate90(img, 1);
        return r ? r : ie_flip(img, 0);
    }
    case 8: return ie_rotate90(img, 3);
    default: return IE_OK;
    }
}

/* ---- decoding ---------------------------------------------------------- */

static ie_image *wrap(uint8_t *px, int w, int h)
{
    ie_image *img = (ie_image *)malloc(sizeof *img);
    if (!img) return NULL;
    img->w = w;
    img->h = h;
    img->px = px;
    return img;
}

/* PAM (P7) is not handled by stb_image; support the common 8-bit tuple types. */
static int decode_pam(const uint8_t *d, size_t n, ie_image **out)
{
    int w = 0, h = 0, depth = 0, maxval = 0;
    size_t i = 3;
    while (i < n) {
        size_t e = i;
        while (e < n && d[e] != '\n') e++;
        char line[128];
        size_t l = e - i < sizeof line - 1 ? e - i : sizeof line - 1;
        memcpy(line, d + i, l);
        line[l] = 0;
        i = e + 1;
        if (strncmp(line, "ENDHDR", 6) == 0) break;
        if (sscanf(line, "WIDTH %d", &w) == 1 || sscanf(line, "HEIGHT %d", &h) == 1 ||
            sscanf(line, "DEPTH %d", &depth) == 1 || sscanf(line, "MAXVAL %d", &maxval) == 1)
            continue;
    }
    if (w <= 0 || h <= 0 || depth < 1 || depth > 4 || maxval != 255) return IE_EUNSUPPORTED;
    if ((size_t)w * h * depth > n - (i < n ? i : n)) return IE_EFORMAT;
    ie_image *img = ie_new(w, h);
    if (!img) return IE_ENOMEM;
    const uint8_t *s = d + i;
    for (size_t k = 0; k < (size_t)w * h; k++, s += depth) {
        uint8_t *p = img->px + k * 4;
        if (depth <= 2) p[0] = p[1] = p[2] = s[0];
        else p[0] = s[0], p[1] = s[1], p[2] = s[2];
        p[3] = depth == 2 ? s[1] : depth == 4 ? s[3] : 255;
    }
    *out = img;
    return IE_OK;
}

int ie_decode(const uint8_t *data, size_t len, ie_image **out)
{
    *out = NULL;
    if (!data || len < 4) return IE_EFORMAT;

    if (memcmp(data, "qoif", 4) == 0) {
        qoi_desc desc;
        void *px = qoi_decode(data, (int)(len > 0x7fffffff ? 0x7fffffff : len), &desc, 4);
        if (!px) return IE_EFORMAT;
        *out = wrap((uint8_t *)px, (int)desc.width, (int)desc.height);
        if (!*out) {
            free(px);
            return IE_ENOMEM;
        }
        return IE_OK;
    }
    if (memcmp(data, "P7\n", 3) == 0) return decode_pam(data, len, out);

    if (len > 0x7fffffff) return IE_EUNSUPPORTED;
    int w, h, comp;
    uint8_t *px = stbi_load_from_memory(data, (int)len, &w, &h, &comp, 4);
    if (!px) return IE_EFORMAT;
    ie_image *img = wrap(px, w, h);
    if (!img) {
        stbi_image_free(px);
        return IE_ENOMEM;
    }
    int r = apply_orientation(img, jpeg_orientation(data, len));
    if (r) {
        ie_free(img);
        return r;
    }
    *out = img;
    return IE_OK;
}

/* ---- encoding ---------------------------------------------------------- */

typedef struct {
    uint8_t *p;
    size_t n, cap;
    int oom;
} buf;

static void buf_put(buf *b, const void *data, size_t n)
{
    if (b->oom) return;
    if (b->n + n > b->cap) {
        size_t cap = b->cap ? b->cap : 1 << 16;
        while (cap < b->n + n) cap *= 2;
        uint8_t *p = (uint8_t *)realloc(b->p, cap);
        if (!p) {
            b->oom = 1;
            return;
        }
        b->p = p;
        b->cap = cap;
    }
    memcpy(b->p + b->n, data, n);
    b->n += n;
}

static void stb_sink(void *ctx, void *data, int size) { buf_put((buf *)ctx, data, (size_t)size); }

/* Pack RGBA into tightly packed `channels` (1 = luma, 3 = RGB), flattening onto white. */
static uint8_t *pack(const ie_image *img, int channels)
{
    size_t n = (size_t)img->w * img->h;
    uint8_t *out = (uint8_t *)malloc(n * channels);
    if (!out) return NULL;
    const uint8_t *s = img->px;
    for (size_t i = 0; i < n; i++, s += 4) {
        unsigned a = s[3], na = 255 - a;
        unsigned r = (s[0] * a + 255 * na + 127) / 255;
        unsigned g = (s[1] * a + 255 * na + 127) / 255;
        unsigned b = (s[2] * a + 255 * na + 127) / 255;
        if (channels == 1) {
            out[i] = (uint8_t)((54 * r + 183 * g + 19 * b + 128) >> 8);
        } else {
            out[i * 3] = (uint8_t)r;
            out[i * 3 + 1] = (uint8_t)g;
            out[i * 3 + 2] = (uint8_t)b;
        }
    }
    return out;
}

int ie_encode(const ie_image *img, int fmt, int quality, uint8_t **out, size_t *out_len)
{
    *out = NULL;
    *out_len = 0;
    if (!img || !img->px) return IE_EINVAL;
    if (fmt == IE_FMT_PNG) return ie_png_encode(img, quality > 0 && quality <= 9 ? quality : 6, out, out_len);

    buf b = {0};
    int alpha = ie_format_has_alpha(fmt) && ie_has_alpha(img);
    int ok = 0;
    char hdr[160];
    uint8_t *tmp = NULL;

    switch (fmt) {
    case IE_FMT_JPEG:
        if (quality <= 0) quality = 90;
        if (quality > 100) quality = 100;
        if (!(tmp = pack(img, 3))) return IE_ENOMEM;
        ok = stbi_write_jpg_to_func(stb_sink, &b, img->w, img->h, 3, tmp, quality);
        break;
    case IE_FMT_BMP:
    case IE_FMT_TGA:
        if (!alpha && !(tmp = pack(img, 3))) return IE_ENOMEM;
        ok = fmt == IE_FMT_BMP
                 ? stbi_write_bmp_to_func(stb_sink, &b, img->w, img->h, alpha ? 4 : 3, alpha ? img->px : tmp)
                 : stbi_write_tga_to_func(stb_sink, &b, img->w, img->h, alpha ? 4 : 3, alpha ? img->px : tmp);
        break;
    case IE_FMT_QOI: {
        if (!alpha && !(tmp = pack(img, 3))) return IE_ENOMEM;
        qoi_desc d = {(unsigned)img->w, (unsigned)img->h, (unsigned char)(alpha ? 4 : 3), QOI_SRGB};
        int n = 0;
        void *q = qoi_encode(alpha ? (const void *)img->px : tmp, &d, &n);
        free(tmp);
        if (!q) return IE_ENOMEM;
        *out = (uint8_t *)q;
        *out_len = (size_t)n;
        return IE_OK;
    }
    case IE_FMT_PPM:
    case IE_FMT_PGM: {
        int ch = fmt == IE_FMT_PPM ? 3 : 1;
        if (!(tmp = pack(img, ch))) return IE_ENOMEM;
        int k = snprintf(hdr, sizeof hdr, "P%d\n%d %d\n255\n", ch == 3 ? 6 : 5, img->w, img->h);
        buf_put(&b, hdr, (size_t)k);
        buf_put(&b, tmp, (size_t)img->w * img->h * ch);
        ok = 1;
        break;
    }
    case IE_FMT_PAM: {
        int k = snprintf(hdr, sizeof hdr,
                         "P7\nWIDTH %d\nHEIGHT %d\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n",
                         img->w, img->h);
        buf_put(&b, hdr, (size_t)k);
        buf_put(&b, img->px, (size_t)img->w * img->h * 4);
        ok = 1;
        break;
    }
    default:
        return IE_EUNSUPPORTED;
    }
    free(tmp);
    if (!ok || b.oom) {
        free(b.p);
        return b.oom ? IE_ENOMEM : IE_EIO;
    }
    *out = b.p;
    *out_len = b.n;
    return IE_OK;
}

void ie_buf_free(void *p) { free(p); }

/* ---- files ------------------------------------------------------------- */

#ifndef IE_NO_FILES

static FILE *open_utf8(const char *path, const char *mode)
{
#ifdef _WIN32
    wchar_t wpath[4096], wmode[8];
    if (!MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, 4096)) return NULL;
    if (!MultiByteToWideChar(CP_UTF8, 0, mode, -1, wmode, 8)) return NULL;
    return _wfopen(wpath, wmode);
#else
    return fopen(path, mode);
#endif
}

int ie_read_file(const char *path, uint8_t **data, size_t *len)
{
    *data = NULL;
    *len = 0;
    FILE *f = open_utf8(path, "rb");
    if (!f) return IE_EIO;
    buf b = {0};
    uint8_t chunk[1 << 16];
    size_t n;
    while ((n = fread(chunk, 1, sizeof chunk, f)) > 0) buf_put(&b, chunk, n);
    int err = ferror(f);
    fclose(f);
    if (err || b.oom) {
        free(b.p);
        return b.oom ? IE_ENOMEM : IE_EIO;
    }
    *data = b.p;
    *len = b.n;
    return IE_OK;
}

int ie_write_file(const char *path, const uint8_t *data, size_t len)
{
    FILE *f = open_utf8(path, "wb");
    if (!f) return IE_EIO;
    size_t n = fwrite(data, 1, len, f);
    int err = fclose(f);
    return n == len && err == 0 ? IE_OK : IE_EIO;
}

int ie_load(const char *path, ie_image **out)
{
    uint8_t *data;
    size_t len;
    int r = ie_read_file(path, &data, &len);
    if (r) return r;
    r = ie_decode(data, len, out);
    free(data);
    return r;
}

int ie_save(const ie_image *img, const char *path, int fmt, int quality)
{
    if (fmt == IE_FMT_UNKNOWN) fmt = ie_format_from_name(path);
    if (fmt == IE_FMT_UNKNOWN) return IE_EUNSUPPORTED;
    uint8_t *data;
    size_t len;
    int r = ie_encode(img, fmt, quality, &data, &len);
    if (r) return r;
    r = ie_write_file(path, data, len);
    free(data);
    return r;
}

#endif
