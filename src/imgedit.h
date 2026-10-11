/*
 * imgedit - a small, fast image editing core in portable C.
 *
 * Every operation works on an 8-bit RGBA image with straight (non-premultiplied)
 * alpha and edits it in place, reallocating the pixel buffer when the size
 * changes. Functions return IE_OK (0) or a negative error code.
 *
 * MIT License - see LICENSE.
 */
#ifndef IMGEDIT_H
#define IMGEDIT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IE_VERSION "1.0.0"

typedef struct ie_image {
    int w, h;
    uint8_t *px; /* w * h * 4 bytes, rows tightly packed */
} ie_image;

enum {
    IE_OK = 0,
    IE_ENOMEM = -1,
    IE_EINVAL = -2,
    IE_EIO = -3,
    IE_EFORMAT = -4,
    IE_EUNSUPPORTED = -5
};

const char *ie_strerror(int err);

/* ---- lifetime ---------------------------------------------------------- */

ie_image *ie_new(int w, int h); /* transparent black, NULL on failure */
ie_image *ie_clone(const ie_image *src);
void ie_free(ie_image *img);
void ie_swap(ie_image *a, ie_image *b);
int ie_has_alpha(const ie_image *img); /* any pixel with alpha < 255 */
int ie_is_gray(const ie_image *img);   /* every pixel has r == g == b */

/* Worker threads used by all operations (0 = one per logical CPU). */
void ie_set_threads(int n);
int ie_get_threads(void);

/* ---- files and formats ------------------------------------------------- */

enum {
    IE_FMT_UNKNOWN = 0,
    IE_FMT_PNG,
    IE_FMT_JPEG,
    IE_FMT_BMP,
    IE_FMT_TGA,
    IE_FMT_QOI,
    IE_FMT_PPM, /* binary P6, alpha flattened */
    IE_FMT_PGM, /* binary P5, luma */
    IE_FMT_PAM  /* P7 RGB_ALPHA */
};

int ie_format_from_name(const char *name); /* "png", ".jpg" or "a/b.jpeg" */
const char *ie_format_ext(int fmt);
int ie_format_has_alpha(int fmt);

/* Decode PNG, JPEG, BMP, GIF (first frame), TGA, PSD, HDR, PNM, PAM or QOI.
 * JPEG EXIF orientation is applied. */
int ie_decode(const uint8_t *data, size_t len, ie_image **out);

/* Encode to a malloc'd buffer (release with ie_buf_free).
 * quality: JPEG 1..100 (0 = 90). For PNG it is the deflate level 1..9
 * (0 = default). Formats without alpha are flattened onto white. */
int ie_encode(const ie_image *img, int fmt, int quality, uint8_t **out, size_t *out_len);
void ie_buf_free(void *p);

/* Parallel PNG encoder used by ie_encode (level 0..9, 0 = stored). */
int ie_png_encode(const ie_image *img, int level, uint8_t **out, size_t *out_len);

#ifndef IE_NO_FILES
int ie_load(const char *path, ie_image **out);              /* UTF-8 path */
int ie_save(const ie_image *img, const char *path, int fmt, int quality);
int ie_read_file(const char *path, uint8_t **data, size_t *len);
int ie_write_file(const char *path, const uint8_t *data, size_t len);
#endif

/* ---- color ------------------------------------------------------------- */

enum {
    IE_INVERT_RGB = 1,       /* photographic negative */
    IE_INVERT_ALPHA = 2,     /* swap transparent and opaque */
    IE_INVERT_LIGHTNESS = 4  /* flip light/dark but keep hue and saturation */
};
int ie_invert(ie_image *img, int mode);

int ie_grayscale(ie_image *img);
int ie_sepia(ie_image *img, float amount); /* 0..1 */

typedef struct ie_adjust {
    float exposure;    /* stops, 0 = unchanged */
    float brightness;  /* -1..1 */
    float contrast;    /* -1..1 */
    float gamma;       /* > 0, 1 = unchanged */
    float saturation;  /* -1..1, -1 = grayscale */
    float hue;         /* degrees */
    float temperature; /* -1..1, positive = warmer */
    float tint;        /* -1..1, positive = magenta */
} ie_adjust;

void ie_adjust_init(ie_adjust *a);
int ie_adjust_is_identity(const ie_adjust *a);
int ie_apply_adjust(ie_image *img, const ie_adjust *a);

/* Stretch levels so the darkest/brightest `clip` fraction of pixels maps to
 * black/white. per_channel also neutralises color casts. */
int ie_auto_levels(ie_image *img, float clip, int per_channel);
int ie_threshold(ie_image *img, int level);  /* black/white at luma level */
int ie_posterize(ie_image *img, int levels); /* 2..255 levels per channel */
int ie_flatten(ie_image *img, uint32_t rgb); /* composite onto 0xRRGGBB */
int ie_alpha_to_gray(ie_image *img);         /* show the alpha channel as an opaque mask */

/* ---- geometry ---------------------------------------------------------- */

int ie_resize(ie_image *img, int w, int h); /* gamma-correct, alpha-aware */
int ie_crop(ie_image *img, int x, int y, int w, int h);
int ie_rotate90(ie_image *img, int turns);  /* clockwise quarter turns */
int ie_rotate(ie_image *img, float degrees, uint32_t fill_rgba); /* expands canvas */
int ie_flip(ie_image *img, int horizontal);
int ie_transpose(ie_image *img);
/* Crop away borders that are transparent (or, for opaque images, the same
 * color as the corners) within tolerance 0..255. */
int ie_trim(ie_image *img, int tolerance);
int ie_pad(ie_image *img, int top, int right, int bottom, int left, uint32_t fill_rgba);

/* ---- filters ----------------------------------------------------------- */

int ie_blur(ie_image *img, float sigma);
int ie_sharpen(ie_image *img, float amount, float sigma, int threshold);

/* ---- background removal ------------------------------------------------ */

/* Model input: img stretched to tw x th, composited onto white where
 * transparent, written as planar float32 [3][th][tw] = (v*scale - mean)/std. */
int ie_nn_input(const ie_image *img, int tw, int th, const float mean[3],
                const float std[3], float scale, float *out);

/* Raw model output -> 0..1 matte, in place. Logits get a sigmoid; minmax
 * stretches the result to the full range (what most matting models expect). */
void ie_nn_output(float *v, size_t n, int minmax);

/* Use a low-resolution matte (mw x mh, values 0..1) as img's alpha. The matte
 * is upsampled with a guided filter so the edge follows the full-resolution
 * photo. radius is in matte pixels and grows automatically when the matte is
 * stretched over a much larger photo; radius <= 0 means plain bilinear.
 * Existing alpha is multiplied in. */
int ie_matte_apply(ie_image *img, const float *matte, int mw, int mh, int radius, float eps);

/* Alpha-channel edits. */
int ie_alpha_levels(ie_image *img, int lo, int hi); /* <=lo -> 0, >=hi -> 255 */
int ie_alpha_grow(ie_image *img, int radius);       /* < 0 shrinks */
int ie_alpha_feather(ie_image *img, float sigma);
int ie_alpha_clean(ie_image *img, float min_area);  /* drop islands < min_area * largest */

/* Re-estimate the colors of semi-transparent edge pixels so the old
 * background does not bleed into the cut-out (removes halos). */
int ie_decontaminate(ie_image *img);

/* Remove a plain background without a model. The key color is taken from the
 * image border when auto_color is set. Colors within `tolerance` (CIE76 delta E)
 * become transparent with a soft ramp `softness` wide. With `connected`, only
 * background reachable from the border is removed. */
int ie_key_background(ie_image *img, int auto_color, uint32_t rgb, float tolerance,
                      float softness, int connected);

enum { IE_BG_TRANSPARENT = 0, IE_BG_COLOR, IE_BG_BLUR, IE_BG_IMAGE };

typedef struct ie_bg_options {
    int mode;              /* IE_BG_* */
    uint32_t color;        /* IE_BG_COLOR: 0xRRGGBB */
    float blur;            /* IE_BG_BLUR: sigma in pixels */
    const ie_image *image; /* IE_BG_IMAGE: scaled to cover */
    int decontaminate;     /* fix edge colors first (recommended) */
} ie_bg_options;

/* Finish a cut-out whose alpha is the subject mask: clean edge colors and put
 * the subject on the requested background. */
int ie_finish_cutout(ie_image *img, const ie_bg_options *opt);

#ifdef __cplusplus
}
#endif

#endif
