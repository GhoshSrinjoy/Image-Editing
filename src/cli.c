/* imgedit command-line tool. Operations run left to right on every input. */
#include "imgedit.h"
#include "nn.h"
#include "parallel.h"
#include "platform.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- options ----------------------------------------------------------- */

typedef enum {
    OP_REMOVE_BG,
    OP_KEY_BG,
    OP_INVERT,
    OP_GRAYSCALE,
    OP_SEPIA,
    OP_ADJUST,
    OP_AUTO_LEVELS,
    OP_THRESHOLD,
    OP_POSTERIZE,
    OP_RESIZE,
    OP_CROP,
    OP_ROTATE,
    OP_FLIP,
    OP_TRIM,
    OP_PAD,
    OP_BLUR,
    OP_SHARPEN,
    OP_FLATTEN
} op_kind;

static const char *OP_NAMES[] = {"remove-bg", "key",  "invert", "grayscale", "sepia", "adjust",
                                 "auto-levels", "threshold", "posterize", "resize", "crop", "rotate",
                                 "flip", "trim", "pad", "blur", "sharpen", "flatten"};

typedef struct {
    op_kind kind;
    int i[4];
    float f[2];
    uint32_t color;
    char mode; /* resize: 0 fit, '!' exact, '^' cover, '>' shrink only, '%' percent */
    ie_adjust adj;
} op;

typedef struct {
    char **inputs;
    int ninputs, cap_inputs;
    const char *output;
    int format, quality, threads, verbose, overwrite;
    /* background removal */
    const char *model, *model_dir, *device;
    int refine;
    float eps;
    int hard, grow, clean, keep_bg, mask_only, fix_edges;
    float feather;
    int bg_mode;
    uint32_t bg_color;
    float bg_blur;
    const char *bg_file;
    ie_image *bg_image;
    float key_tol, key_soft;
    int key_connected;
    op *ops;
    int nops;
    /* runtime */
    ie_model *nn;
    const ie_model_info *nn_info;
    int uses_model;
} cli;

static void die(const char *fmt, const char *arg)
{
    fprintf(stderr, "imgedit: ");
    fprintf(stderr, fmt, arg);
    fprintf(stderr, "\n");
    exit(2);
}

/* ---- small parsers ----------------------------------------------------- */

static int parse_num(const char *s, float *out)
{
    if (!s || !*s) return 0;
    char *end;
    double v = strtod(s, &end);
    if (*end || v != v) return 0;
    *out = (float)v;
    return 1;
}

static int parse_int(const char *s, int *out)
{
    float f;
    if (!parse_num(s, &f) || f != floorf(f) || fabsf(f) > 1e9f) return 0;
    *out = (int)f;
    return 1;
}

static int hexval(int c) { return c >= '0' && c <= '9' ? c - '0' : (c | 32) >= 'a' && (c | 32) <= 'f' ? (c | 32) - 'a' + 10 : -1; }

static int parse_color(const char *s, uint32_t *rgb)
{
    static const struct {
        const char *name;
        uint32_t v;
    } named[] = {{"white", 0xFFFFFF}, {"black", 0x000000}, {"red", 0xFF0000},    {"green", 0x00A000},
                 {"lime", 0x00FF00},  {"blue", 0x0000FF},  {"gray", 0x808080},   {"grey", 0x808080},
                 {"yellow", 0xFFFF00}, {"cyan", 0x00FFFF}, {"magenta", 0xFF00FF}, {"orange", 0xFFA500},
                 {"pink", 0xFFC0CB},  {"purple", 0x800080}, {"navy", 0x000080},  {"beige", 0xF5F5DC}};
    if (!s || !*s) return 0;
    for (size_t i = 0; i < sizeof named / sizeof named[0]; i++) {
        const char *a = s, *b = named[i].name;
        while (*a && (*a | 32) == *b) a++, b++;
        if (!*a && !*b) {
            *rgb = named[i].v;
            return 1;
        }
    }
    int r, g, b;
    char tail;
    if (sscanf(s, "%d,%d,%d%c", &r, &g, &b, &tail) == 3 && r >= 0 && r < 256 && g >= 0 && g < 256 && b >= 0 && b < 256) {
        *rgb = (uint32_t)(r << 16 | g << 8 | b);
        return 1;
    }
    const char *h = s[0] == '#' ? s + 1 : s;
    size_t n = strlen(h);
    if (n != 3 && n != 6) return 0;
    uint32_t v = 0;
    for (size_t i = 0; i < n; i++) {
        int d = hexval(h[i]);
        if (d < 0) return 0;
        v = n == 3 ? v << 8 | (uint32_t)(d * 17) : v << 4 | (uint32_t)d;
    }
    *rgb = v;
    return 1;
}

/* "800x600", "800x600!", "800x600^", "800x600>", "50%", "800", "x600". */
static int parse_resize(const char *s, op *o)
{
    float a = 0, b = 0;
    size_t n = strlen(s);
    if (!n) return 0;
    char buf[64];
    if (n >= sizeof buf) return 0;
    memcpy(buf, s, n + 1);
    o->mode = 0;
    if (buf[n - 1] == '%') {
        buf[n - 1] = 0;
        if (!parse_num(buf, &a) || a <= 0) return 0;
        o->mode = '%';
        o->f[0] = a / 100.0f;
        return 1;
    }
    if (buf[n - 1] == '!' || buf[n - 1] == '^' || buf[n - 1] == '>') o->mode = buf[n - 1], buf[--n] = 0;
    char *x = strchr(buf, 'x');
    if (!x) x = strchr(buf, 'X');
    if (x) {
        *x = 0;
        if (*buf && (!parse_num(buf, &a) || a <= 0)) return 0;
        if (x[1] && (!parse_num(x + 1, &b) || b <= 0)) return 0;
    } else if (!parse_num(buf, &a) || a <= 0) {
        return 0;
    }
    if (a == 0 && b == 0) return 0;
    o->i[0] = (int)lrintf(a);
    o->i[1] = (int)lrintf(b);
    return 1;
}

/* "WxH+X+Y" or "WxH" (centered). */
static int parse_crop(const char *s, op *o)
{
    int w, h, x = 0, y = 0;
    char c;
    if (sscanf(s, "%dx%d+%d+%d%c", &w, &h, &x, &y, &c) == 4 || sscanf(s, "%dx%d%c", &w, &h, &c) == 2) {
        if (w <= 0 || h <= 0) return 0;
        o->i[0] = w, o->i[1] = h, o->i[2] = x, o->i[3] = y;
        o->mode = strchr(s, '+') ? 0 : 'c';
        return 1;
    }
    return 0;
}

/* ---- argument handling ------------------------------------------------- */

static op *add_op(cli *c, op_kind k)
{
    c->ops = (op *)realloc(c->ops, sizeof(op) * (size_t)(c->nops + 1));
    if (!c->ops) die("%s", "out of memory");
    op *o = &c->ops[c->nops++];
    memset(o, 0, sizeof *o);
    o->kind = k;
    ie_adjust_init(&o->adj);
    return o;
}

/* Consecutive adjustment flags share one pass. */
static ie_adjust *adjust_op(cli *c)
{
    if (c->nops && c->ops[c->nops - 1].kind == OP_ADJUST) return &c->ops[c->nops - 1].adj;
    return &add_op(c, OP_ADJUST)->adj;
}

static void add_input(cli *c, const char *s)
{
    if (c->ninputs == c->cap_inputs) {
        c->cap_inputs = c->cap_inputs ? c->cap_inputs * 2 : 16;
        c->inputs = (char **)realloc(c->inputs, sizeof(char *) * (size_t)c->cap_inputs);
        if (!c->inputs) die("%s", "out of memory");
    }
    size_t n = strlen(s) + 1;
    c->inputs[c->ninputs] = (char *)malloc(n);
    if (!c->inputs[c->ninputs]) die("%s", "out of memory");
    memcpy(c->inputs[c->ninputs++], s, n);
}

static void usage(FILE *f)
{
    fputs(
        "imgedit " IE_VERSION " - fast image editing: background removal, inversion, adjustments, conversion\n"
        "\n"
        "Usage:\n"
        "  imgedit [options] <input>... [operations] [-o <output>]\n"
        "  imgedit models              list background removal models\n"
        "  imgedit info <input>...     show image details\n"
        "\n"
        "Inputs may be files or folders. Operations run in the order given.\n"
        "\n"
        "Background\n"
        "  -r, --remove-bg         cut out the subject with a neural model\n"
        "  -m, --model NAME        fast, balanced, best, max, portrait, anime, ... or a .onnx file\n"
        "                          (default: best with a GPU, balanced on CPU)\n"
        "      --key [COLOR]       remove a plain background by color (default: detected from the border)\n"
        "      --tolerance N       color distance for --key (default 10)\n"
        "      --softness N        edge softness for --key (default 8)\n"
        "      --everywhere        with --key, also remove matching colors not connected to the border\n"
        "      --refine N          edge refinement radius, 0 = off (default 4, grows for big photos)\n"
        "      --hard [N]          hard mask edge, cut at alpha N (default 128)\n"
        "      --shrink N          shrink the mask by N pixels (--grow N to grow)\n"
        "      --feather SIGMA     soften the mask edge\n"
        "      --clean             remove small stray blobs from the mask\n"
        "      --keep-bg           invert the mask: remove the subject, keep the background\n"
        "      --mask              output the mask (white = subject) instead of the cut-out\n"
        "      --bg VALUE          new background: a color (white, #ff8800, 0,128,255), blur[=SIGMA] or an image file\n"
        "      --no-fix-edges      skip edge color cleanup\n"
        "      --device NAME       auto (default), cpu, gpu, cuda, dml or coreml\n"
        "      --model-dir DIR     where models are stored (default: ~/.u2net, shared with rembg)\n"
        "\n"
        "Color\n"
        "      --invert [MODE]     rgb (default), lightness (keeps hues) or alpha\n"
        "      --grayscale         --sepia [AMOUNT]\n"
        "      --brightness V      --contrast V      (-1..1)\n"
        "      --exposure EV       --gamma G\n"
        "      --saturation V      --hue DEGREES     --temperature V   --tint V\n"
        "      --auto-levels       --auto-color      stretch contrast (auto-color also fixes casts)\n"
        "      --threshold N       black and white at level N (0..255)\n"
        "      --posterize N       N levels per channel\n"
        "\n"
        "Transform\n"
        "      --resize SPEC       800x600 (fit inside), 800x600! (exact), 800x600^ (cover),\n"
        "                          800x600> (only shrink), 50%, 800 (width), x600 (height)\n"
        "      --crop SPEC         WxH+X+Y, or WxH centered\n"
        "      --rotate DEGREES    clockwise; non-multiples of 90 expand the canvas\n"
        "      --flip h|v          mirror horizontally or vertically\n"
        "      --trim [FUZZ]       crop transparent or uniform borders\n"
        "      --pad N[:COLOR]     add a border (transparent by default)\n"
        "\n"
        "Filters\n"
        "      --blur SIGMA        --sharpen [AMOUNT]\n"
        "\n"
        "Output\n"
        "  -o, --output PATH       output file or folder (default: <name>-out.<ext> next to the input)\n"
        "  -f, --format FMT        png, jpg, bmp, tga, qoi, ppm, pgm or pam\n"
        "  -q, --quality N         JPEG quality 1-100 (default 90); PNG compression 1-9 (default 6)\n"
        "      --flatten [COLOR]   remove transparency (default white)\n"
        "  -j, --threads N         worker threads (default: all cores)\n"
        "  -y, --overwrite         allow writing over input files\n"
        "  -v, --verbose           show timings\n"
        "  -h, --help              --version\n"
        "\n"
        "Examples\n"
        "  imgedit photo.jpg -r -o cutout.png\n"
        "  imgedit photo.jpg -r --bg white -o product.jpg\n"
        "  imgedit shots/ -r -m best --trim -o cutouts/\n"
        "  imgedit scan.png --invert lightness --auto-levels -o scan-dark.png\n"
        "  imgedit *.png -f jpg -q 85 -o jpg/\n",
        f);
}

static int is_option(const char *s) { return s[0] == '-' && s[1] && !(s[1] >= '0' && s[1] <= '9') && s[1] != '.'; }

/* Value for an option that requires one: "--opt=value" or the next argument. */
static const char *need(int argc, char **argv, int *i, const char *inline_val, const char *name)
{
    if (inline_val) return inline_val;
    if (*i + 1 >= argc) die("%s needs a value", name);
    return argv[++*i];
}

/* Optional value: taken from "--opt=value", or the next argument when `accept` likes it. */
static const char *maybe(int argc, char **argv, int *i, const char *inline_val, int (*accept)(const char *))
{
    if (inline_val) return inline_val;
    if (*i + 1 < argc && accept(argv[*i + 1]) && !pf_exists(argv[*i + 1])) return argv[++*i];
    return NULL;
}

static int accept_num(const char *s)
{
    float f;
    return parse_num(s, &f);
}
static int accept_color(const char *s)
{
    uint32_t c;
    return parse_color(s, &c) && !accept_num(s);
}
static int accept_invert(const char *s)
{
    return !strcmp(s, "rgb") || !strcmp(s, "lightness") || !strcmp(s, "luma") || !strcmp(s, "alpha") ||
           !strcmp(s, "mask");
}

static float need_num(int argc, char **argv, int *i, const char *v, const char *name)
{
    float f;
    const char *s = need(argc, argv, i, v, name);
    if (!parse_num(s, &f)) die("%s expects a number", name);
    return f;
}

static void parse_args(cli *c, int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        char *a = argv[i];
        if (!is_option(a)) {
            add_input(c, a);
            continue;
        }
        if (!strcmp(a, "--")) {
            while (++i < argc) add_input(c, argv[i]);
            break;
        }
        /* Split "--name=value". */
        char name[64];
        const char *val = NULL;
        const char *eq = strchr(a, '=');
        size_t len = eq ? (size_t)(eq - a) : strlen(a);
        if (len >= sizeof name) die("unknown option %s", a);
        memcpy(name, a, len);
        name[len] = 0;
        if (eq) val = eq + 1;
#define IS(s) (!strcmp(name, s))
        if (IS("-h") || IS("--help")) {
            usage(stdout);
            exit(0);
        } else if (IS("--version")) {
            printf("imgedit %s\n", IE_VERSION);
            exit(0);
        } else if (IS("-o") || IS("--output")) {
            c->output = need(argc, argv, &i, val, name);
        } else if (IS("-f") || IS("--format")) {
            const char *s = need(argc, argv, &i, val, name);
            c->format = ie_format_from_name(s);
            if (!c->format) die("unknown output format '%s'", s);
        } else if (IS("-q") || IS("--quality")) {
            c->quality = (int)need_num(argc, argv, &i, val, name);
        } else if (IS("-j") || IS("--threads")) {
            c->threads = (int)need_num(argc, argv, &i, val, name);
        } else if (IS("-y") || IS("--overwrite")) {
            c->overwrite = 1;
        } else if (IS("-v") || IS("--verbose")) {
            c->verbose = 1;
        } else if (IS("-r") || IS("--remove-bg")) {
            add_op(c, OP_REMOVE_BG);
            c->uses_model = 1;
        } else if (IS("-m") || IS("--model")) {
            c->model = need(argc, argv, &i, val, name);
        } else if (IS("--model-dir")) {
            c->model_dir = need(argc, argv, &i, val, name);
        } else if (IS("--device")) {
            c->device = need(argc, argv, &i, val, name);
        } else if (IS("--key")) {
            op *o = add_op(c, OP_KEY_BG);
            const char *s = maybe(argc, argv, &i, val, accept_color);
            o->i[0] = !s;
            if (s && !parse_color(s, &o->color)) die("bad color '%s'", s);
        } else if (IS("--tolerance")) {
            c->key_tol = need_num(argc, argv, &i, val, name);
        } else if (IS("--softness")) {
            c->key_soft = need_num(argc, argv, &i, val, name);
        } else if (IS("--everywhere")) {
            c->key_connected = 0;
        } else if (IS("--refine")) {
            c->refine = (int)need_num(argc, argv, &i, val, name);
        } else if (IS("--eps")) {
            c->eps = need_num(argc, argv, &i, val, name);
        } else if (IS("--hard")) {
            const char *s = maybe(argc, argv, &i, val, accept_num);
            c->hard = s ? (int)atof(s) : 128;
        } else if (IS("--shrink")) {
            c->grow = -(int)need_num(argc, argv, &i, val, name);
        } else if (IS("--grow")) {
            c->grow = (int)need_num(argc, argv, &i, val, name);
        } else if (IS("--feather")) {
            c->feather = need_num(argc, argv, &i, val, name);
        } else if (IS("--clean")) {
            c->clean = 1;
        } else if (IS("--keep-bg")) {
            c->keep_bg = 1;
        } else if (IS("--mask")) {
            c->mask_only = 1;
        } else if (IS("--no-fix-edges")) {
            c->fix_edges = 0;
        } else if (IS("--bg")) {
            const char *s = need(argc, argv, &i, val, name);
            if (!strncmp(s, "blur", 4) && (s[4] == 0 || s[4] == '=' || s[4] == ':')) {
                c->bg_mode = IE_BG_BLUR;
                c->bg_blur = s[4] ? (float)atof(s + 5) : 0;
            } else if (!strcmp(s, "transparent") || !strcmp(s, "none")) {
                c->bg_mode = IE_BG_TRANSPARENT;
            } else if (parse_color(s, &c->bg_color) && !pf_exists(s)) {
                c->bg_mode = IE_BG_COLOR;
            } else if (pf_exists(s)) {
                c->bg_mode = IE_BG_IMAGE;
                c->bg_file = s;
            } else {
                die("--bg: '%s' is not a color, blur or an existing image", s);
            }
        } else if (IS("--invert")) {
            op *o = add_op(c, OP_INVERT);
            const char *s = maybe(argc, argv, &i, val, accept_invert);
            o->i[0] = !s || !strcmp(s, "rgb")                          ? IE_INVERT_RGB
                      : !strcmp(s, "lightness") || !strcmp(s, "luma") ? IE_INVERT_LIGHTNESS
                      : !strcmp(s, "alpha") || !strcmp(s, "mask")     ? IE_INVERT_ALPHA
                                                                      : 0;
            if (!o->i[0]) die("--invert: unknown mode '%s' (rgb, lightness, alpha)", s);
        } else if (IS("--grayscale") || IS("--greyscale") || IS("--gray")) {
            add_op(c, OP_GRAYSCALE);
        } else if (IS("--sepia")) {
            const char *s = maybe(argc, argv, &i, val, accept_num);
            add_op(c, OP_SEPIA)->f[0] = s ? (float)atof(s) : 1.0f;
        } else if (IS("--brightness")) {
            adjust_op(c)->brightness = need_num(argc, argv, &i, val, name);
        } else if (IS("--contrast")) {
            adjust_op(c)->contrast = need_num(argc, argv, &i, val, name);
        } else if (IS("--exposure")) {
            adjust_op(c)->exposure = need_num(argc, argv, &i, val, name);
        } else if (IS("--gamma")) {
            adjust_op(c)->gamma = need_num(argc, argv, &i, val, name);
        } else if (IS("--saturation")) {
            adjust_op(c)->saturation = need_num(argc, argv, &i, val, name);
        } else if (IS("--hue")) {
            adjust_op(c)->hue = need_num(argc, argv, &i, val, name);
        } else if (IS("--temperature")) {
            adjust_op(c)->temperature = need_num(argc, argv, &i, val, name);
        } else if (IS("--tint")) {
            adjust_op(c)->tint = need_num(argc, argv, &i, val, name);
        } else if (IS("--auto-levels") || IS("--auto-color")) {
            add_op(c, OP_AUTO_LEVELS)->i[0] = IS("--auto-color");
        } else if (IS("--threshold")) {
            add_op(c, OP_THRESHOLD)->i[0] = (int)need_num(argc, argv, &i, val, name);
        } else if (IS("--posterize")) {
            add_op(c, OP_POSTERIZE)->i[0] = (int)need_num(argc, argv, &i, val, name);
        } else if (IS("--resize")) {
            const char *s = need(argc, argv, &i, val, name);
            if (!parse_resize(s, add_op(c, OP_RESIZE))) die("--resize: cannot read '%s'", s);
        } else if (IS("--crop")) {
            const char *s = need(argc, argv, &i, val, name);
            if (!parse_crop(s, add_op(c, OP_CROP))) die("--crop: expected WxH+X+Y or WxH, got '%s'", s);
        } else if (IS("--rotate")) {
            add_op(c, OP_ROTATE)->f[0] = need_num(argc, argv, &i, val, name);
        } else if (IS("--flip")) {
            const char *s = need(argc, argv, &i, val, name);
            if (strcmp(s, "h") && strcmp(s, "v") && strcmp(s, "horizontal") && strcmp(s, "vertical"))
                die("--flip expects h or v, got '%s'", s);
            add_op(c, OP_FLIP)->i[0] = s[0] == 'h';
        } else if (IS("--flop")) {
            add_op(c, OP_FLIP)->i[0] = 1;
        } else if (IS("--trim")) {
            const char *s = maybe(argc, argv, &i, val, accept_num);
            add_op(c, OP_TRIM)->i[0] = s ? atoi(s) : 8;
        } else if (IS("--pad")) {
            const char *s = need(argc, argv, &i, val, name);
            op *o = add_op(c, OP_PAD);
            char buf[64];
            snprintf(buf, sizeof buf, "%s", s);
            char *colon = strchr(buf, ':');
            o->color = 0; /* transparent */
            if (colon) {
                uint32_t rgb;
                *colon = 0;
                if (!parse_color(colon + 1, &rgb)) die("--pad: bad color in '%s'", s);
                o->color = rgb << 8 | 0xFF;
            }
            if (!parse_int(buf, &o->i[0]) || o->i[0] < 0) die("--pad: bad size in '%s'", s);
        } else if (IS("--blur")) {
            add_op(c, OP_BLUR)->f[0] = need_num(argc, argv, &i, val, name);
        } else if (IS("--sharpen")) {
            const char *s = maybe(argc, argv, &i, val, accept_num);
            add_op(c, OP_SHARPEN)->f[0] = s ? (float)atof(s) : 0.8f;
        } else if (IS("--flatten")) {
            const char *s = maybe(argc, argv, &i, val, accept_color);
            op *o = add_op(c, OP_FLATTEN);
            o->color = 0xFFFFFF;
            if (s && !parse_color(s, &o->color)) die("bad color '%s'", s);
        } else {
            die("unknown option '%s' (see --help)", a);
        }
#undef IS
    }
}

/* ---- models ------------------------------------------------------------ */

#define PATHLEN 4096

/* snprintf succeeded without truncation; over-long paths are skipped, not cut. */
static int fits(int written, size_t size) { return written >= 0 && (size_t)written < size; }

static int model_dirs(const cli *c, char dirs[][PATHLEN], int max)
{
    int n = 0;
    char buf[PATHLEN];
    const char *e;
#define ADD(...)                                                                  \
    do {                                                                          \
        if (n < max && fits(snprintf(dirs[n], PATHLEN, __VA_ARGS__), PATHLEN)) n++; \
    } while (0)
    if (c->model_dir) ADD("%s", c->model_dir);
    if ((e = pf_getenv("IMGEDIT_MODELS"))) ADD("%s", e);
    if (pf_exe_dir(buf, sizeof buf) == 0) ADD("%s%cmodels", buf, PF_SEP);
    if ((e = pf_getenv("U2NET_HOME"))) ADD("%s", e);
    if (pf_home_dir(buf, sizeof buf) == 0) ADD("%s%c.u2net", buf, PF_SEP);
#undef ADD
    return n;
}

static int find_model_file(const cli *c, const ie_model_info *m, char *path, size_t n)
{
    static char dirs[6][PATHLEN];
    int nd = model_dirs(c, dirs, 6);
    for (int i = 0; i < nd; i++)
        if (fits(snprintf(path, n, "%s%c%s", dirs[i], PF_SEP, m->file), n) && pf_exists(path)) return 1;
    return 0;
}

static int download_model(const cli *c, const ie_model_info *m, char *path, size_t n)
{
    char dir[PATHLEN], home[PATHLEN], tmp[PATHLEN + 8];
    const char *e;
    int ok;
    if (c->model_dir) ok = fits(snprintf(dir, sizeof dir, "%s", c->model_dir), sizeof dir);
    else if ((e = pf_getenv("IMGEDIT_MODELS"))) ok = fits(snprintf(dir, sizeof dir, "%s", e), sizeof dir);
    else if ((e = pf_getenv("U2NET_HOME"))) ok = fits(snprintf(dir, sizeof dir, "%s", e), sizeof dir);
    else if (pf_home_dir(home, sizeof home) == 0) ok = fits(snprintf(dir, sizeof dir, "%s%c.u2net", home, PF_SEP), sizeof dir);
    else return 0;
    if (!ok || !fits(snprintf(path, n, "%s%c%s", dir, PF_SEP, m->file), n) ||
        !fits(snprintf(tmp, sizeof tmp, "%s.part", path), sizeof tmp)) {
        fprintf(stderr, "imgedit: model folder path is too long\n");
        return 0;
    }
    if (pf_mkdirs(dir) != 0) {
        fprintf(stderr, "imgedit: cannot create %s\n", dir);
        return 0;
    }
    fprintf(stderr, "Downloading model '%s' (%d MB) to %s\n", m->name, m->mb, dir);
    const char *argv[] = {"curl", "-fL", "--retry", "2", "-#", "-o", tmp, m->url, NULL};
    int code = pf_run(argv);
    if (code != 0) {
        pf_remove(tmp);
        if (code < 0) fprintf(stderr, "imgedit: curl is not available. Download %s\n  into %s\n", m->url, dir);
        else fprintf(stderr, "imgedit: download failed (curl exit %d): %s\n", code, m->url);
        return 0;
    }
    if (pf_rename(tmp, path) != 0) {
        fprintf(stderr, "imgedit: cannot move %s into place\n", tmp);
        return 0;
    }
    return 1;
}

static int ensure_model(cli *c)
{
    if (c->nn) return 1;
    char path[PATHLEN + 64], err[1024] = "";
    static ie_model_info custom;
    const char *name = c->model ? c->model : "balanced";
    size_t len = strlen(name);
    if (len > 5 && pf_exists(name) && !pf_is_dir(name)) {
        if (!fits(snprintf(path, sizeof path, "%s", name), sizeof path)) die("model path too long: %s", name);
        const ie_model_info *known = ie_model_for_file(name);
        if (known) {
            custom = *known;
        } else { /* unknown model: assume ImageNet normalisation and size from the model */
            memset(&custom, 0, sizeof custom);
            custom.name = "custom";
            custom.file = name;
            custom.mean[0] = 0.485f, custom.mean[1] = 0.456f, custom.mean[2] = 0.406f;
            custom.std[0] = 0.229f, custom.std[1] = 0.224f, custom.std[2] = 0.225f;
            custom.scale = 1 / 255.0f;
            custom.size = 1024;
            custom.minmax = 1;
        }
        c->nn_info = &custom;
    } else {
        c->nn_info = ie_model_find(name);
        if (!c->nn_info) {
            fprintf(stderr, "imgedit: unknown model '%s' (run 'imgedit models')\n", name);
            return 0;
        }
        if (!find_model_file(c, c->nn_info, path, sizeof path) && !download_model(c, c->nn_info, path, sizeof path))
            return 0;
    }
    double t = pf_now();
    if (ie_model_open(&c->nn, path, c->nn_info, c->device, c->threads, err, sizeof err) != IE_OK) {
        fprintf(stderr, "imgedit: %s\n", err);
        return 0;
    }
    if (c->verbose) {
        char desc[1200] = "";
        ie_runtime_load(desc, sizeof desc, NULL, 0);
        fprintf(stderr, "%s\nmodel %s on %s, loaded in %.0f ms\n", desc, c->nn_info->name, ie_model_device(c->nn),
                (pf_now() - t) * 1000);
    }
    return 1;
}

static void list_models(const cli *c)
{
    int n;
    const ie_model_info *m = ie_models(&n);
    printf("%-18s %-9s %7s  %-10s %s\n", "MODEL", "ALIAS", "SIZE", "STATUS", "NOTES");
    for (int i = 0; i < n; i++) {
        char path[PATHLEN + 64];
        int have = find_model_file(c, &m[i], path, sizeof path);
        printf("%-18s %-9s %4d MB  %-10s %s\n", m[i].name, m[i].alias ? m[i].alias : "", m[i].mb,
               have ? "ready" : "download", m[i].about);
    }
    char desc[1200], err[1200];
    if (ie_runtime_load(desc, sizeof desc, err, sizeof err) == IE_OK) printf("\nRuntime: %s\n", desc);
    else printf("\nRuntime: %s\n", err);
    const char *gpu = ie_runtime_gpu();
    printf("Models download on first use. Default here: %s.\n", gpu ? "best (GPU)" : "balanced (CPU)");
}

/* ---- per-file processing ---------------------------------------------- */

static void finish_mask(cli *c, ie_image *img)
{
    if (c->clean) ie_alpha_clean(img, 0.01f);
    if (c->hard >= 0) ie_alpha_levels(img, c->hard - 1, c->hard);
    if (c->grow) ie_alpha_grow(img, c->grow);
    if (c->feather > 0) ie_alpha_feather(img, c->feather);
    if (c->keep_bg) ie_invert(img, IE_INVERT_ALPHA);
    if (c->mask_only) {
        ie_alpha_to_gray(img);
        return;
    }
    ie_bg_options o;
    memset(&o, 0, sizeof o);
    o.mode = c->bg_mode;
    o.color = c->bg_color;
    o.blur = c->bg_blur > 0 ? c->bg_blur : 0.012f * (img->w > img->h ? img->w : img->h);
    o.image = c->bg_image;
    o.decontaminate = c->fix_edges;
    ie_finish_cutout(img, &o);
}

/* `note` receives a breakdown of the slower steps for --verbose. */
static int run_op(cli *c, const op *o, ie_image *img, char *err, size_t errlen, char *note, size_t notelen)
{
    switch (o->kind) {
    case OP_REMOVE_BG: {
        float *matte;
        int mw, mh;
        double t0 = pf_now();
        int r = ie_model_matte(c->nn, img, &matte, &mw, &mh, err, errlen);
        if (r) return r;
        double t1 = pf_now();
        r = ie_matte_apply(img, matte, mw, mh, c->refine, c->eps);
        free(matte);
        if (r) return r;
        double t2 = pf_now();
        finish_mask(c, img);
        snprintf(note, notelen, " [model %.0f, edges %.0f, finish %.0f]", (t1 - t0) * 1000, (t2 - t1) * 1000,
                 (pf_now() - t2) * 1000);
        return IE_OK;
    }
    case OP_KEY_BG: {
        int r = ie_key_background(img, o->i[0], o->color, c->key_tol, c->key_soft, c->key_connected);
        if (!r) finish_mask(c, img);
        return r;
    }
    case OP_INVERT: return ie_invert(img, o->i[0]);
    case OP_GRAYSCALE: return ie_grayscale(img);
    case OP_SEPIA: return ie_sepia(img, o->f[0]);
    case OP_ADJUST: return ie_apply_adjust(img, &o->adj);
    case OP_AUTO_LEVELS: return ie_auto_levels(img, 0.005f, o->i[0]);
    case OP_THRESHOLD: return ie_threshold(img, o->i[0]);
    case OP_POSTERIZE: return ie_posterize(img, o->i[0]);
    case OP_RESIZE: {
        double w = img->w, h = img->h, tw = o->i[0], th = o->i[1];
        if (o->mode == '%') {
            tw = w * o->f[0];
            th = h * o->f[0];
        } else if (o->mode != '!' || tw == 0 || th == 0) {
            double sx = tw > 0 ? tw / w : 0, sy = th > 0 ? th / h : 0, s;
            if (sx == 0) s = sy;
            else if (sy == 0) s = sx;
            else s = o->mode == '^' ? (sx > sy ? sx : sy) : (sx < sy ? sx : sy);
            if (o->mode == '>' && s > 1) s = 1;
            tw = w * s;
            th = h * s;
        }
        int nw = (int)lrint(tw), nh = (int)lrint(th);
        return ie_resize(img, nw < 1 ? 1 : nw, nh < 1 ? 1 : nh);
    }
    case OP_CROP: {
        int x = o->i[2], y = o->i[3];
        if (o->mode == 'c') x = (img->w - o->i[0]) / 2, y = (img->h - o->i[1]) / 2;
        return ie_crop(img, x, y, o->i[0], o->i[1]);
    }
    case OP_ROTATE: return ie_rotate(img, o->f[0], 0);
    case OP_FLIP: return ie_flip(img, o->i[0]);
    case OP_TRIM: return ie_trim(img, o->i[0]);
    case OP_PAD: return ie_pad(img, o->i[0], o->i[0], o->i[0], o->i[0], o->color);
    case OP_BLUR: return ie_blur(img, o->f[0]);
    case OP_SHARPEN: return ie_sharpen(img, o->f[0], 1.0f, 2);
    case OP_FLATTEN: return ie_flatten(img, o->color);
    }
    return IE_EINVAL;
}

static const char *base_name(const char *p)
{
    const char *b = p;
    for (; *p; p++)
        if (*p == '/' || *p == '\\') b = p + 1;
    return b;
}

static int same_path(const char *a, const char *b)
{
#ifdef _WIN32
    for (; *a && *b; a++, b++) {
        int x = *a == '/' ? '\\' : (*a | (*a >= 'A' && *a <= 'Z' ? 32 : 0));
        int y = *b == '/' ? '\\' : (*b | (*b >= 'A' && *b <= 'Z' ? 32 : 0));
        if (x != y) return 0;
    }
    return *a == *b;
#else
    return !strcmp(a, b);
#endif
}

typedef struct {
    cli *c;
    int dir_output;
    char *failed; /* per file, so threads never share a counter */
} batch;

/* Returns 0 on success. */
static int process_one(batch *b, const char *in)
{
    cli *c = b->c;
    char err[1024] = "", out[2048], times[512] = "";
    size_t tl = 0;
    double t0 = pf_now(), t = t0;
    ie_image *img = NULL;
    int r = ie_load(in, &img);
    if (r) {
        fprintf(stderr, "%s: %s\n", in, ie_strerror(r));
        return 1;
    }
    int src_fmt = ie_format_from_name(in);
    if (c->verbose) tl += (size_t)snprintf(times + tl, sizeof times - tl, "load %.0f ms", (pf_now() - t) * 1000);

    for (int k = 0; k < c->nops && r == IE_OK; k++) {
        char note[128] = "";
        t = pf_now();
        r = run_op(c, &c->ops[k], img, err, sizeof err, note, sizeof note);
        if (c->verbose && tl < sizeof times)
            tl += (size_t)snprintf(times + tl, sizeof times - tl, ", %s %.0f ms%s", OP_NAMES[c->ops[k].kind],
                                   (pf_now() - t) * 1000, note);
    }
    if (r) {
        fprintf(stderr, "%s: %s\n", in, err[0] ? err : ie_strerror(r));
        ie_free(img);
        return 1;
    }

    /* Output format: --format, then the -o extension, then keep the input's
     * format unless the result needs alpha that it cannot hold. */
    int fmt = c->format;
    if (!fmt && c->output && !b->dir_output) fmt = ie_format_from_name(c->output);
    if (!fmt) {
        fmt = src_fmt ? src_fmt : IE_FMT_PNG;
        if (!ie_format_has_alpha(fmt) && ie_has_alpha(img)) fmt = IE_FMT_PNG;
    }
    const char *bn = base_name(in);
    const char *dot = strrchr(bn, '.');
    int stem = dot && dot != bn ? (int)(dot - bn) : (int)strlen(bn);
    if (c->output && !b->dir_output) {
        snprintf(out, sizeof out, "%s", c->output);
    } else if (c->output) {
        size_t ol = strlen(c->output);
        int sep = ol && (c->output[ol - 1] == '/' || c->output[ol - 1] == '\\');
        snprintf(out, sizeof out, "%s%s%.*s.%s", c->output, sep ? "" : (PF_SEP == '/' ? "/" : "\\"), stem, bn,
                 ie_format_ext(fmt));
    } else {
        snprintf(out, sizeof out, "%.*s%.*s-out.%s", (int)(bn - in), in, stem, bn, ie_format_ext(fmt));
    }
    if (same_path(out, in) && !c->overwrite) {
        fprintf(stderr, "%s: output would overwrite the input (use -y to allow)\n", in);
        ie_free(img);
        return 1;
    }
    t = pf_now();
    r = ie_save(img, out, fmt, c->quality);
    if (c->verbose && tl < sizeof times)
        tl += (size_t)snprintf(times + tl, sizeof times - tl, ", save %.0f ms", (pf_now() - t) * 1000);
    if (r) {
        fprintf(stderr, "%s: cannot write %s: %s\n", in, out, ie_strerror(r));
    } else {
        char line[3000];
        if (c->verbose)
            snprintf(line, sizeof line, "%s -> %s  %dx%d  %.0f ms (%s)\n", in, out, img->w, img->h,
                     (pf_now() - t0) * 1000, times);
        else
            snprintf(line, sizeof line, "%s -> %s  %dx%d  %.2f s\n", in, out, img->w, img->h, pf_now() - t0);
        fputs(line, stdout);
        fflush(stdout);
    }
    ie_free(img);
    return r != IE_OK;
}

typedef struct {
    batch *b;
    char **files;
} batch_ctx;

static void batch_range(void *p, int i0, int i1)
{
    batch_ctx *bc = (batch_ctx *)p;
    for (int i = i0; i < i1; i++) bc->b->failed[i] = (char)process_one(bc->b, bc->files[i]);
}

static int is_image_name(const char *name)
{
    static const char *ext[] = {"png", "jpg", "jpeg", "jpe", "jfif", "bmp", "tga", "gif", "psd",
                                "hdr", "pic", "ppm", "pgm", "pnm", "pam", "qoi", NULL};
    const char *dot = strrchr(name, '.');
    if (!dot) return 0;
    for (int i = 0; ext[i]; i++) {
        const char *a = dot + 1, *e = ext[i];
        while (*a && (*a | 32) == *e) a++, e++;
        if (!*a && !*e) return 1;
    }
    return 0;
}

static int cmd_info(int n, char **files)
{
    int bad = 0;
    for (int i = 0; i < n; i++) {
        uint8_t *data;
        size_t len;
        ie_image *img;
        if (ie_read_file(files[i], &data, &len) || ie_decode(data, len, &img)) {
            fprintf(stderr, "%s: cannot read image\n", files[i]);
            bad = 1;
            continue;
        }
        printf("%s: %dx%d, %s, %s, %.1f KB\n", files[i], img->w, img->h,
               ie_format_ext(ie_format_from_name(files[i])) ? ie_format_ext(ie_format_from_name(files[i])) : "?",
               ie_has_alpha(img) ? "with transparency" : (ie_is_gray(img) ? "grayscale" : "opaque color"),
               len / 1024.0);
        free(data);
        ie_free(img);
    }
    return bad;
}

int main(int argc, char **argv)
{
    pf_init_args(&argc, &argv);
    cli c;
    memset(&c, 0, sizeof c);
    c.refine = 4;
    c.eps = 1e-4f;
    c.hard = -1;
    c.fix_edges = 1;
    c.key_tol = 10;
    c.key_soft = 8;
    c.key_connected = 1;
    c.device = "auto";

    if (argc >= 2 && !strcmp(argv[1], "models")) {
        for (int i = 2; i + 1 < argc; i++)
            if (!strcmp(argv[i], "--model-dir")) c.model_dir = argv[i + 1];
        list_models(&c);
        return 0;
    }
    if (argc >= 2 && !strcmp(argv[1], "info")) return cmd_info(argc - 2, argv + 2);
    if (argc < 2) {
        usage(stderr);
        return 2;
    }
    parse_args(&c, argc, argv);
    if (c.threads > 0) ie_set_threads(c.threads);
    if (!c.ninputs) die("%s", "no input files (see --help)");

    /* Expand folders into their image files. */
    char **files = NULL;
    int nfiles = 0, cap = 0, from_dir = 0;
    for (int i = 0; i < c.ninputs; i++) {
        char **names = NULL;
        int nn = 0;
        if (pf_is_dir(c.inputs[i]) && pf_list_dir(c.inputs[i], &names, &nn) == 0) {
            from_dir = 1;
            for (int k = 0; k < nn; k++) {
                if (!is_image_name(names[k])) continue;
                if (nfiles == cap) files = (char **)realloc(files, sizeof(char *) * (size_t)(cap = cap ? cap * 2 : 64));
                size_t L = strlen(c.inputs[i]) + strlen(names[k]) + 2;
                files[nfiles] = (char *)malloc(L);
                snprintf(files[nfiles++], L, "%s%c%s", c.inputs[i], PF_SEP, names[k]);
            }
            pf_free_list(names, nn);
        } else {
            if (nfiles == cap) files = (char **)realloc(files, sizeof(char *) * (size_t)(cap = cap ? cap * 2 : 64));
            files[nfiles++] = c.inputs[i];
        }
    }
    if (!nfiles) die("%s", "no images found");

    batch b;
    memset(&b, 0, sizeof b);
    b.c = &c;
    b.dir_output = c.output && (nfiles > 1 || from_dir || pf_is_dir(c.output) ||
                                c.output[strlen(c.output) - 1] == '/' || c.output[strlen(c.output) - 1] == '\\');
    if (b.dir_output && nfiles > 1 && ie_format_from_name(c.output) && !pf_is_dir(c.output))
        die("several inputs need an output folder, not the file name %s", c.output);
    if (b.dir_output && pf_mkdirs(c.output) != 0) die("cannot create folder %s", c.output);
    b.failed = (char *)calloc((size_t)nfiles, 1);
    if (!b.failed) die("%s", "out of memory");

    if (c.bg_mode == IE_BG_IMAGE && ie_load(c.bg_file, &c.bg_image) != IE_OK)
        die("cannot read background image %s", c.bg_file);
    /* Default model: the sharpest one when a GPU can run it, else the CPU-friendly one. */
    if (c.uses_model && !c.model) c.model = ie_runtime_gpu() ? "best" : "balanced";
    if (c.uses_model && !strcmp(c.device, "auto")) {
        /* A GPU session costs seconds to set up, which only pays off for heavy
         * models or for batches; small jobs finish sooner on the CPU. */
        const ie_model_info *m = ie_model_find(c.model ? c.model : "balanced");
        if (!(m && m->mb >= 200) && nfiles < 8) c.device = "cpu";
    }
    if (c.uses_model && !ensure_model(&c)) return 1;

    double t0 = pf_now();
    if (nfiles > 1 && !c.uses_model) {
        /* Many files without a model: one file per thread is faster than
         * splitting every small image across all cores. */
        int threads = ie_get_threads();
        ie_set_threads(threads < 8 ? threads : 8);
        batch_ctx bc = {&b, files};
        ie_parallel_for(nfiles, 1, batch_range, &bc);
        ie_set_threads(c.threads);
    } else {
        for (int i = 0; i < nfiles; i++) b.failed[i] = (char)process_one(&b, files[i]);
    }
    int failures = 0;
    for (int i = 0; i < nfiles; i++) failures += b.failed[i];
    if (nfiles > 1) fprintf(stderr, "%d of %d images done in %.2f s\n", nfiles - failures, nfiles, pf_now() - t0);
    ie_model_close(c.nn);
    ie_free(c.bg_image);
    return failures ? 1 : 0;
}
