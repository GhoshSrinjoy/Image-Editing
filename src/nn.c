#include "nn.h"
#include "internal.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../third_party/onnxruntime/onnxruntime_c_api.h"
#include "platform.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
typedef HMODULE lib_t;
#else
#include <dlfcn.h>
typedef void *lib_t;
#endif

/* ---- model registry ---------------------------------------------------- */

#define REMBG "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
#define IMAGENET_MEAN {0.485f, 0.456f, 0.406f}
#define IMAGENET_STD {0.229f, 0.224f, 0.225f}

static const ie_model_info MODELS[] = {
    {"u2netp", "fast", "u2netp.onnx", REMBG "u2netp.onnx", 320, IMAGENET_MEAN, IMAGENET_STD, 1 / 255.0f, 1, 5,
     "tiny general model, very fast, softer edges"},
    {"birefnet-512", "balanced", "birefnet-lite-512.onnx",
     "https://huggingface.co/studioludens/birefnet-lite-512/resolve/4a3c40c36c94093cc1e724d9ea428b8fa4b57dc7/onnx/model.onnx",
     512, IMAGENET_MEAN, IMAGENET_STD, 1 / 255.0f, 1, 192, "BiRefNet at 512 px: finds whole subjects, about 1 s on a CPU"},
    {"isnet", NULL, "isnet-general-use.onnx", REMBG "isnet-general-use.onnx", 1024, {0.5f, 0.5f, 0.5f}, {1, 1, 1},
     1 / 255.0f, 1, 179, "IS-Net general, quick on CPU"},
    {"birefnet-lite", "best", "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
     REMBG "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx", 1024, IMAGENET_MEAN, IMAGENET_STD, 1 / 255.0f, 1, 224,
     "BiRefNet at 1024 px: sharpest hair and detail (default with a GPU)"},
    {"birefnet", "max", "BiRefNet-general-epoch_244.onnx", REMBG "BiRefNet-general-epoch_244.onnx", 1024,
     IMAGENET_MEAN, IMAGENET_STD, 1 / 255.0f, 1, 973, "full-size BiRefNet, highest quality, slow without a GPU"},
    {"birefnet-portrait", NULL, "BiRefNet-portrait-epoch_150.onnx", REMBG "BiRefNet-portrait-epoch_150.onnx", 1024,
     IMAGENET_MEAN, IMAGENET_STD, 1 / 255.0f, 1, 973, "BiRefNet tuned for portraits"},
    {"modnet", "portrait", "modnet.onnx", "https://huggingface.co/Xenova/modnet/resolve/main/onnx/model.onnx", 0,
     {0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, 1 / 255.0f, 0, 26, "portrait matting, fine hair on people"},
    {"u2net", NULL, "u2net.onnx", REMBG "u2net.onnx", 320, IMAGENET_MEAN, IMAGENET_STD, 1 / 255.0f, 1, 176,
     "classic U2-Net"},
    {"u2net-human", "human", "u2net_human_seg.onnx", REMBG "u2net_human_seg.onnx", 320, IMAGENET_MEAN, IMAGENET_STD,
     1 / 255.0f, 1, 176, "U2-Net trained on people"},
    {"isnet-anime", "anime", "isnet-anime.onnx", REMBG "isnet-anime.onnx", 1024, {0.485f, 0.456f, 0.406f},
     {1, 1, 1}, 1 / 255.0f, 1, 176, "anime and illustrations"},
    {"silueta", NULL, "silueta.onnx", REMBG "silueta.onnx", 320, IMAGENET_MEAN, IMAGENET_STD, 1 / 255.0f, 1, 44,
     "compressed U2-Net"},
};

const ie_model_info *ie_models(int *count)
{
    *count = (int)(sizeof MODELS / sizeof MODELS[0]);
    return MODELS;
}

static int ieq(const char *a, const char *b)
{
    for (; *a && *b; a++, b++) {
        int x = *a >= 'A' && *a <= 'Z' ? *a + 32 : *a, y = *b >= 'A' && *b <= 'Z' ? *b + 32 : *b;
        if (x != y) return 0;
    }
    return *a == *b;
}

const ie_model_info *ie_model_find(const char *name)
{
    if (!name) return NULL;
    for (size_t i = 0; i < sizeof MODELS / sizeof MODELS[0]; i++)
        if (ieq(name, MODELS[i].name) || (MODELS[i].alias && ieq(name, MODELS[i].alias))) return &MODELS[i];
    return NULL;
}

const ie_model_info *ie_model_for_file(const char *path)
{
    const char *base = path;
    for (const char *p = path; *p; p++)
        if (*p == '/' || *p == '\\') base = p + 1;
    for (size_t i = 0; i < sizeof MODELS / sizeof MODELS[0]; i++)
        if (ieq(base, MODELS[i].file)) return &MODELS[i];
    return NULL;
}

/* ---- runtime loading --------------------------------------------------- */

static const OrtApi *ort;
static OrtEnv *env;
static lib_t lib;
static char lib_desc[1100];
static char lib_path[1024]; /* the runtime actually loaded */
static char lib_error[1100];
static int load_tried;

static lib_t lib_open(const char *path)
{
#ifdef _WIN32
    wchar_t w[1024];
    if (!MultiByteToWideChar(CP_UTF8, 0, path, -1, w, 1024)) return NULL;
    int has_dir = strchr(path, '\\') || strchr(path, '/');
    return LoadLibraryExW(w, NULL, has_dir ? LOAD_WITH_ALTERED_SEARCH_PATH : 0);
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

static void *lib_sym(lib_t h, const char *name)
{
#ifdef _WIN32
    return (void *)GetProcAddress(h, name);
#else
    return dlsym(h, name);
#endif
}

static int try_runtime(const char *path)
{
    lib_t h = lib_open(path);
    if (!h) return 0;
    typedef const OrtApiBase *(ORT_API_CALL * get_base_fn)(void);
    get_base_fn get_base = (get_base_fn)lib_sym(h, "OrtGetApiBase");
    if (!get_base) return 0;
    const OrtApiBase *base = get_base();
    /* Ask for the newest API this library has (its minor version), so older
     * system copies work without printing version errors. */
    const char *ver = base->GetVersionString ? base->GetVersionString() : "1.0.0";
    int major = 0, minor = 0;
    sscanf(ver, "%d.%d", &major, &minor);
    int want = minor < ORT_API_VERSION ? minor : ORT_API_VERSION;
    if (major != 1 || want < 12) {
        snprintf(lib_error, sizeof lib_error, "%s is ONNX Runtime %s; 1.12 or newer is needed", path, ver);
        return 0;
    }
    const OrtApi *api = base->GetApi((uint32_t)want);
    if (!api) return 0;
    /* Errors come back as statuses and are reported by imgedit itself. */
    OrtStatus *st = api->CreateEnv(ORT_LOGGING_LEVEL_FATAL, "imgedit", &env);
    if (st) {
        api->ReleaseStatus(st);
        return 0;
    }
    ort = api;
    lib = h;
    snprintf(lib_desc, sizeof lib_desc, "ONNX Runtime %s (%s)", ver, path);
    snprintf(lib_path, sizeof lib_path, "%s", path);
    return 1;
}

int ie_runtime_load(char *desc, size_t desclen, char *err, size_t errlen)
{
    if (!load_tried) {
        load_tried = 1;
        char dir[1024], path[1200];
#ifdef _WIN32
        static const char *names[] = {"onnxruntime.dll", NULL};
#elif defined(__APPLE__)
        static const char *names[] = {"libonnxruntime.dylib", "/opt/homebrew/lib/libonnxruntime.dylib",
                                      "/usr/local/lib/libonnxruntime.dylib", NULL};
#else
        static const char *names[] = {"libonnxruntime.so", "libonnxruntime.so.1", NULL};
#endif
        const char *env_path = pf_getenv("IMGEDIT_ORT");
        int ok = env_path && try_runtime(env_path);
        int have_dir = pf_exe_dir(dir, sizeof dir) == 0;
        for (int i = 0; !ok && have_dir && names[i]; i++) {
            if (names[i][0] == '/') continue;
            snprintf(path, sizeof path, "%s%c%s", dir, PF_SEP, names[i]);
            ok = pf_exists(path) && try_runtime(path);
        }
        for (int i = 0; !ok && names[i]; i++) ok = try_runtime(names[i]);
        if (!ok && !lib_error[0])
            snprintf(lib_error, sizeof lib_error,
                     "ONNX Runtime not found. Put the onnxruntime library next to imgedit, install it "
                     "system-wide, or set IMGEDIT_ORT to its path (see README).");
    }
    if (!ort) {
        if (err) snprintf(err, errlen, "%s", lib_error);
        return IE_EUNSUPPORTED;
    }
    if (desc) snprintf(desc, desclen, "%s", lib_desc);
    return IE_OK;
}

/* ---- sessions ---------------------------------------------------------- */

struct ie_model {
    OrtSession *session;
    OrtMemoryInfo *mem;
    char *in_name, *out_name;
    ONNXTensorElementDataType in_type;
    int in_w, in_h; /* <= 0 when dynamic */
    ie_model_info info;
    char device[16];
};

static int status_err(OrtStatus *st, const char *what, char *err, size_t errlen)
{
    if (!st) return 0;
    if (err) snprintf(err, errlen, "%s: %s", what, ort->GetErrorMessage(st));
    ort->ReleaseStatus(st);
    return 1;
}

/* For calls whose failure is harmless (the default stays in effect). */
static void ignore(OrtStatus *st)
{
    if (st) ort->ReleaseStatus(st);
}

/* CUDA lives in a separate provider library; only probe for it when it is
 * installed next to the runtime. */
static int cuda_present(void)
{
#ifdef _WIN32
    static const char *name = "onnxruntime_providers_cuda.dll";
#else
    static const char *name = "libonnxruntime_providers_cuda.so";
#endif
    char dir[1024], probe[1200];
    snprintf(dir, sizeof dir, "%s", lib_path);
    char *slash = NULL;
    for (char *c = dir; *c; c++)
        if (*c == '/' || *c == '\\') slash = c;
    if (!slash) return 0; /* found through the system search path: no CUDA build ships that way */
    *slash = 0;
    snprintf(probe, sizeof probe, "%s%c%s", dir, PF_SEP, name);
    return pf_exists(probe);
}

static int append_cuda(OrtSessionOptions *so)
{
    if (!cuda_present()) return 0;
    OrtCUDAProviderOptionsV2 *o = NULL;
    OrtStatus *st = ort->CreateCUDAProviderOptions(&o);
    if (st) { /* CPU-only builds say no here */
        ort->ReleaseStatus(st);
        return 0;
    }
    st = ort->SessionOptionsAppendExecutionProvider_CUDA_V2(so, o);
    ort->ReleaseCUDAProviderOptions(o);
    if (st) {
        ort->ReleaseStatus(st);
        return 0;
    }
    return 1;
}

static int append_dml(OrtSessionOptions *so)
{
    typedef OrtStatus *(ORT_API_CALL * dml_fn)(OrtSessionOptions *, int);
    dml_fn fn = (dml_fn)lib_sym(lib, "OrtSessionOptionsAppendExecutionProvider_DML");
    if (!fn) { /* newer builds expose it through the provider API table */
        const void *dml_api = NULL;
        OrtStatus *st = ort->GetExecutionProviderApi("DML", ORT_API_VERSION, &dml_api);
        if (st) {
            ort->ReleaseStatus(st);
            return 0;
        }
        if (!dml_api) return 0;
        fn = *(const dml_fn *)dml_api; /* first entry of OrtDmlApi */
    }
    /* DirectML needs sequential execution without memory patterns. */
    OrtStatus *st = ort->DisableMemPattern(so);
    if (!st) st = ort->SetSessionExecutionMode(so, ORT_SEQUENTIAL);
    if (!st) st = fn(so, 0);
    if (st) {
        ort->ReleaseStatus(st);
        return 0;
    }
    return 1;
}

static int append_coreml(OrtSessionOptions *so)
{
    typedef OrtStatus *(ORT_API_CALL * coreml_fn)(OrtSessionOptions *, uint32_t);
    coreml_fn fn = (coreml_fn)lib_sym(lib, "OrtSessionOptionsAppendExecutionProvider_CoreML");
    OrtStatus *st = fn ? fn(so, 0) : ort->SessionOptionsAppendExecutionProvider(so, "CoreML", NULL, NULL, 0);
    if (st) {
        ort->ReleaseStatus(st);
        return 0;
    }
    return 1;
}

/* Returns the device actually attached, "cpu" when nothing else worked. */
static const char *attach_device(OrtSessionOptions *so, const char *device)
{
    int any = !device || !*device || ieq(device, "auto") || ieq(device, "gpu");
    if ((any || ieq(device, "cuda")) && append_cuda(so)) return "cuda";
    if ((any || ieq(device, "dml") || ieq(device, "directml")) && append_dml(so)) return "dml";
    if ((any || ieq(device, "coreml")) && append_coreml(so)) return "coreml";
    return "cpu";
}

const char *ie_runtime_gpu(void)
{
    if (ie_runtime_load(NULL, 0, NULL, 0) != IE_OK) return NULL;
    OrtSessionOptions *so = NULL;
    OrtStatus *st = ort->CreateSessionOptions(&so);
    if (st) {
        ort->ReleaseStatus(st);
        return NULL;
    }
    const char *dev = attach_device(so, "gpu");
    ort->ReleaseSessionOptions(so);
    return strcmp(dev, "cpu") ? dev : NULL;
}

static OrtSession *create_session(const char *path, OrtSessionOptions *so, char *err, size_t errlen)
{
    OrtSession *s = NULL;
#ifdef _WIN32
    wchar_t w[2048];
    if (!MultiByteToWideChar(CP_UTF8, 0, path, -1, w, 2048)) return NULL;
    OrtStatus *st = ort->CreateSession(env, w, so, &s);
#else
    OrtStatus *st = ort->CreateSession(env, path, so, &s);
#endif
    if (status_err(st, "loading model", err, errlen)) return NULL;
    return s;
}

int ie_model_open(ie_model **out, const char *path, const ie_model_info *info, const char *device, int threads,
                  char *err, size_t errlen)
{
    *out = NULL;
    int r = ie_runtime_load(NULL, 0, err, errlen);
    if (r) return r;
    ie_model *m = (ie_model *)calloc(1, sizeof *m);
    if (!m) return IE_ENOMEM;
    m->info = *info;

    OrtSessionOptions *so = NULL;
    OrtAllocator *alloc = NULL;
    OrtTypeInfo *ti = NULL;
    const OrtTensorTypeAndShapeInfo *tsi = NULL;
    int ok = 0;
    if (status_err(ort->CreateSessionOptions(&so), "session options", err, errlen)) goto done;
    if (threads > 0) ignore(ort->SetIntraOpNumThreads(so, threads));
    ignore(ort->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL));

    const char *dev = ieq(device ? device : "auto", "cpu") ? "cpu" : attach_device(so, device);
    m->session = create_session(path, so, err, errlen);
    if (!m->session && strcmp(dev, "cpu") != 0) {
        /* The GPU provider could not take this model: fall back to the CPU. */
        ort->ReleaseSessionOptions(so);
        so = NULL;
        if (status_err(ort->CreateSessionOptions(&so), "session options", err, errlen)) goto done;
        if (threads > 0) ignore(ort->SetIntraOpNumThreads(so, threads));
        ignore(ort->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL));
        dev = "cpu";
        m->session = create_session(path, so, err, errlen);
    }
    if (!m->session) goto done;
    snprintf(m->device, sizeof m->device, "%s", dev);

    if (status_err(ort->GetAllocatorWithDefaultOptions(&alloc), "allocator", err, errlen)) goto done;
    char *name = NULL;
    if (status_err(ort->SessionGetInputName(m->session, 0, alloc, &name), "input name", err, errlen)) goto done;
    m->in_name = (char *)malloc(strlen(name) + 1);
    if (m->in_name) memcpy(m->in_name, name, strlen(name) + 1);
    ignore(ort->AllocatorFree(alloc, name));
    if (status_err(ort->SessionGetOutputName(m->session, 0, alloc, &name), "output name", err, errlen)) goto done;
    m->out_name = (char *)malloc(strlen(name) + 1);
    if (m->out_name) memcpy(m->out_name, name, strlen(name) + 1);
    ignore(ort->AllocatorFree(alloc, name));
    if (!m->in_name || !m->out_name) goto done;

    if (status_err(ort->SessionGetInputTypeInfo(m->session, 0, &ti), "input info", err, errlen)) goto done;
    if (status_err(ort->CastTypeInfoToTensorInfo(ti, &tsi), "input info", err, errlen) || !tsi) goto done;
    ignore(ort->GetTensorElementType(tsi, &m->in_type));
    size_t nd = 0;
    ignore(ort->GetDimensionsCount(tsi, &nd));
    if (nd == 4) {
        int64_t dims[4];
        ignore(ort->GetDimensions(tsi, dims, 4));
        m->in_h = dims[2] > 0 ? (int)dims[2] : 0;
        m->in_w = dims[3] > 0 ? (int)dims[3] : 0;
    }
    if (m->in_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && m->in_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        snprintf(err, errlen, "model input must be a float tensor");
        goto done;
    }
    if (status_err(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &m->mem), "memory info", err,
                   errlen))
        goto done;
    ok = 1;
done:
    if (ti) ort->ReleaseTypeInfo(ti);
    if (so) ort->ReleaseSessionOptions(so);
    if (!ok) {
        ie_model_close(m);
        return IE_EFORMAT;
    }
    *out = m;
    return IE_OK;
}

const char *ie_model_device(const ie_model *m) { return m->device; }

void ie_model_close(ie_model *m)
{
    if (!m) return;
    if (m->mem) ort->ReleaseMemoryInfo(m->mem);
    if (m->session) ort->ReleaseSession(m->session);
    free(m->in_name);
    free(m->out_name);
    free(m);
}

/* ---- inference --------------------------------------------------------- */

static uint16_t f32_to_f16(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000, mant = x & 0x7fffff;
    int exp = (int)((x >> 23) & 0xff) - 127 + 15;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = mant >> shift;
        if ((mant >> (shift - 1)) & 1) half++;
        return (uint16_t)(sign | half);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00 | (((x >> 23) & 0xff) == 0xff && mant ? 0x200 : 0));
    uint32_t h = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000) h++; /* round half up; carries into the exponent correctly */
    return (uint16_t)h;
}

static float f16_to_f32(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1f, mant = h & 0x3ff, x;
    if (exp == 0) {
        if (!mant) {
            x = sign;
        } else {
            exp = 127 - 15 + 1;
            while (!(mant & 0x400)) mant <<= 1, exp--;
            x = sign | (exp << 23) | ((mant & 0x3ff) << 13);
        }
    } else if (exp == 31) {
        x = sign | 0x7f800000 | (mant << 13);
    } else {
        x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &x, 4);
    return f;
}

static int round32(double v)
{
    int r = (int)(v / 32.0 + 0.5) * 32;
    return r < 32 ? 32 : r;
}

int ie_model_matte(ie_model *m, const ie_image *img, float **matte, int *mw, int *mh, char *err, size_t errlen)
{
    *matte = NULL;
    int tw = m->in_w, th = m->in_h;
    if (tw <= 0 || th <= 0) {
        if (m->info.size > 0) {
            tw = th = m->info.size;
        } else { /* dynamic: shortest side 512, both sides multiples of 32 */
            double s = 512.0 / (img->w < img->h ? img->w : img->h);
            tw = round32(img->w * s);
            th = round32(img->h * s);
            if (tw > 2048) tw = 2048;
            if (th > 2048) th = 2048;
        }
    }
    const size_t n = (size_t)tw * th * 3;
    float *input = (float *)malloc(sizeof(float) * n);
    uint16_t *half = NULL;
    if (!input) return IE_ENOMEM;
    int r = ie_nn_input(img, tw, th, m->info.mean, m->info.std, m->info.scale, input);
    if (r) {
        free(input);
        return r;
    }
    void *data = input;
    size_t bytes = sizeof(float) * n;
    if (m->in_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        half = (uint16_t *)malloc(sizeof(uint16_t) * n);
        if (!half) {
            free(input);
            return IE_ENOMEM;
        }
        for (size_t i = 0; i < n; i++) half[i] = f32_to_f16(input[i]);
        data = half;
        bytes = sizeof(uint16_t) * n;
    }

    int64_t shape[4] = {1, 3, th, tw};
    OrtValue *in = NULL, *out = NULL;
    OrtTensorTypeAndShapeInfo *oi = NULL;
    r = IE_EFORMAT;
    if (status_err(ort->CreateTensorWithDataAsOrtValue(m->mem, data, bytes, shape, 4, m->in_type, &in),
                   "input tensor", err, errlen))
        goto done;
    const char *in_names[1] = {m->in_name}, *out_names[1] = {m->out_name};
    if (status_err(ort->Run(m->session, NULL, in_names, (const OrtValue *const *)&in, 1, out_names, 1, &out),
                   "inference", err, errlen))
        goto done;
    if (status_err(ort->GetTensorTypeAndShape(out, &oi), "output info", err, errlen)) goto done;
    size_t nd = 0;
    int64_t dims[8] = {0};
    ONNXTensorElementDataType ot = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    ignore(ort->GetDimensionsCount(oi, &nd));
    if (nd < 2 || nd > 8) {
        snprintf(err, errlen, "unexpected output rank %d", (int)nd);
        goto done;
    }
    ignore(ort->GetDimensions(oi, dims, nd));
    ignore(ort->GetTensorElementType(oi, &ot));
    int oh = (int)dims[nd - 2], ow = (int)dims[nd - 1];
    void *od = NULL;
    if (oh <= 0 || ow <= 0 || status_err(ort->GetTensorMutableData(out, &od), "output data", err, errlen)) goto done;
    size_t on = (size_t)ow * oh;
    float *res = (float *)malloc(sizeof(float) * on);
    if (!res) {
        r = IE_ENOMEM;
        goto done;
    }
    if (ot == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
        for (size_t i = 0; i < on; i++) res[i] = f16_to_f32(((const uint16_t *)od)[i]);
    else if (ot == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        memcpy(res, od, sizeof(float) * on);
    else {
        free(res);
        snprintf(err, errlen, "model output must be a float tensor");
        goto done;
    }
    ie_nn_output(res, on, m->info.minmax);
    *matte = res;
    *mw = ow;
    *mh = oh;
    r = IE_OK;
done:
    if (oi) ort->ReleaseTensorTypeAndShapeInfo(oi);
    if (out) ort->ReleaseValue(out);
    if (in) ort->ReleaseValue(in);
    free(input);
    free(half);
    return r;
}
