/*
 * Background matting models run through ONNX Runtime. The runtime library is
 * loaded on first use, so the tool has no link-time dependency on it and
 * everything except model-based background removal works without it.
 */
#ifndef IE_NN_H
#define IE_NN_H

#include "imgedit.h"

typedef struct ie_model_info {
    const char *name;  /* e.g. "isnet" */
    const char *alias; /* e.g. "balanced", or NULL */
    const char *file;  /* file name in the model directory */
    const char *url;   /* where to download it */
    int size;          /* square input size, 0 = dynamic (shortest side 512) */
    float mean[3], std[3], scale;
    int minmax; /* stretch the predicted matte to the full range */
    int mb;     /* download size */
    const char *about;
} ie_model_info;

const ie_model_info *ie_models(int *count);
const ie_model_info *ie_model_find(const char *name_or_alias);
const ie_model_info *ie_model_for_file(const char *path); /* match by file name */

typedef struct ie_model ie_model;

/* Load ONNX Runtime (once). Fills `desc` with the library path and version. */
int ie_runtime_load(char *desc, size_t desclen, char *err, size_t errlen);

/* device: "auto" (GPU when the runtime has one, else CPU), "cpu", "gpu",
 * "cuda", "dml" or "coreml". threads: 0 = runtime default. */
int ie_model_open(ie_model **out, const char *path, const ie_model_info *info, const char *device, int threads,
                  char *err, size_t errlen);
const char *ie_model_device(const ie_model *m);

/* Predict the subject matte: a malloc'd mw x mh float image with values 0..1. */
int ie_model_matte(ie_model *m, const ie_image *img, float **matte, int *mw, int *mh, char *err, size_t errlen);
void ie_model_close(ie_model *m);

#endif
