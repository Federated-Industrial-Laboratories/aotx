/* Purpose: Read versioned control evidence and exact accepted settings.
 * Owns: Portable file references and bounded admission results.
 * Threading: One disk reader checks the complete control batch.
 * Lifetime: Evidence remains with the control through packing and recovery. */
#ifndef AOTX_RUNTIME_QUALIFICATION_H
#define AOTX_RUNTIME_QUALIFICATION_H
#include "disk/runtime/control.h"
#define AOTX_QUALIFICATION_BYTES 8192u
#define AOTX_QUALIFICATION_DOSES 16u
#define AOTX_QUALIFICATION_REFERENCES 7u
#define AOTX_QUALIFICATION_UNAVAILABLE 0u
#define AOTX_QUALIFICATION_ACCEPTED 1u
#define AOTX_QUALIFICATION_MEASUREMENT 2u
#define AOTX_QUALIFICATION_CHECKS 255u
typedef struct aotx_control_permit {
    unsigned status, count;
    int dose[AOTX_QUALIFICATION_DOSES]; /* Units of 0.0001. No interpolation. */
    unsigned char digest[32];
} aotx_control_permit;
typedef struct aotx_control_reference_file {
    char name[256];
    unsigned char digest[32];
} aotx_control_reference_file;
typedef struct aotx_qualification {
    unsigned kind, checks;
    aotx_control_permit permit;
    /* Binding, source, commitments, examples, calibration, acceptance, consumer. */
    aotx_control_reference_file reference[AOTX_QUALIFICATION_REFERENCES];
} aotx_qualification;
#ifdef __cplusplus
extern "C" {
#endif
int aotx_qualification_parse(const void *bytes, size_t length, aotx_qualification *out);
/* No file leaves the control unavailable. An invalid file is an error. */
int aotx_qualification_read(const char *store, const char *asset, unsigned kind,
    aotx_control_permit *permit);
int aotx_qualification_references(const struct aotx_ccir_view *view,
    const struct aotx_runtime_index *index);
#ifdef __cplusplus
}
#endif
#endif
