/* Purpose: Define bounded tool call forms selected from model template bytes.
 * Owns: One byte table and its span bounds.
 * Threading: One caller for each model file.
 * Lifetime: From model load to model release. */
#ifndef AOTX_MODELFILE_CALL_FORMAT_H
#define AOTX_MODELFILE_CALL_FORMAT_H

#include <stddef.h>
#include <stdint.h>

#define AOTX_CALL_FORMAT_BYTES 2048u
#define AOTX_CALL_FORMAT_SPAN_BYTES 1024u

enum aotx_call_format_kind {
    AOTX_CALL_NONE, AOTX_CALL_HERMES, AOTX_CALL_LLAMA_JSON, AOTX_CALL_QWEN_XML,
    AOTX_CALL_FORMAT_KINDS
};

enum aotx_call_format_span {
    AOTX_CALL_SYSTEM_PREFIX, AOTX_CALL_TOOLS_HEAD, AOTX_CALL_TOOLS_TAIL,
    AOTX_CALL_INSTRUCTION, AOTX_CALL_HEAD, AOTX_CALL_TAIL,
    AOTX_CALL_NAME_HEAD, AOTX_CALL_NAME_TAIL, AOTX_CALL_NAME_CLOSE,
    AOTX_CALL_ARG_HEAD, AOTX_CALL_ARG_TAIL, AOTX_CALL_ARG_KEY,
    AOTX_CALL_RESULT_HEAD, AOTX_CALL_RESULT_TAIL, AOTX_CALL_FORMAT_SPANS
};

typedef struct aotx_call_format {
    unsigned char bytes[AOTX_CALL_FORMAT_BYTES];
    uint16_t offset[AOTX_CALL_FORMAT_SPANS];
    uint16_t length[AOTX_CALL_FORMAT_SPANS];
    uint32_t kind;
    uint32_t result_json;
} aotx_call_format;

#ifdef __cplusplus
extern "C" {
#endif
struct aotx_modelfile;
/* Unknown template bytes select no tool form. Text conversation remains available. */
int aotx_call_format_read(const struct aotx_modelfile *file, aotx_call_format *out);
int aotx_call_format_make(unsigned int kind, aotx_call_format *out);
int aotx_call_format_valid(const aotx_call_format *format);
#ifdef __cplusplus
}
#endif
#endif
