/* Purpose: Define bounded model turn spans and the disk-side reader.
 * Owns: The byte table and its end-token ids.
 * Threading: One caller for each table; no mutable shared state.
 * Lifetime: From model load to model release. */
#ifndef AOTX_MODELFILE_WRAP_H
#define AOTX_MODELFILE_WRAP_H

#include <stdint.h>
#include <stddef.h>

#define AOTX_WRAP_SPANS 9u
#define AOTX_WRAP_SPAN_BYTES 64u
#define AOTX_WRAP_BYTES 432u
#define AOTX_WRAP_ENDS 8u

enum aotx_wrap_span {
    AOTX_WRAP_SYSTEM_HEAD, AOTX_WRAP_SYSTEM_TAIL,
    AOTX_WRAP_USER_HEAD, AOTX_WRAP_USER_TAIL,
    AOTX_WRAP_ASSISTANT_HEAD, AOTX_WRAP_ASSISTANT_TAIL,
    AOTX_WRAP_GENERATION_HEAD, AOTX_WRAP_THINK_OPEN, AOTX_WRAP_THINK_CLOSE
};

/* The prefix is the first bytes of system_head. Without system text, emit it once
 * before the first user header. Stored turns never emit this prefix. */
typedef struct aotx_wrap {
    unsigned char bytes[AOTX_WRAP_BYTES];
    uint16_t offset[AOTX_WRAP_SPANS];
    uint8_t length[AOTX_WRAP_SPANS];
    uint8_t prefix_length;
    uint32_t end_ids[AOTX_WRAP_ENDS];
    uint32_t end_count;
    uint32_t kind;
    uint32_t usable;
    uint32_t think_open_id;
    uint32_t think_close_id;
} aotx_wrap;

#ifdef __cplusplus
static_assert(sizeof(aotx_wrap) == 512u, "wrap table size");
extern "C" {
#else
_Static_assert(sizeof(aotx_wrap) == 512u, "wrap table size");
#endif

struct aotx_modelfile;
struct aotx_manifest_entry;
int aotx_wrap_read(const struct aotx_modelfile *file,
                   const struct aotx_manifest_entry *entry, aotx_wrap *wrap);
int aotx_wrap_valid(const aotx_wrap *wrap);
/* Compare known-kind spans with the fixed table. Explicit manifest spans have no fixed form. */
int aotx_wrap_matches(const aotx_wrap *wrap);
void aotx_wrap_print(const char *name, const aotx_wrap *wrap);

#ifdef __cplusplus
}
#endif
#endif
