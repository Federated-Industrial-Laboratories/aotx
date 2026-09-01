/* Purpose: Connect the optional streams to the drain derivation state.
 * Owns: Nothing; the stream modules own their descriptors.
 * Threading: One thread; records are taken in journal order.
 * Lifetime: One drain run. */
#include "disk/drain/affect_derive.h"
#include "disk/drain/affect_stream.h"
#include "disk/drain/quality_stream.h"

int aotx_affect_derive_open(aotx_derive *state, const char *boot_dir, unsigned int mask)
{
    if ((mask & AOTX_DERIVE_AFFECT) != 0
        && aotx_affect_stream_open(&state->affect_stream, boot_dir) != 0) return -1;
    if ((mask & AOTX_DERIVE_QUALITY) != 0
        && aotx_quality_stream_open(&state->quality_stream, boot_dir) != 0) return -1;
    return 0;
}

int aotx_affect_derive_record(aotx_derive *state, const aotx_record_header *header)
{
    if (header->type == AOTX_REC_AFFECT_TRACE
        && (state->mask & AOTX_DERIVE_AFFECT) != 0) {
        return aotx_affect_stream_record(state->affect_stream, header) == 0 ? 1 : -1;
    }
    if (header->type == AOTX_REC_QUALITY
        && (state->mask & AOTX_DERIVE_QUALITY) != 0) {
        return aotx_quality_stream_record(state->quality_stream, header) == 0 ? 1 : -1;
    }
    return 0;
}

int aotx_affect_derive_sync(aotx_derive *state)
{
    if (aotx_affect_stream_sync(state->affect_stream) != 0) return -1;
    return aotx_quality_stream_sync(state->quality_stream);
}

void aotx_affect_derive_close(aotx_derive *state)
{
    aotx_affect_stream_close(state->affect_stream);
    state->affect_stream = NULL;
    aotx_quality_stream_close(state->quality_stream);
    state->quality_stream = NULL;
}
