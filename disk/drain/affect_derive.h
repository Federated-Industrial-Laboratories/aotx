/* Purpose: Connect the optional streams to the drain derivation state.
 * Owns: Nothing; the stream modules own their descriptors.
 * Threading: One thread; records are taken in journal order.
 * Lifetime: One drain run. */
#ifndef AOTX_DRAIN_AFFECT_DERIVE_H
#define AOTX_DRAIN_AFFECT_DERIVE_H

#include "disk/drain/derive.h"

int aotx_affect_derive_open(aotx_derive *state, const char *boot_dir, unsigned int mask);
int aotx_affect_derive_record(aotx_derive *state, const aotx_record_header *header);
int aotx_affect_derive_sync(aotx_derive *state);
void aotx_affect_derive_close(aotx_derive *state);

#endif
