/* Purpose: Carry what the glue reads of the card to the kernel that records it.
 * Owns: Nothing; the caller holds the values it passes.
 * Launch shape: One block of one thread.
 * Lifetime: One call after the boot record. */
#ifndef AOTX_PROFILE_CARD_CUH
#define AOTX_PROFILE_CARD_CUH

#include "profile/profile.cuh"
#include "seam/wire.h"

/* What the driver gives about the card. The glue reads these values and passes them; it
 * writes no record and formats no text. The profile, the architecture and the slot count
 * of the build come from the device side, so the record states the build that ran. */
typedef struct aotx_card_read {
    char               name[AOTX_CARD_NAME_BYTES];
    unsigned long long memory_total;
    unsigned long long memory_free;
    unsigned int       compute_major;
    unsigned int       compute_minor;
} aotx_card_read;

/* Write one CARD record. The record follows the boot record of the run. */
__global__ void aotx_profile_note_card(aotx_card_read card);

#endif
