/* Purpose: Write the record that states the card and the build.
 * Owns: Nothing; the record ring holds the result.
 * Launch shape: One block of one thread.
 * Lifetime: One node after the boot record.
 *
 * The profile name, the architecture and the slot count come from the profile header of
 * the build. A report from another card therefore states them. */
#include "profile/card.cuh"
#include "seam/seam.cuh"

__global__ void aotx_profile_note_card(aotx_card_read card)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    aotx_card_body body;
    for (unsigned int i = 0u; i < (unsigned int)AOTX_CARD_NAME_BYTES; ++i) {
        body.name[i] = card.name[i];
    }
    body.name[AOTX_CARD_NAME_BYTES - 1u] = '\0';
    body.memory_total = card.memory_total;
    body.memory_free = card.memory_free;
    body.compute_major = card.compute_major;
    body.compute_minor = card.compute_minor;
    const char *profile = AOTX_PROFILE_NAME;
    unsigned int at = 0u;
    while (at + 1u < (unsigned int)AOTX_CARD_PROFILE_BYTES && profile[at] != '\0') {
        body.profile[at] = profile[at];
        at += 1u;
    }
    while (at < (unsigned int)AOTX_CARD_PROFILE_BYTES) {
        body.profile[at] = '\0';
        at += 1u;
    }
    body.arch = (unsigned int)AOTX_ARCH;
    body.slots = (unsigned int)AOTX_SLOTS;
    aotx_seam_write(AOTX_WRITER_SYSTEM, AOTX_CLASS_B, AOTX_REC_CARD, 0u,
                    &body, (unsigned int)sizeof body);
}
