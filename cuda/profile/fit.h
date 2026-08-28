/* Purpose: State the device memory a profile needs and whether a card holds it.
 * Owns: Nothing; the caller holds the figures it passes.
 * Threading: One thread; the function holds no state between calls.
 * Lifetime: The call.
 *
 * The header is plain C with no CUDA symbol, so a host program checks the rule with no
 * card. The boot glue calls it before any placement. */
#ifndef AOTX_PROFILE_FIT_H
#define AOTX_PROFILE_FIT_H

#include "profile/profile.cuh"

/* The need of a profile has five parts. The list below names each one.
 *
 * The first part is the weights region. The region is virtual. Physical memory goes behind
 * a part of it when a tensor of that part arrives. The profile holds its models, so the
 * whole region counts.
 *
 * The second part is the record ring: AOTX_DEVICE_RING_SLOTS slots of
 * AOTX_PROFILE_SLOT_BYTES. The third part is the scratch arena with the three guard gaps
 * of the reservation.
 *
 * The fourth part is the display. It holds the pixel buffer of the device, the buffer
 * object of the window, the cell grid and the font. A grid of 160 by 50 cells of 8 by 16
 * pixels gives a pixel buffer of 4 MB. A figure of 16 MB holds every part with room to
 * spare.
 *
 * The fifth part is the key value pages of AOTX_PROFILE_KV_SLOTS sequences at their full
 * page count. The key value range is virtual and is larger than any card. The need
 * therefore counts the sequences a run holds at a full context at one time. */
#define AOTX_PROFILE_SLOT_BYTES    256ull
#define AOTX_PROFILE_PAGE_BYTES    (2ull * 1024ull * 1024ull)
#define AOTX_PROFILE_SCRATCH_BYTES (64ull * 1024ull * 1024ull)
#define AOTX_PROFILE_GUARD_BYTES   (3ull * 2ull * 1024ull * 1024ull)
#define AOTX_PROFILE_UI_BYTES      (16ull * 1024ull * 1024ull)
#define AOTX_PROFILE_KV_SLOTS      4ull

/* The bytes one profile needs, from the three figures that change with the profile. */
static inline unsigned long long aotx_profile_need(unsigned long long weights_bytes,
                                                   unsigned long long ring_slots,
                                                   unsigned long long pages_each)
{
    return weights_bytes
         + ring_slots * AOTX_PROFILE_SLOT_BYTES
         + AOTX_PROFILE_SCRATCH_BYTES
         + AOTX_PROFILE_GUARD_BYTES
         + AOTX_PROFILE_UI_BYTES
         + AOTX_PROFILE_KV_SLOTS * pages_each * AOTX_PROFILE_PAGE_BYTES;
}

/* One row for each profile, in order of the memory it needs. A refusal names the profile
 * that fits, and a build holds one profile header. The rows therefore repeat the three
 * figures of the four headers. A check compares the row of the build with the figures of
 * its header, so a row that drifts is a failure. */
typedef struct aotx_profile_row {
    const char        *name;
    unsigned long long weights_bytes;
    unsigned long long ring_slots;
    unsigned long long pages_each;
} aotx_profile_row;

#define AOTX_PROFILE_ROWS 4u

static const aotx_profile_row aotx_profile_table[AOTX_PROFILE_ROWS] = {
    { "8g",  5ull * 1024ull * 1024ull * 1024ull,  32768ull,  96ull },
    { "12g", 8ull * 1024ull * 1024ull * 1024ull,  65536ull, 160ull },
    { "24g", 16ull * 1024ull * 1024ull * 1024ull, 131072ull, 320ull },
    { "48g", 40ull * 1024ull * 1024ull * 1024ull, 262144ull, 640ull }
};

/* Report whether the card holds the profile of the build, and give the bytes it needs.
 * The card must hold the need in its free memory. Its whole memory must also hold the
 * weights region, because a card that is smaller than that region never holds the
 * profile. The return is 1 when the profile fits and 0 when it does not. */
static inline int aotx_profile_fits(unsigned long long free_bytes,
                                    unsigned long long total_bytes,
                                    unsigned long long *need_bytes)
{
    unsigned long long need = aotx_profile_need((unsigned long long)AOTX_MEM_WEIGHTS_BYTES,
                                                (unsigned long long)AOTX_DEVICE_RING_SLOTS,
                                                (unsigned long long)AOTX_KV_PAGES_EACH);
    if (need_bytes != 0) {
        *need_bytes = need;
    }
    if ((unsigned long long)AOTX_MEM_WEIGHTS_BYTES > total_bytes) {
        return 0;
    }
    return (need <= free_bytes) ? 1 : 0;
}

/* Give the name of the largest profile the card holds, or a null pointer when no profile
 * holds. The rows are in order of the memory they need, so the last row that fits is the
 * largest one. */
static inline const char *aotx_profile_that_fits(unsigned long long free_bytes,
                                                 unsigned long long total_bytes)
{
    const char *found = 0;
    for (unsigned int i = 0u; i < AOTX_PROFILE_ROWS; ++i) {
        const aotx_profile_row *row = &aotx_profile_table[i];
        unsigned long long need = aotx_profile_need(row->weights_bytes, row->ring_slots,
                                                    row->pages_each);
        if (row->weights_bytes <= total_bytes && need <= free_bytes) {
            found = row->name;
        }
    }
    return found;
}

/* Give the row of the profile of the build, or a null pointer. */
static inline const aotx_profile_row *aotx_profile_row_of(const char *name)
{
    for (unsigned int i = 0u; i < AOTX_PROFILE_ROWS; ++i) {
        const char *at = aotx_profile_table[i].name;
        unsigned int b = 0u;
        while (at[b] != '\0' && name[b] != '\0' && at[b] == name[b]) {
            b += 1u;
        }
        if (at[b] == '\0' && name[b] == '\0') {
            return &aotx_profile_table[i];
        }
    }
    return 0;
}

#endif
