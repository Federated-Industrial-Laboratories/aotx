/* Purpose: Read the card, refuse a profile it cannot hold, and record what ran.
 * Owns: Nothing; the record ring holds the result.
 * Launch shape: Host glue only; one one-thread kernel writes the record.
 * Lifetime: The start of the run, after the boot record. */
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

#include "boot/boot.cuh"
#include "boot/check.h"
#include "profile/card.cuh"
#include "profile/fit.h"

/* Megabytes of a byte count, for the lines this file writes. */
static unsigned long long aotx_boot_mb(unsigned long long bytes)
{
    return bytes >> 20;
}

void aotx_boot_version(void)
{
    printf("aotx %s profile %s arch sm_%d slots %u\n", AOTX_VERSION, AOTX_PROFILE_NAME,
           (int)AOTX_ARCH, (unsigned int)AOTX_SLOTS);
}

/* What the driver gave. The check keeps it, and the record kernel takes it after the boot
 * record of the run. */
static aotx_card_read aotx_boot_card_hold;
static int aotx_boot_card_read = 0;

int aotx_boot_card_check(void)
{
    cudaDeviceProp props;
    int device = 0;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    memset(&props, 0, sizeof props);
    aotx_check_runtime(cudaGetDevice(&device), "cudaGetDevice");
    aotx_check_runtime(cudaGetDeviceProperties(&props, device), "cudaGetDeviceProperties");
    aotx_check_driver(cuMemGetInfo(&free_bytes, &total_bytes), "cuMemGetInfo");

    aotx_card_read card;
    memset(&card, 0, sizeof card);
    for (unsigned int i = 0u; i + 1u < (unsigned int)AOTX_CARD_NAME_BYTES; ++i) {
        card.name[i] = props.name[i];
        if (props.name[i] == '\0') {
            break;
        }
    }
    card.name[AOTX_CARD_NAME_BYTES - 1u] = '\0';
    card.memory_total = (unsigned long long)total_bytes;
    card.memory_free = (unsigned long long)free_bytes;
    card.compute_major = (unsigned int)props.major;
    card.compute_minor = (unsigned int)props.minor;

    /* The check comes before any placement, so a run that cannot hold its tables stops
     * with the figures and names the profile that fits. */
    unsigned long long need = 0ull;
    if (aotx_profile_fits(card.memory_free, card.memory_total, &need) == 0) {
        const char *fits = aotx_profile_that_fits(card.memory_free, card.memory_total);
        fprintf(stderr, "profile %s needs %llu MB; %llu MB free\n", AOTX_PROFILE_NAME,
                aotx_boot_mb(need), aotx_boot_mb(card.memory_free));
        if (fits != 0) {
            fprintf(stderr, "the profile %s fits this card\n", fits);
        } else {
            fprintf(stderr, "no profile of this build fits this card\n");
        }
        return 2;
    }
    printf("card: %s %llu MB total %llu MB free capability %u.%u\n", card.name,
           aotx_boot_mb(card.memory_total), aotx_boot_mb(card.memory_free),
           card.compute_major, card.compute_minor);
    printf("build: profile %s arch sm_%d slots %u needs %llu MB\n", AOTX_PROFILE_NAME,
           (int)AOTX_ARCH, (unsigned int)AOTX_SLOTS, aotx_boot_mb(need));
    aotx_boot_card_hold = card;
    aotx_boot_card_read = 1;
    return 0;
}

void aotx_boot_card_note(void)
{
    if (aotx_boot_card_read == 0) {
        return;
    }
    aotx_profile_note_card<<<1, 1>>>(aotx_boot_card_hold);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}
