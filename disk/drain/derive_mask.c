/* Purpose: Turn the derived-file name list into its bit mask.
 * Owns: Nothing; the caller holds the mask.
 * Threading: One thread at drain start.
 * Lifetime: One command-line parse. */
#include "disk/drain/derive.h"

#include <string.h>

int aotx_derive_mask(const char *list, unsigned *out)
{
    static const char *names[8] = { "console", "note", "bus", "bulk", "sequence",
                                    "requests", "transcript", "tokens" };
    const char *at = list;
    unsigned mask = 0u;
    if (strcmp(list, "none") == 0) {
        *out = 0u;
        return 0;
    }
    while (*at != '\0') {
        size_t length = strcspn(at, ",");
        int found = 0;
        for (int i = 0; i < 8; ++i) {
            if (strlen(names[i]) == length && memcmp(names[i], at, length) == 0) {
                mask |= 1u << i;
                found = 1;
            }
        }
        if (found == 0) {
            return -1;
        }
        at += length;
        if (*at == ',') {
            at++;
        }
    }
    *out = mask;
    return 0;
}
