/* Purpose: Turn the derived-file name list into its bit mask.
 * Owns: Nothing; the caller holds the mask.
 * Threading: One thread at drain start.
 * Lifetime: One command-line parse. */
#include "disk/drain/derive.h"

#include <string.h>

int aotx_derive_mask(const char *list, unsigned *out)
{
#ifdef AOTX_AFFECT
    static const char *names[11] = { "console", "note", "bus", "bulk", "sequence",
                                     "requests", "transcript", "tokens", "pages",
                                     "affect", "quality" };
    const int name_count = 11;
#else
    static const char *names[9] = { "console", "note", "bus", "bulk", "sequence",
                                    "requests", "transcript", "tokens", "pages" };
    const int name_count = 9;
#endif
    const char *at = list;
    unsigned mask = 0u;
    if (strcmp(list, "none") == 0) {
        *out = 0u;
        return 0;
    }
    while (*at != '\0') {
        size_t length = strcspn(at, ",");
        int found = 0;
        for (int i = 0; i < name_count; ++i) {
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
