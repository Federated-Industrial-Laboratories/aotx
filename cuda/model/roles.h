/* Purpose: Name the role of a model file and read the role list of a run.
 * Owns: Nothing; the caller holds the list and the names.
 * Threading: One thread; the caller makes the calls one at a time.
 * Lifetime: Each call site. */
#ifndef AOTX_MODEL_ROLES_H
#define AOTX_MODEL_ROLES_H

#include <string.h>

#include "model/model.cuh"
#include "profile/profile.cuh"

/* The roles a run loads when the caller names none: the two small models and the language
 * file the profile names. The other language file is not in the list, because a run which
 * does not ask for it must not pay for its bytes. The weights region of a profile holds
 * the file that profile names, and not both files. */
#define AOTX_ROLES_DEFAULT "embedding,reranker," AOTX_PROFILE_LANGUAGE

/* The name of each role, in the order of the role numbers. */
static const char *aotx_role_name[AOTX_MODEL_ROLES] = {
    "embedding", "reranker", "language", "language-q4"
};

/* The role of a name, or the role count when the name is not a role. */
static inline unsigned int aotx_role_of(const char *name)
{
    for (unsigned int i = 0u; i < AOTX_MODEL_ROLES; ++i) {
        if (strcmp(name, aotx_role_name[i]) == 0) {
            return i;
        }
    }
    return AOTX_MODEL_ROLES;
}

/* Report whether a list with commas between the names holds a name. A null list is the
 * default list. */
static inline int aotx_role_wanted(const char *list, const char *name)
{
    const char *walk = (list == 0) ? AOTX_ROLES_DEFAULT : list;
    size_t bytes = strlen(name);
    while (*walk != '\0') {
        const char *end = strchr(walk, ',');
        size_t piece = (end == 0) ? strlen(walk) : (size_t)(end - walk);
        if (piece == bytes && memcmp(walk, name, bytes) == 0) {
            return 1;
        }
        if (end == 0) {
            break;
        }
        walk = end + 1;
    }
    return 0;
}

/* Give the first name of a list that is not a role. The name goes in out and the return is
 * 1. A list whose every name is a role gives 0. An empty name is not a role. */
static inline int aotx_role_unknown(const char *list, char *out, unsigned int max)
{
    const char *walk = (list == 0) ? AOTX_ROLES_DEFAULT : list;
    for (;;) {
        const char *end = strchr(walk, ',');
        size_t piece = (end == 0) ? strlen(walk) : (size_t)(end - walk);
        char one[64];
        if (piece != 0u && piece < sizeof one) {
            memcpy(one, walk, piece);
            one[piece] = '\0';
            if (aotx_role_of(one) < AOTX_MODEL_ROLES) {
                if (end == 0) {
                    return 0;
                }
                walk = end + 1;
                continue;
            }
        }
        if (piece >= max) {
            piece = max - 1u;
        }
        if (piece == 0u) {
            memcpy(out, "(empty)", 8);
            return 1;
        }
        memcpy(out, walk, piece);
        out[piece] = '\0';
        return 1;
    }
}

/* The names of a list that are roles. The count comes back, and each role goes in out. */
static inline unsigned int aotx_role_list(const char *list, unsigned int *out)
{
    unsigned int held = 0u;
    for (unsigned int i = 0u; i < AOTX_MODEL_ROLES; ++i) {
        if (aotx_role_wanted(list, aotx_role_name[i]) != 0) {
            out[held++] = i;
        }
    }
    return held;
}

#endif
