/* Purpose: Check the rule that states the memory a profile needs and the profile table.
 * Owns: The counts of the cases.
 * Launch shape: Host only; the check opens no context and needs no card.
 * Lifetime: One run of the test program. */
#include <stdio.h>
#include <string.h>

#include "model/roles.h"
#include "model/kinds.h"
#include "profile/fit.h"

static unsigned int aotx_profile_test_applied;
static unsigned int aotx_profile_test_failed;

static void aotx_profile_test_check(int ok, const char *what)
{
    aotx_profile_test_applied += 1u;
    if (!ok) {
        aotx_profile_test_failed += 1u;
        printf("profile: FAILED %s\n", what);
    }
}

/* The need of a row, from the figures of that row. */
static unsigned long long aotx_profile_test_need(const aotx_profile_row *row)
{
    return aotx_profile_need(row->weights_bytes, row->ring_slots, row->pages_each);
}

/* The row of the build states the same three figures as the header of the build. */
static void aotx_profile_test_case_row(void)
{
    const aotx_profile_row *row = aotx_profile_row_of(AOTX_PROFILE_NAME);
    aotx_profile_test_check(row != 0, "the table holds a row for the profile of the build");
    if (row == 0) {
        return;
    }
    aotx_profile_test_check(row->weights_bytes == (unsigned long long)AOTX_MEM_WEIGHTS_BYTES,
                            "the row states the weights region of the header");
    aotx_profile_test_check(row->ring_slots == (unsigned long long)AOTX_DEVICE_RING_SLOTS,
                            "the row states the ring slots of the header");
    aotx_profile_test_check(row->pages_each == (unsigned long long)AOTX_KV_PAGES_EACH,
                            "the row states the pages of a slot of the header");
    printf("profile: %s needs %llu MB, arch sm_%d, slots %u\n", AOTX_PROFILE_NAME,
           aotx_profile_test_need(row) >> 20, (int)AOTX_ARCH, (unsigned int)AOTX_SLOTS);
}

/* The four profiles need more memory in the order of the table, and each figure is the
 * figure the rule gives. */
static void aotx_profile_test_case_order(void)
{
    unsigned long long before = 0ull;
    for (unsigned int i = 0u; i < AOTX_PROFILE_ROWS; ++i) {
        unsigned long long need = aotx_profile_test_need(&aotx_profile_table[i]);
        printf("profile: %-4s needs %llu MB\n", aotx_profile_table[i].name, need >> 20);
        aotx_profile_test_check(need > before,
                                "each profile of the table needs more than the one before");
        before = need;
    }
}

/* The rule takes a card whose free memory is over the need and refuses one that is under
 * it by one byte. The check runs at every profile of the table. */
static void aotx_profile_test_case_edges(void)
{
    for (unsigned int i = 0u; i < AOTX_PROFILE_ROWS; ++i) {
        const aotx_profile_row *row = &aotx_profile_table[i];
        unsigned long long need = aotx_profile_test_need(row);
        unsigned long long total = need + (need >> 2);
        const char *fits = aotx_profile_that_fits(need, total);
        const char *under = aotx_profile_that_fits(need - 1ull, total);
        aotx_profile_test_check(fits != 0 && strcmp(fits, row->name) == 0,
                                "a card with the need free holds that profile");
        aotx_profile_test_check(under == 0 || strcmp(under, row->name) != 0,
                                "a card one byte under the need does not hold it");
    }
}

/* The rule of the build refuses a card that has less free memory than the need, and takes
 * one that has the need. A card whose whole memory is under the weights region never holds
 * the profile, whatever its free memory reports. */
static void aotx_profile_test_case_build(void)
{
    unsigned long long need = 0ull;
    unsigned long long large = 1024ull * 1024ull * 1024ull * 1024ull;
    int took = aotx_profile_fits(large, large, &need);
    aotx_profile_test_check(took == 1 && need > 0ull,
                            "a card with a terabyte free holds the profile of the build");
    aotx_profile_test_check(aotx_profile_fits(need, large, 0) == 1,
                            "a card with the need free holds the profile of the build");
    aotx_profile_test_check(aotx_profile_fits(need - 1ull, large, 0) == 0,
                            "a card one byte under the need does not hold it");
    aotx_profile_test_check(
        aotx_profile_fits(large, (unsigned long long)AOTX_MEM_WEIGHTS_BYTES - 1ull, 0) == 0,
        "a card whose memory is under the weights region does not hold the profile");
}

/* Every figure of the profile keeps the shape the kernels need. */
static void aotx_profile_test_case_shape(void)
{
    aotx_profile_test_check(AOTX_SLOTS >= 32u && AOTX_SLOTS <= 256u,
                            "the slot count is from 32 to 256");
    aotx_profile_test_check((AOTX_SLOTS & (AOTX_SLOTS - 1u)) == 0u,
                            "the slot count is a power of two");
    aotx_profile_test_check(
        (AOTX_DEVICE_RING_SLOTS & (AOTX_DEVICE_RING_SLOTS - 1ull)) == 0ull
        && (AOTX_INBOUND_SLOTS & (AOTX_INBOUND_SLOTS - 1ull)) == 0ull,
        "the two ring slot counts are powers of two");
    aotx_profile_test_check(
        (AOTX_HOST_RING_DATA_BYTES & (AOTX_HOST_RING_DATA_BYTES - 1ull)) == 0ull
        && (AOTX_BULK_RING_DATA_BYTES & (AOTX_BULK_RING_DATA_BYTES - 1ull)) == 0ull,
        "the two host ring sizes are powers of two");
    aotx_profile_test_check(AOTX_PROFILE_NAME[0] != '\0'
                            && AOTX_PROFILE_LANGUAGE[0] != '\0',
                            "the profile names itself and its language file");
    aotx_profile_test_check(AOTX_MODULE_SLOTS == AOTX_SLOTS,
                            "the module slots follow the slot count");
    aotx_profile_test_check(AOTX_SKILL_BYTES > 0u && AOTX_CATALOGUE_BYTES > 0ull
                            && AOTX_MODELS_RESIDENT > 0u,
                            "the figures of the later steps carry values");
    aotx_profile_test_check(aotx_role_of(AOTX_PROFILE_LANGUAGE)
                            == (unsigned int)AOTX_PROFILE_LANGUAGE_ROLE,
                            "the language role of the profile is the role of its name");
    printf("profile: the language file is %s, role %u\n", AOTX_PROFILE_LANGUAGE,
           (unsigned int)AOTX_PROFILE_LANGUAGE_ROLE);
}

typedef struct aotx_profile_tensor_set {
    const char *absent;
} aotx_profile_tensor_set;

static int aotx_profile_tensor_present(void *context, const char *name)
{
    const aotx_profile_tensor_set *set = (const aotx_profile_tensor_set *)context;
    return set->absent == NULL || strcmp(set->absent, name) != 0;
}

/* Every kind has complete rows, every tensor name fits, and every loaded layer has a kind. */
static void aotx_profile_test_case_layer_kinds(void)
{
    char name[AOTX_DESC_BUFFER];
    for (unsigned int k = 0u; k < AOTX_LAYER_KIND_COUNT; ++k) {
        const aotx_layer_kind *kind = &aotx_layer_kind_table[k];
        aotx_profile_test_check(kind->name[0] != '\0' && kind->tensors != 0u
                                && kind->state != AOTX_LAYER_STATE_NONE
                                && kind->capture != NULL && kind->keys != 0u,
                                "a layer kind row holds all fields");
        for (unsigned int i = 0u; i < kind->tensors; ++i) {
            aotx_profile_test_check(
                aotx_layer_name(name, sizeof name, AOTX_MODEL_MAX_LAYERS - 1u,
                                &kind->tensor[i]) == 0,
                "the name builder builds a tensor name from the kind table");
        }
        for (unsigned int i = 0u; i < kind->keys; ++i) {
            aotx_profile_test_check(kind->key[i].name[0] != '\0'
                                    && kind->key[i].member < sizeof(aotx_model_desc),
                                    "a layer metadata key names a descriptor member");
        }
    }

    aotx_model_desc desc;
    memset(&desc, 0, sizeof desc);
    memset(desc.kind, AOTX_LAYER_KIND_INVALID, sizeof desc.kind);
    desc.layers = AOTX_MODEL_MAX_LAYERS;
    aotx_layer_desc_fill(&desc, AOTX_LAYER_KIND_ATTENTION);
    aotx_profile_test_check(aotx_layer_desc_valid(&desc) != 0,
                            "every layer of a descriptor has a kind");
    desc.kind[AOTX_MODEL_MAX_LAYERS / 2u] = AOTX_LAYER_KIND_INVALID;
    aotx_profile_test_check(aotx_layer_desc_valid(&desc) == 0,
                            "a descriptor with a missing layer kind is refused");

    desc.layers = 1u;
    aotx_layer_desc_fill(&desc, AOTX_LAYER_KIND_ATTENTION);
    aotx_layer_name(name, sizeof name, 0u, &aotx_layer_attention_tensor[0]);
    aotx_profile_tensor_set set = { name };
    char missing[AOTX_DESC_BUFFER];
    aotx_profile_test_check(
        aotx_layer_required(&desc, aotx_profile_tensor_present, &set,
                            missing, sizeof missing) != 0
        && strcmp(missing, name) == 0,
        "a missing required tensor is refused with its name");
}

int main(void)
{
    aotx_profile_test_case_row();
    aotx_profile_test_case_order();
    aotx_profile_test_case_edges();
    aotx_profile_test_case_build();
    aotx_profile_test_case_shape();
    aotx_profile_test_case_layer_kinds();
    printf("profile: %u cases applied, %u passed, %u failed\n", aotx_profile_test_applied,
           aotx_profile_test_applied - aotx_profile_test_failed, aotx_profile_test_failed);
    if (aotx_profile_test_applied == 0u) {
        printf("profile: no case ran\n");
        return 1;
    }
    return (aotx_profile_test_failed == 0u) ? 0 : 1;
}
