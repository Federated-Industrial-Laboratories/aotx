/* Purpose: Hold the settings the device reads and apply a change to one of them.
 * Owns: The settings table and the address of the control page.
 * Launch shape: One thread; the apply step and the parser call these in slot order.
 * Lifetime: The whole run.
 *
 * The table holds one row for every number setting of keys.h. The row of a key is the
 * place of that key in the list, and no map exists beside the list. The apply admits a
 * setting the device reads. It refuses every other one with a reason. */
#ifndef AOTX_SETTINGS_CUH
#define AOTX_SETTINGS_CUH

#include "seam/wire.h"
#include "settings/keys.h"

/* What an apply gives back. */
#define AOTX_SETTING_TOOK      0u   /* the row holds the value */
#define AOTX_SETTING_UNKNOWN   1u   /* no key of the list has that name */
#define AOTX_SETTING_NOT_HERE  2u   /* the key is read at boot or by the terminal program */
#define AOTX_SETTING_RANGE     3u   /* the value is outside the range of the key */

/* One row: the value in the scaled unit of the key, and the tick of the last change. */
typedef struct aotx_setting_row {
    long long          value;
    unsigned long long changed;   /* the tick of the last change, or zero */
} aotx_setting_row;

/* A set line waits here until the tick commit node writes its record. The command runs
 * inside the apply of the inbound records. The apply holds the state hash in its own hand
 * until it ends, so a record the command wrote there would fold into nothing. The
 * commit writes the records of a tick after every applied line and every token. That is
 * the order the journal holds them in, so a replay folds them in the same order. */
#define AOTX_SETTING_PENDING_MAX 64u
typedef struct aotx_setting_pending {
    long long    value;
    unsigned int scale;
    unsigned int key_len;
    char         key[AOTX_SETTING_WIRE_KEY_BYTES];
} aotx_setting_pending;

typedef struct aotx_settings_state {
    aotx_setting_row row[AOTX_SETTING_NUMBER_COUNT];
    unsigned int     applied;     /* settings the table took since start */
    unsigned int     refused;     /* settings the table refused since start */
    unsigned int     pending_count;
    unsigned long long affect_revision;
    aotx_setting_pending pending[AOTX_SETTING_PENDING_MAX];
} aotx_settings_state;

extern __device__ aotx_settings_state aotx_setting_table;

/* The page the pump reads. The device writes it in the tick commit node with a release
 * store, and the pump acquire-loads it once a tick. The glue reads numbers and parses
 * nothing. */
typedef struct aotx_settings_page {
    unsigned long long period_ns;   /* the pace of the pump */
    unsigned long long budget_ns;   /* the decode budget of one tick */
    unsigned long long mirror_hz;   /* frames a second of the thread that runs the raster */
} aotx_settings_page;

extern __device__ aotx_settings_page *aotx_settings_control;

/* Identify the settings exposed by the affect operator resource. */
__device__ __forceinline__ bool aotx_setting_affect(unsigned index)
{
    switch (index) {
#define AOTX_AFFECT_KEY(symbol, ...) case symbol: return true;
    AOTX_SETTING_AFFECT_NUMBERS(AOTX_AFFECT_KEY)
#undef AOTX_AFFECT_KEY
    default: return false;
    }
}

/* The name of one number setting. */
__device__ __forceinline__ const char *aotx_setting_name(unsigned int index)
{
    switch (index) {
#define AOTX_SETTING_ONE(symbol, name, side, effect, ...) case symbol: return name;
    AOTX_SETTING_NUMBERS(AOTX_SETTING_ONE)
#undef AOTX_SETTING_ONE
    default: return "-";
    }
}

/* The side that reads one number setting. */
__device__ __forceinline__ unsigned int aotx_setting_side(unsigned int index)
{
    switch (index) {
#define AOTX_SETTING_ONE(symbol, name, side, effect, ...) \
    case symbol: return AOTX_SETTING_SIDE_##side;
    AOTX_SETTING_NUMBERS(AOTX_SETTING_ONE)
#undef AOTX_SETTING_ONE
    default: return 0u;
    }
}

/* When a change of one number setting takes effect. */
__device__ __forceinline__ unsigned int aotx_setting_effect(unsigned int index)
{
    switch (index) {
#define AOTX_SETTING_ONE(symbol, name, side, effect, ...) \
    case symbol: return AOTX_SETTING_AT_##effect;
    AOTX_SETTING_NUMBERS(AOTX_SETTING_ONE)
#undef AOTX_SETTING_ONE
    default: return 0u;
    }
}

/* The word of an effect, which the settings command prints. */
__device__ __forceinline__ const char *aotx_setting_effect_name(unsigned int effect)
{
    switch (effect) {
    case AOTX_SETTING_AT_BOOT:     return "boot";
    case AOTX_SETTING_AT_TICK:     return "tick";
    case AOTX_SETTING_AT_SEQUENCE: return "sequence";
    case AOTX_SETTING_AT_TASK:     return "task";
    case AOTX_SETTING_AT_REQUEST:  return "request";
    case AOTX_SETTING_AT_FRAME:    return "frame";
    default:                       return "read";
    }
}

__device__ __forceinline__ long long aotx_setting_default(unsigned int index)
{
    switch (index) {
#define AOTX_SETTING_ONE(symbol, name, side, effect, value, ...) \
    case symbol: return (long long)(value);
    AOTX_SETTING_NUMBERS(AOTX_SETTING_ONE)
#undef AOTX_SETTING_ONE
    default: return 0;
    }
}

__device__ __forceinline__ long long aotx_setting_least(unsigned int index)
{
    switch (index) {
#define AOTX_SETTING_ONE(symbol, name, side, effect, value, least, ...) \
    case symbol: return (long long)(least);
    AOTX_SETTING_NUMBERS(AOTX_SETTING_ONE)
#undef AOTX_SETTING_ONE
    default: return 0;
    }
}

__device__ __forceinline__ long long aotx_setting_most(unsigned int index)
{
    switch (index) {
#define AOTX_SETTING_ONE(symbol, name, side, effect, value, least, most, ...) \
    case symbol: return (long long)(most);
    AOTX_SETTING_NUMBERS(AOTX_SETTING_ONE)
#undef AOTX_SETTING_ONE
    default: return 0;
    }
}

__device__ __forceinline__ int aotx_setting_scale(unsigned int index)
{
    switch (index) {
#define AOTX_SETTING_ONE(symbol, name, side, effect, value, least, most, scale) \
    case symbol: return (scale);
    AOTX_SETTING_NUMBERS(AOTX_SETTING_ONE)
#undef AOTX_SETTING_ONE
    default: return AOTX_SETTING_SCALE_ONE;
    }
}

/* The value of one row, in the scaled unit of the key. */
__device__ __forceinline__ long long aotx_setting_value(unsigned int index)
{
    return (index < (unsigned int)AOTX_SETTING_NUMBER_COUNT)
         ? aotx_setting_table.row[index].value : 0;
}

/* The value of one row as a whole number. A key of the fixed scale gives its whole part. */
__device__ __forceinline__ unsigned int aotx_setting_count(unsigned int index)
{
    long long value = aotx_setting_value(index);
    long long scale = (long long)aotx_setting_scale(index);
    if (value < 0) {
        return 0u;
    }
    return (unsigned int)(value / ((scale > 0) ? scale : 1));
}

/* The value of one row as a fraction. */
__device__ __forceinline__ float aotx_setting_fraction(unsigned int index)
{
    long long scale = (long long)aotx_setting_scale(index);
    return (float)aotx_setting_value(index) / (float)((scale > 0) ? scale : 1);
}

/* Report whether a key text of a given length is the name. The compare runs in place, so
 * the table holds no array of names of its own. */
__device__ __forceinline__ int aotx_setting_is(const char *name, const char *key,
                                               unsigned int length)
{
    unsigned int at = 0u;
    while (at < length && name[at] != '\0' && name[at] == key[at]) {
        at += 1u;
    }
    return (at == length && name[at] == '\0') ? 1 : 0;
}

/* Find the row of a key text. The return is 1 when the name is one of the list. */
__device__ __forceinline__ int aotx_setting_find(const char *key, unsigned int length,
                                                 unsigned int *index)
{
#define AOTX_SETTING_ONE(symbol, name, side, effect, ...) \
    if (aotx_setting_is(name, key, length)) { *index = (unsigned int)symbol; return 1; }
    AOTX_SETTING_NUMBERS(AOTX_SETTING_ONE)
#undef AOTX_SETTING_ONE
    return 0;
}

/* Fill every row with the default of its key. The call runs once before the first tick. */
__device__ void aotx_settings_reset(void);

/* Judge a key and a value with no change to the table. The result is one of the four
 * results above, and index names the row of a key the list holds. */
__device__ unsigned int aotx_settings_judge(const char *key, unsigned int key_len,
                                            long long value, unsigned int scale,
                                            unsigned int *index);

/* Apply one setting record. The return is one of the four results above. A result that is
 * not AOTX_SETTING_TOOK leaves the table as it was, and writes one console line and one
 * bus note. */
__device__ unsigned int aotx_settings_apply(const aotx_setting_body *body,
                                            unsigned long long tick);

/* Write the record of every set line the tick holds, fold each into the state hash and
 * give the table its value. The tick commit node calls this on one thread, before the
 * statistics record. */
__device__ void aotx_settings_commit(unsigned long long tick);

/* Write the values of the control page with a release store. The tick commit node calls
 * this. */
__device__ void aotx_settings_publish(void);

/* Fill the table with the default of every key and publish the control page. */
__global__ void aotx_settings_boot(void);

/* Open and close the control page. The glue calls these. */
int aotx_settings_page_open(void);
void aotx_settings_page_close(void);

/* Read the values of the control page with an acquire load. A run with no page gives the
 * default of the key. */
unsigned long long aotx_settings_period_ns(void);
unsigned long long aotx_settings_budget_ns(void);
unsigned long long aotx_settings_mirror_hz(void);

/* The default of one number setting, in the scaled unit of the key. A host program reads
 * it, so a check states no figure of its own. */
long long aotx_settings_default(unsigned int index);

#endif
