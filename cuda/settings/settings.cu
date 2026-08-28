/* Purpose: Hold the settings table, apply a change, and publish the two pump values.
 * Owns: The settings table, the line the refusals write and the control page address.
 * Launch shape: One thread; the apply step and the parser call these in slot order.
 * Lifetime: The whole run. */
#include "bus/bus.cuh"
#include "settings/console.cuh"

/* The table holds the default of every key from the load of the module. A caller that
 * reads a row before the first tick therefore reads a value and not a zero. */
#define AOTX_SETTING_ROW(symbol, name, side, effect, value, ...) { (long long)(value), 0ull },
__device__ aotx_settings_state aotx_setting_table = {
    { AOTX_SETTING_NUMBERS(AOTX_SETTING_ROW) }, 0u, 0u
};
#undef AOTX_SETTING_ROW
__device__ aotx_settings_page *aotx_settings_control = 0;

/* The line a refusal builds. One thread applies a setting, so one line is enough and no
 * frame of the kernel holds it. */
static __device__ aotx_cli_out aotx_setting_out;

__device__ void aotx_settings_reset(void)
{
    for (unsigned int i = 0u; i < (unsigned int)AOTX_SETTING_NUMBER_COUNT; ++i) {
        aotx_setting_table.row[i].value = aotx_setting_default(i);
        aotx_setting_table.row[i].changed = 0ull;
    }
    aotx_setting_table.applied = 0u;
    aotx_setting_table.refused = 0u;
}

/* Add a value in the scaled unit of its key. A key of the fixed scale gives a point and
 * the decimals that carry data; a trailing zero is dropped. */
static __device__ void aotx_setting_add_value(aotx_cli_out *out, long long value,
                                             int scale)
{
    if (value < 0) {
        aotx_cli_say(out, "-");
        value = -value;
    }
    if (scale <= AOTX_SETTING_SCALE_ONE) {
        aotx_cli_num(out, (unsigned long long)value);
        return;
    }
    aotx_cli_num(out, (unsigned long long)(value / (long long)scale));
    unsigned int fraction = (unsigned int)(value % (long long)scale);
    if (fraction == 0u) {
        return;
    }
    char digits[8];
    unsigned int at = 0u;
    unsigned int place = (unsigned int)scale / 10u;
    while (place > 0u) {
        digits[at] = (char)('0' + (char)((fraction / place) % 10u));
        at += 1u;
        place /= 10u;
    }
    while (at > 1u && digits[at - 1u] == '0') {
        at -= 1u;
    }
    aotx_cli_say(out, ".");
    aotx_cli_add(out, digits, at);
}

/* Write one line to the console and put the same text on the bus as a note. Every refusal
 * of a setting takes this path, so the operator and the bus hold the same reason. */
static __device__ void aotx_settings_refuse(aotx_cli_out *out, unsigned long long tick)
{
    aotx_console_write(out->text, out->at);
    aotx_bus_append(AOTX_WRITER_SYSTEM, AOTX_BUS_NOTE, 0u, out->text, out->at,
                    0ull, 0ull, 0.0f, tick);
    aotx_setting_table.refused += 1u;
    aotx_cli_clear(out);
}

/* Judge a key and a value with no change to the table. The result is one of the four
 * results of settings.cuh, and index names the row of a key the list holds. */
__device__ unsigned int aotx_settings_judge(const char *key, unsigned int key_len,
                                            long long value, unsigned int scale,
                                            unsigned int *index)
{
    if (aotx_setting_find(key, key_len, index) == 0) {
        return AOTX_SETTING_UNKNOWN;
    }
    if (aotx_setting_side(*index) != AOTX_SETTING_SIDE_DEVICE) {
        return AOTX_SETTING_NOT_HERE;
    }
    /* A record carries the scale that made the value. A record of another scale states a
     * value the range of the key does not measure, so the table refuses it. */
    if ((int)scale != aotx_setting_scale(*index)
        || value < aotx_setting_least(*index) || value > aotx_setting_most(*index)) {
        return AOTX_SETTING_RANGE;
    }
    return AOTX_SETTING_TOOK;
}

/* Write the reason of a result into a line. Every refusal of a setting takes this wording,
 * whether a record or a command line gave the value. */
static __device__ void aotx_settings_reason(aotx_cli_out *out, unsigned int result,
                                            unsigned int index, const char *key,
                                            unsigned int key_len)
{
    if (result == AOTX_SETTING_UNKNOWN) {
        aotx_cli_say(out, "set: the key ");
        aotx_cli_add(out, key, key_len);
        aotx_cli_say(out, " is not known");
        return;
    }
    if (result == AOTX_SETTING_NOT_HERE) {
        aotx_cli_say(out, "set: the key ");
        aotx_cli_say(out, aotx_setting_name(index));
        aotx_cli_say(out, " is not a setting the device reads");
        return;
    }
    aotx_cli_say(out, "set: ");
    aotx_cli_say(out, aotx_setting_name(index));
    aotx_cli_say(out, " takes ");
    aotx_setting_add_value(out, aotx_setting_least(index), aotx_setting_scale(index));
    aotx_cli_say(out, " to ");
    aotx_setting_add_value(out, aotx_setting_most(index), aotx_setting_scale(index));
}

__device__ unsigned int aotx_settings_apply(const aotx_setting_body *body,
                                            unsigned long long tick)
{
    aotx_cli_out *out = &aotx_setting_out;
    unsigned int key_len = body->key_len;
    unsigned int index = 0u;
    if (key_len > AOTX_SETTING_WIRE_KEY_BYTES) {
        key_len = AOTX_SETTING_WIRE_KEY_BYTES;
    }
    unsigned int result = aotx_settings_judge(body->key, key_len, body->value, body->scale,
                                              &index);
    if (result != AOTX_SETTING_TOOK) {
        aotx_cli_clear(out);
        aotx_settings_reason(out, result, index, body->key, key_len);
        aotx_settings_refuse(out, tick);
        return result;
    }
    aotx_setting_table.row[index].value = body->value;
    aotx_setting_table.row[index].changed = tick;
    aotx_setting_table.applied += 1u;
    return AOTX_SETTING_TOOK;
}

__device__ void aotx_settings_publish(void)
{
    aotx_settings_page *page = aotx_settings_control;
    if (page == 0) {
        return;
    }
    /* The budget goes first and the period follows it with a release store. A reader that
     * takes the period with an acquire load therefore sees the budget beside it. */
    page->budget_ns = (unsigned long long)aotx_setting_count(AOTX_SET_DECODE_BUDGET_MS)
                    * 1000000ull;
    aotx_seam_release_sys(&page->period_ns,
                          (unsigned long long)aotx_setting_count(AOTX_SET_TICK_PERIOD_MS)
                          * 1000000ull);
}

/* Read a text as a value in the scaled unit of a key. The text is a whole number, or a
 * number with a point and up to four decimals. The return is 1 when every byte belongs to
 * the number and the whole part stays under a million million. */
static __device__ int aotx_setting_number_of(const char *text, unsigned int length,
                                             int scale, long long *value)
{
    unsigned int at = 0u;
    int minus = 0;
    if (length > 0u && text[0] == '-') {
        minus = 1;
        at = 1u;
    }
    if (at >= length) {
        return 0;
    }
    long long whole = 0;
    unsigned int digits = 0u;
    while (at < length && text[at] >= '0' && text[at] <= '9') {
        whole = whole * 10 + (long long)(text[at] - '0');
        if (whole > 1000000000000ll) {
            return 0;
        }
        digits += 1u;
        at += 1u;
    }
    if (digits == 0u) {
        return 0;
    }
    long long fraction = 0;
    if (at < length && text[at] == '.') {
        if (scale <= AOTX_SETTING_SCALE_ONE) {
            return 0;
        }
        at += 1u;
        unsigned int taken = 0u;
        long long place = (long long)scale / 10;
        while (at < length && text[at] >= '0' && text[at] <= '9') {
            if (taken >= 4u || place <= 0) {
                return 0;
            }
            fraction += (long long)(text[at] - '0') * place;
            place /= 10;
            taken += 1u;
            at += 1u;
        }
        if (taken == 0u) {
            return 0;
        }
    }
    if (at != length) {
        return 0;
    }
    long long got = whole * (long long)scale + fraction;
    *value = minus ? -got : got;
    return 1;
}

__device__ void aotx_settings_set_command(aotx_cli_out *out, const char *key,
                                          unsigned int key_len, const char *value,
                                          unsigned int value_len, unsigned long long tick)
{
    /* A replay of the journal sends every line again. The setting record of a set line
     * stands in the journal beside that line, and the apply of the record makes the
     * change. This command therefore writes no record and changes nothing while a replay
     * runs, so a restored run folds every setting body one time. */
    if (aotx_seam.replaying != 0ull) {
        return;
    }
    if (key_len == 0u || value_len == 0u) {
        aotx_cli_say(out, "set: give a key and a value");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        return;
    }
    if (key_len >= AOTX_SETTING_WIRE_KEY_BYTES) {
        aotx_cli_say(out, "set: the key is too long");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        return;
    }
    unsigned int index = 0u;
    if (aotx_setting_find(key, key_len, &index) == 0) {
        aotx_settings_reason(out, AOTX_SETTING_UNKNOWN, index, key, key_len);
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        aotx_setting_table.refused += 1u;
        return;
    }
    int scale = aotx_setting_scale(index);
    long long got = 0;
    if (aotx_setting_number_of(value, value_len, scale, &got) == 0) {
        aotx_cli_say(out, "set: the value is not a number of ");
        aotx_cli_say(out, aotx_setting_name(index));
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        return;
    }
    /* The judgement comes before the record, so a line the table refuses writes no record
     * and the journal holds the changes that landed. */
    unsigned int result = aotx_settings_judge(key, key_len, got, (unsigned int)scale,
                                              &index);
    if (result != AOTX_SETTING_TOOK) {
        aotx_settings_reason(out, result, index, key, key_len);
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        aotx_setting_table.refused += 1u;
        return;
    }

    /* The body is built in the slot of the ring and not in a frame of this kernel. The
     * record is class A, so the state hash folds it and a restore applies it again. */
    if (!aotx_cli_allow()) {
        return;
    }
    unsigned long long seq = aotx_seam_claim(1u);
    aotx_record_header *header = aotx_seam_slot(seq);
    aotx_setting_body *body = (aotx_setting_body *)aotx_seam_body(header);
    body->value = got;
    body->scale = (unsigned int)scale;
    body->key_len = key_len;
    for (unsigned int i = 0u; i < AOTX_SETTING_WIRE_KEY_BYTES; ++i) {
        body->key[i] = (i < key_len) ? key[i] : '\0';
    }
    aotx_seam_publish(header, seq, AOTX_WRITER_CONSOLE, AOTX_CLASS_A, AOTX_REC_SETTING,
                      0u, (unsigned int)sizeof *body);
    aotx_seam.apply.state_hash = aotx_seam_fnv1a(aotx_seam.apply.state_hash,
                                                 aotx_seam_body_of(seq),
                                                 (unsigned int)sizeof *body);
    aotx_seam.apply.applied_count += 1ull;

    /* The apply of the record gives the value to the table. The live run and the replay
     * therefore take one path. */
    aotx_settings_apply(body, tick);
    if (!aotx_cli_allow()) {
        return;
    }
    aotx_cli_say(out, "set: ");
    aotx_cli_say(out, aotx_setting_name(index));
    aotx_cli_say(out, " ");
    aotx_setting_add_value(out, got, scale);
    aotx_cli_say(out, " ");
    aotx_cli_say(out, aotx_setting_effect_name(aotx_setting_effect(index)));
    aotx_console_write(out->text, out->at);
    aotx_cli_clear(out);
}

__device__ void aotx_settings_show_command(aotx_cli_out *out)
{
    aotx_cli_say(out, "settings: key value effect");
    aotx_cli_console(out);
    for (unsigned int i = 0u; i < (unsigned int)AOTX_SETTING_NUMBER_COUNT; ++i) {
        if (aotx_setting_side(i) != AOTX_SETTING_SIDE_DEVICE) {
            continue;
        }
        aotx_cli_say(out, "  ");
        aotx_cli_say(out, aotx_setting_name(i));
        aotx_cli_say(out, " ");
        aotx_setting_add_value(out, aotx_setting_table.row[i].value, aotx_setting_scale(i));
        aotx_cli_say(out, " ");
        aotx_cli_say(out, aotx_setting_effect_name(aotx_setting_effect(i)));
        aotx_cli_console(out);
    }
}

/* Fill the table with the default of every key and publish the two pump values. The glue
 * launches this once, before the first tick. */
__global__ void aotx_settings_boot(void)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    aotx_settings_reset();
    aotx_settings_publish();
}
