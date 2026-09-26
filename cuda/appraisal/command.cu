/* Purpose: Set independent appraisal writes, recall and background work controls.
 * Owns: Console parsing and deferred operator requests; accepted settings are typed memory.
 * Launch shape: The ordered command parser processes each input line.
 * Lifetime: One runtime and its recorded operator input. */
#include "appraisal/appraisal.cuh"
#include "appraisal/schema.cuh"
#include "cli/cli.cuh"
#include "policy/state.cuh"

static __device__ bool aotx_appraisal_is(const unsigned char *p, uint32_t n, const char *word) {
    uint32_t i = 0; while (i < n && word[i] && p[i] == (unsigned char)word[i]) ++i;
    return i == n && !word[i];
}
static __device__ bool aotx_appraisal_numbers(const unsigned char *p, uint32_t n, uint32_t *v, uint32_t count) {
    uint32_t at = 0;
    for (uint32_t i = 0; i < count; ++i) {
        while (at < n && p[at] == ' ') ++at;
        uint32_t begin = at; v[i] = 0;
        while (at < n && p[at] >= '0' && p[at] <= '9') {
            uint32_t digit = p[at++] - '0';
            if (v[i] > (UINT32_MAX - digit) / 10) return false;
            v[i] = v[i] * 10 + digit;
        }
        if (at == begin || (at < n && p[at] != ' ')) return false;
    }
    while (at < n && p[at] == ' ') ++at;
    return at == n;
}
static __device__ void aotx_appraisal_status(aotx_cli_out *out) {
    aotx_appraisal_refresh();
    aotx_cli_say(out, "appraisal: flags "); aotx_cli_num(out, aotx_appraisal.write_flags);
    aotx_cli_say(out, " pending "); aotx_cli_num(out, aotx_appraisal.pending);
    aotx_cli_say(out, " active "); aotx_cli_num(out, aotx_appraisal.active);
    aotx_cli_say(out, " status "); aotx_cli_num(out, aotx_appraisal.last_status);
    aotx_console_write(out->text, out->at); aotx_cli_clear(out);
    aotx_cli_say(out, "appraisal work: calls "); aotx_cli_num(out, aotx_appraisal.calls);
    aotx_cli_say(out, " completed "); aotx_cli_num(out, aotx_appraisal.completed);
    aotx_cli_say(out, " refused "); aotx_cli_num(out, aotx_appraisal.refused);
    aotx_cli_say(out, " interrupted "); aotx_cli_num(out, aotx_appraisal.interrupted);
    if (aotx_appraisal.control_pending || aotx_appraisal.explicit_pending) {
        aotx_console_write(out->text, out->at); aotx_cli_clear(out);
        aotx_cli_say(out, "appraisal control: pending");
    }
}
__device__ void aotx_appraisal_command(const unsigned char *p, uint32_t n, aotx_cli_out *out) {
    while (n && *p == ' ') { ++p; --n; }
    while (n && p[n - 1] == ' ') --n;
    aotx_appraisal_refresh();
    if (!n || aotx_appraisal_is(p, n, "status")) { aotx_appraisal_status(out); return; }
    if (aotx_appraisal_is(p, n, "run")) {
        if (!aotx_seam.replaying) aotx_appraisal.explicit_pending = 1;
        aotx_appraisal_status(out); return;
    }
    unsigned char next[AOTX_APPRAISAL_CONFIG_BYTES] = {};
    if (aotx_appraisal.control_pending) for (uint32_t j = 0; j < sizeof(next); ++j) next[j] = aotx_appraisal.control[j];
    else if (aotx_appraisal.config < aotx_live_store.count) {
        const unsigned char *c = aotx_live_store.objects[aotx_appraisal.config];
        const unsigned char *before = aotx_live_store.payload + aotx_cog_u64(c + AOTX_CO_OFFSET);
        for (uint32_t j = 0; j < sizeof(next); ++j) next[j] = before[j];
    } else {
        for (uint32_t j = 0; j < 8; ++j) next[j] = "AOTXAPC1"[j];
        aotx_cog_put(next + 8, 1, 4); aotx_cog_put(next + 16, 160, 4);
        aotx_cog_put(next + 20, 512, 4); aotx_cog_put(next + 24, 16384, 4);
        aotx_cog_put(next + 28, 250000, 4); aotx_cog_put(next + 32, 250000, 4);
        aotx_cog_put(next + 36, AOTX_RECALL_BATCH, 4);
        for (uint32_t j = 0; j < 32; ++j) next[40 + j] = aotx_appraisal_processor[j];
    }
    uint32_t flags = aotx_cog_u32(next + 12), split = 0, values[4];
    while (split < n && p[split] != ' ') ++split;
    uint32_t start = split; while (start < n && p[start] == ' ') ++start;
    bool valid = true;
    if (aotx_appraisal_is(p, n, "on")) flags |= AOTX_APPRAISAL_WRITE | AOTX_APPRAISAL_RECALL;
    else if (aotx_appraisal_is(p, n, "off")) flags = 0;
    else if (aotx_appraisal_is(p, split, "writes") || aotx_appraisal_is(p, split, "recall") ||
        aotx_appraisal_is(p, split, "background")) {
        uint32_t bit = aotx_appraisal_is(p, split, "writes") ? AOTX_APPRAISAL_WRITE :
            aotx_appraisal_is(p, split, "recall") ? AOTX_APPRAISAL_RECALL : AOTX_APPRAISAL_BACKGROUND;
        if (aotx_appraisal_is(p + start, n - start, "on")) flags |= bit;
        else if (aotx_appraisal_is(p + start, n - start, "off")) flags &= ~bit;
        else valid = false;
    } else if (aotx_appraisal_is(p, split, "limits")) {
        valid = aotx_appraisal_numbers(p + start, n - start, values, 4);
        if (valid) {
            valid = values[0] && values[1] && values[1] <= AOTX_INTAKE_REPLY && values[2] &&
                values[3] && values[3] <= AOTX_RECALL_BATCH;
            if (valid) for (uint32_t j = 0; j < 4; ++j) aotx_cog_put(next + (j == 3 ? 36 : 16 + 4 * j), values[j], 4);
        }
    } else if (aotx_appraisal_is(p, split, "priority")) {
        valid = aotx_appraisal_numbers(p + start, n - start, values, 2);
        if (valid) {
            valid = values[0] <= AOTX_COG_SCALE && values[1] <= AOTX_COG_SCALE;
            if (valid) for (uint32_t j = 0; j < 2; ++j) aotx_cog_put(next + 28 + 4 * j, values[j], 4);
        }
    } else valid = false;
    if (!valid) {
        aotx_cli_say(out, "Use: appraisal [status|on|off|writes on/off|recall on/off|background on/off|run]");
        aotx_console_write(out->text, out->at); aotx_cli_clear(out);
        aotx_cli_say(out, "Use: appraisal limits PAGES TOKENS TICKS ROWS; appraisal priority FLOOR BOOST"); return;
    }
    if ((flags & AOTX_APPRAISAL_BACKGROUND) && aotx_policy.enabled &&
        aotx_policy.config.abi < AOTX_POLICY_APPRAISAL_ABI) {
        aotx_appraisal.last_status = AOTX_COG_LAYOUT; aotx_appraisal_status(out); return;
    }
    aotx_cog_put(next + 12, flags, 4);
    for (uint32_t j = 0; j < sizeof(next); ++j) aotx_appraisal.control[j] = next[j];
    aotx_appraisal.control_pending = 1; aotx_appraisal.last_status = 0;
    aotx_appraisal_status(out);
}
