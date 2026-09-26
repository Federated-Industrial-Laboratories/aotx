/* Purpose: Expose policy state and recorded operator controls through the console.
 * Owns: Pause and stop flags; admitted decision state remains recoverable.
 * Launch shape: The ordered command parser processes each input line.
 * Lifetime: One selected runtime policy. */
#include "policy/control.cuh"
#include "reflection/state.cuh"
#include "cli/cli.cuh"
#include "cognitive/checkpoint.cuh"
#include "appraisal/appraisal.cuh"

static __device__ bool aotx_policy_is(const unsigned char *p, uint32_t n, const char *text) {
    uint32_t j = 0;
    while (j < n && text[j] && p[j] == (unsigned char)text[j]) ++j;
    return j == n && !text[j];
}
__device__ void aotx_policy_command(const unsigned char *p, uint32_t n, aotx_cli_out *out) {
    while (n && *p == ' ') { ++p; --n; }
    while (n && p[n - 1] == ' ') --n;
    if (!aotx_policy.enabled) { aotx_cli_say(out, "policy: off"); return; }
    uint32_t action = aotx_policy_is(p, n, "pause") ? AOTX_POLICY_PAUSE :
        aotx_policy_is(p, n, "resume") ? AOTX_POLICY_RESUME : aotx_policy_is(p, n, "stop") ? AOTX_POLICY_STOP :
        aotx_policy_is(p, n, "review on") ? AOTX_POLICY_REVIEW_ON :
        aotx_policy_is(p, n, "review off") ? AOTX_POLICY_REVIEW_OFF : 0;
    if (action) {
        uint32_t status = aotx_policy_control_check(action, aotx_review.control_revision);
        if (status) { aotx_cli_say(out, "policy: control refused status "); aotx_cli_num(out, status); return; }
        aotx_policy_control_apply(action);
    } else if (n && !aotx_policy_is(p, n, "status")) {
        aotx_cli_say(out, "Use: policy [status|pause|resume|stop|review on|review off]"); return;
    }
    aotx_cli_say(out, "policy: ");
    aotx_cli_say(out, aotx_policy.fatal ? "error" : aotx_policy.stopped ? "stopped" :
        aotx_policy.paused ? "paused" : aotx_policy.pending ? "recording" : "quiet");
    aotx_cli_say(out, " mode "); aotx_cli_num(out, aotx_policy.config.mode);
    aotx_cli_say(out, " decision "); aotx_cli_num(out, aotx_policy.decision);
    aotx_cli_say(out, " memory source "); aotx_cli_num(out, aotx_policy.source);
    aotx_cli_say(out, " state bytes "); aotx_cli_num(out, aotx_policy.config.state_bytes);
    aotx_cli_say(out, " status "); aotx_cli_num(out, aotx_policy.status);
    aotx_console_write(out->text, out->at); aotx_cli_clear(out);
    aotx_cli_say(out, "policy: calls "); aotx_cli_num(out, aotx_policy.calls);
    aotx_cli_say(out, " last ns "); aotx_cli_num(out, aotx_policy.elapsed_ns);
    aotx_cli_say(out, " maximum ns "); aotx_cli_num(out, aotx_policy.maximum_ns);
    aotx_cli_say(out, " state hash "); aotx_cli_num(out, aotx_policy.state_hash);
    aotx_cli_say(out, " saved generation "); aotx_cli_num(out, aotx_checkpoint.generation);
    if (aotx_policy.config.abi >= AOTX_POLICY_APPRAISAL_ABI) {
        aotx_console_write(out->text, out->at); aotx_cli_clear(out);
        aotx_cli_say(out, "policy: appraisal pending "); aotx_cli_num(out, aotx_appraisal_pending());
        aotx_cli_say(out, " work revision "); aotx_cli_num(out, aotx_appraisal_revision());
        aotx_cli_say(out, " accepted revision "); aotx_cli_num(out, aotx_policy.work_revision);
    }
    if (aotx_policy.config.abi == AOTX_POLICY_REVIEW_ABI) {
        aotx_console_write(out->text, out->at); aotx_cli_clear(out);
        aotx_cli_say(out, "policy: review "); aotx_cli_say(out, aotx_review.enabled ? "on" : "off");
        aotx_cli_say(out, " active "); aotx_cli_num(out, aotx_review.active);
        aotx_cli_say(out, " pending "); aotx_cli_num(out, aotx_review.pending);
        aotx_cli_say(out, " completed "); aotx_cli_num(out, aotx_review.completed);
        aotx_cli_say(out, " interrupted "); aotx_cli_num(out, aotx_review.interrupted);
        aotx_cli_say(out, " status "); aotx_cli_num(out, aotx_review.status);
        aotx_cli_say(out, " control revision "); aotx_cli_num(out, aotx_review.control_revision);
        aotx_console_write(out->text, out->at); aotx_cli_clear(out);
        aotx_cli_say(out, "policy: review frontier "); aotx_cli_num(out, aotx_review.frontier);
        aotx_cli_say(out, " result bytes "); aotx_cli_num(out, aotx_review.active ? aotx_live.written : 0);
        aotx_cli_say(out, "/"); aotx_cli_num(out, aotx_review.active ? aotx_live.choice_bytes : 0);
        aotx_cli_say(out, " maximum ns "); aotx_cli_num(out, aotx_review.maximum_ns);
    }
}
