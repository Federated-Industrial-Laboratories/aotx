/* Purpose: Change or stop the decode of one agent from a command line.
 * Owns: Nothing; the sampler and sequence tables hold the changes.
 * Launch shape: One thread; the command parser calls this for one agent.
 * Lifetime: One agent command. */
#ifndef AOTX_CLI_AGENT_DECODE_CUH
#define AOTX_CLI_AGENT_DECODE_CUH

static __device__ __noinline__ void aotx_cli_agent_decode(aotx_cli_out *out,
                                                          unsigned int agent,
                                                          aotx_cli_word action,
                                                          const unsigned char *text,
                                                          unsigned int length,
                                                          unsigned int *position)
{
    if (aotx_cli_is(action, "stop")) {
        unsigned int state = aotx_seqs.slot[agent].state;
        if (state != AOTX_SEQ_STATE_PREFILL && state != AOTX_SEQ_STATE_DECODE) {
            aotx_cli_say(out, "agent: no reply runs");
            aotx_cli_count.refused += 1u;
        } else {
            aotx_seq_stop(agent);
            aotx_cli_say(out, "agent: the reply stops");
        }
        aotx_cli_console(out);
        return;
    }
    aotx_cli_word value = aotx_cli_take(text, length, position);
    unsigned int result = aotx_sampler_set(agent, (const char *)action.at, action.length,
                                           (const char *)value.at, value.length);
    if (result == AOTX_SAMPLER_TOOK) {
        aotx_cli_say(out, "agent: ");
        aotx_cli_add(out, (const char *)action.at, action.length);
        aotx_cli_say(out, " ");
        aotx_cli_add(out, (const char *)value.at, value.length);
        aotx_cli_say(out, " changes at the next turn");
    } else if (result == AOTX_SAMPLER_UNKNOWN) {
        aotx_cli_say(out, "agent: the decode key is not known");
        aotx_cli_count.refused += 1u;
    } else if (result == AOTX_SAMPLER_VALUE) {
        aotx_cli_say(out, "agent: the decode value is not a number");
        aotx_cli_count.refused += 1u;
    } else {
        aotx_cli_say(out, "agent: the decode value is outside its range");
        aotx_cli_count.refused += 1u;
    }
    aotx_cli_console(out);
}

#endif
