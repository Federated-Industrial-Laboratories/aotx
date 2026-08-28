/* Purpose: Check the agents panel over the sequence table and the decode count of the tick.
 * Owns: The sequence fixtures of the panel check.
 * Threading: One thread; the panel check calls these one at a time.
 * Lifetime: The program.
 *
 * The file is a part of the panel check. It reads the helpers of that check, so it comes
 * after them in the same translation unit. */
#ifndef AOTX_TEST_UI_AGENTS_H
#define AOTX_TEST_UI_AGENTS_H

/* The sample fixture. The period is 17 ms, which the pump never paces to. The step is the
 * reply tokens one sample adds for slot zero. */
#define AOTX_TEST_PERIOD    17000000ull
#define AOTX_TEST_BASE_NS   4000000000ull
#define AOTX_TEST_STEP      3u

/* Fill sequence slots with content that differs from slot to slot, so a row on the wrong
 * slot cannot pass. The stride leaves free slots between the ones that are taken. The row of
 * a sequence is then not its slot number, and a panel that reads one for the other fails. */
__global__ void aotx_test_sequences(unsigned int count, unsigned int stride,
                                    unsigned long long tick)
{
    unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= AOTX_SEQ_SLOTS) {
        return;
    }
    aotx_seq *seq = &aotx_seqs.slot[slot];
    if (stride == 0u || slot % stride != 0u || slot / stride >= count) {
        seq->state = AOTX_SEQ_STATE_FREE;
        for (unsigned int i = 0u; i < AOTX_SAY_WINDOW; ++i) {
            aotx_say_window[slot][i].tick = 0ull;
        }
        return;
    }
    seq->state = AOTX_SEQ_STATE_PREFILL + (slot % 3u);
    seq->role = slot % AOTX_MODEL_ROLES;
    seq->held = 100u + slot;
    seq->sampled = AOTX_TEST_STEP * (slot + 1u) * (AOTX_SAY_WINDOW - 1u);
    seq->opened = tick - (unsigned long long)(10u * (slot + 1u));

    /* The window the rate reads. The samples carry a period of AOTX_TEST_PERIOD ns. The
     * pump never paces to that period, so a rate taken from the pace cannot pass. */
    for (unsigned int i = 0u; i < AOTX_SAY_WINDOW; ++i) {
        aotx_say_sample *at = &aotx_say_window[slot][(tick - (AOTX_SAY_WINDOW - 1u) + i)
                                                     % AOTX_SAY_WINDOW];
        at->tick = tick - (AOTX_SAY_WINDOW - 1u) + i;
        at->ns = AOTX_TEST_BASE_NS + (unsigned long long)i * AOTX_TEST_PERIOD;
        at->opened = seq->opened;
        at->sampled = AOTX_TEST_STEP * (slot + 1u) * i;
    }
    if (slot == 0u) {
        aotx_seqs.live = count;
    }
}

static const char *aotx_test_role_of(unsigned int role)
{
    static const char *names[AOTX_MODEL_ROLES] = {
        "embedding", "reranker", "language", "language-q4"
    };
    return (role < AOTX_MODEL_ROLES) ? names[role] : "-";
}

static const char *aotx_test_state_of(unsigned int state)
{
    switch (state) {
    case AOTX_SEQ_STATE_FREE:    return "free";
    case AOTX_SEQ_STATE_PREFILL: return "prefill";
    case AOTX_SEQ_STATE_DECODE:  return "decode";
    case AOTX_SEQ_STATE_DONE:    return "done";
    default:                     return "-";
    }
}

/* Build the row the panel must hold for one slot, from the same fields the fixture wrote.
 * The rate is the tokens of the window over the time of the window. Both come from the two
 * samples the fixture wrote, so no part of the figure comes from the pace of the pump. */
static void aotx_test_agent_row(unsigned int slot, char *out, size_t max)
{
    unsigned int sampled = AOTX_TEST_STEP * (slot + 1u) * (AOTX_SAY_WINDOW - 1u);
    unsigned long long span = (unsigned long long)(AOTX_SAY_WINDOW - 1u) * AOTX_TEST_PERIOD;
    unsigned long long rate = (unsigned long long)sampled * 1000000000ull / span;
    snprintf(out, max, "%u %s %s %u %u %llu", slot,
             aotx_test_role_of(slot % AOTX_MODEL_ROLES),
             aotx_test_state_of(AOTX_SEQ_STATE_PREFILL + (slot % 3u)), 100u + slot, sampled,
             rate);
}

/* The agents panel holds one row for each sequence slot that is not free. The check runs at
 * one slot and at 64. A table with more sequences than the panel holds keeps its last row
 * for the count that is left. */
static void aotx_test_agents_panel(unsigned int count, unsigned int stride)
{
    const aotx_ui_panel *panel = &aotx_test_panels[AOTX_UI_AGENTS];
    unsigned long long tick = 4096ull;
    unsigned int rows = (unsigned int)panel->rows - 2u;
    unsigned int shown = (count > rows) ? (rows - 1u) : count;
    unsigned int matched = 0u;
    char want[80];

    aotx_check_runtime(cudaMemcpyToSymbol(aotx_time_tick, &tick, sizeof tick),
                       "cudaMemcpyToSymbol");
    aotx_test_sequences<<<1, AOTX_SEQ_SLOTS>>>(count, stride, tick);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_ui_agents<<<1, AOTX_UI_PANEL_THREADS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_read_grid();

    for (unsigned int rank = 0u; rank < shown; ++rank) {
        unsigned int slot = rank * stride;
        aotx_test_agent_row(slot, want, sizeof want);
        if (aotx_test_row_says(AOTX_UI_AGENTS, rank + 2u, 1u, want)) {
            matched += 1u;
        } else {
            printf("ui: the agents row %u is not '%s'\n", rank + 2u, want);
        }
    }
    aotx_test_check(matched == shown, "every row of the agents panel holds its own slot");
    aotx_test_check(aotx_test_row_says(AOTX_UI_AGENTS, 1u, 1u,
                                       "slot role state position reply rate"),
                    "the agents panel names its columns");
    if (count > rows) {
        snprintf(want, sizeof want, "and %u more", count - shown);
        aotx_test_check(aotx_test_row_says(AOTX_UI_AGENTS, rows + 1u, 1u, want),
                        "the last row states the sequences the panel could not hold");
    } else {
        aotx_test_check(aotx_test_row_blank(AOTX_UI_AGENTS, rows + 1u),
                        "a table that fits leaves the last row of the panel empty");
    }

    /* The tick panel states how many sequences decode. */
    aotx_ui_tick<<<1, AOTX_UI_PANEL_THREADS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_read_grid();
    snprintf(want, sizeof want, "decode %u", count);
    aotx_test_check(aotx_test_row_says(AOTX_UI_TICK, 10u, 1u, want),
                    "the tick panel states the sequences that decode");
    printf("ui: %u sequences at a stride of %u gave %u rows of the agents panel\n",
           count, stride, matched);

    /* Give the table back empty, so a later case sees the panel with no sequence. */
    aotx_test_sequences<<<1, AOTX_SEQ_SLOTS>>>(0u, stride, tick);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

#endif
