/* Purpose: Check the update law, the quantization, the state record, its fold and the apply.
 * Owns: Nothing; the test program owns the ring, the settings table and the state table.
 * Launch shape: The turn node at one block of one thread for each agent; the scripts at one
 *   thread for each agent.
 * Lifetime: One run of the test program. */
#ifndef AOTX_TESTS_AFFECT_LAW_H
#define AOTX_TESTS_AFFECT_LAW_H

#define AOTX_AFFECT_LAW_SLACK   1      /* Q1.15 steps a float result may differ from double */
#define AOTX_AFFECT_LAW_TURNS   4u
#define AOTX_AFFECT_LAW_DECAY   16u
#define AOTX_AFFECT_LAW_HALF    16384  /* 0.5 in Q1.15 */
#define AOTX_AFFECT_LAW_HIGH    29491  /* 0.9 in Q1.15 */

/* The figures of the law, as the host reference reads them. */
typedef struct aotx_affect_law_figures {
    double decay_fast, decay_slow, gain_fast, gain_slow, probe_gain;
    double cap[AOTX_AFFECT_STATE_AXES];
} aotx_affect_law_figures;

/* The settings table with the defaults, then one row with a given value. */
__global__ void aotx_affect_test_settings(unsigned int index, long long value)
{
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        aotx_settings_reset();
        if (index < (unsigned int)AOTX_SETTING_NUMBER_COUNT) {
            aotx_setting_table.row[index].value = value;
        }
    }
}

/* A turn with no event for every agent under the count: the gear states a plain end. */
__global__ void aotx_affect_test_quiet(unsigned int count)
{
    unsigned int a = threadIdx.x;
    if (a >= count) {
        return;
    }
    aotx_affect_sums *acc = &aotx_affect_acc[a];
    aotx_affect_sums clear = {};
    *acc = clear;
    aotx_agent_gear[a].last_token = 0u;
    aotx_agent_gear[a].limit_end = 0u;
    aotx_agent_gear[a].stopped = 0u;
    aotx_agent_gear[a].out_tokens = 10u;
    aotx_agents.agent[a].budget_left = 3u;
    aotx_agents.agent[a].turn = a + 1u;
    aotx_seqs.slot[a].think_tokens = 2u;
    acc->sampled = 4u;
    acc->logprob_sum = -4.0f;
    acc->flag = 1u;
    aotx_affect_end(a);
}

/* The apply of one scripted state record for each agent under the count. The thread of
 * the count applies a record whose agent is past the table, which changes nothing. */
__global__ void aotx_affect_test_apply(unsigned int count)
{
    unsigned int a = threadIdx.x;
    if (a > count || a >= AOTX_SLOTS) {
        return;
    }
    aotx_affect_body body;
    body.agent = (a == count) ? AOTX_SLOTS : a;
    body.turn = a + 9u;
    for (unsigned int j = 0u; j < AOTX_AFFECT_STATE_AXES; ++j) {
        body.fast[j] = (short)(100 * (int)a + 7 * (int)j - 300);
        body.slow[j] = (short)(-50 * (int)a + 3 * (int)j + 200);
    }
    body.scale = (unsigned short)(1000u + a);
    body.axes = 2u;
    body.reason = 1u << (a % 15u);
    body.flags = 0u;
    aotx_affect_apply(&body);
}

/* The open of a sequence for every agent under the count, with the setting as the table
 * holds it. */
__global__ void aotx_affect_test_open(unsigned int count)
{
    unsigned int a = threadIdx.x;
    if (a >= count) {
        return;
    }
    aotx_model_how how = {};
    aotx_affect_open(a, &how);
}

static void aotx_affect_law_state_get(aotx_affect_agent_state *state)
{
    aotx_check_runtime(cudaMemcpyFromSymbol(state, aotx_affect_state,
                                            AOTX_SLOTS * sizeof *state),
                       "cudaMemcpyFromSymbol");
}

static void aotx_affect_law_state_set(const aotx_affect_agent_state *state)
{
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_affect_state, state,
                                          AOTX_SLOTS * sizeof *state),
                       "cudaMemcpyToSymbol");
}

/* The same value in every slot of the state table. */
static void aotx_affect_law_preset(short fast, short slow, unsigned int sign_by_agent)
{
    static aotx_affect_agent_state state[AOTX_SLOTS];
    memset(state, 0, sizeof state);
    for (unsigned int a = 0u; a < AOTX_SLOTS; ++a) {
        int flip = (sign_by_agent != 0u && (a & 1u) != 0u) ? -1 : 1;
        for (unsigned int j = 0u; j < AOTX_AFFECT_DATA_AXES; ++j) {
            state[a].fast[j] = (short)(flip * fast);
            state[a].slow[j] = (short)(flip * slow);
        }
    }
    aotx_affect_law_state_set(state);
}

static void aotx_affect_law_settings(unsigned int index, long long value)
{
    aotx_affect_test_settings<<<1, 1>>>(index, value);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

/* A settings table with no figure and a zero state table. The law then moves nothing, so
 * a case that reads the trace alone sees a zero effective state. */
static void aotx_affect_law_none(void)
{
    static aotx_settings_state none;
    memset(&none, 0, sizeof none);
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_setting_table, &none, sizeof none),
                       "cudaMemcpyToSymbol");
    aotx_affect_law_preset(0, 0, 0u);
}

/* The defaults of the law from the settings list, so the reference states no figure. */
static void aotx_affect_law_defaults(aotx_affect_law_figures *f)
{
    f->decay_fast = (double)aotx_settings_default(AOTX_SET_AFFECT_DECAY_FAST) / 10000.0;
    f->decay_slow = (double)aotx_settings_default(AOTX_SET_AFFECT_DECAY_SLOW) / 10000.0;
    f->gain_fast = (double)aotx_settings_default(AOTX_SET_AFFECT_GAIN_FAST) / 10000.0;
    f->gain_slow = (double)aotx_settings_default(AOTX_SET_AFFECT_GAIN_SLOW) / 10000.0;
    f->probe_gain = (double)aotx_settings_default(AOTX_SET_AFFECT_PROBE_GAIN) / 10000.0;
    f->cap[0] = (double)aotx_settings_default(AOTX_SET_AFFECT_CAP_VALENCE) / 10000.0;
    f->cap[1] = (double)aotx_settings_default(AOTX_SET_AFFECT_CAP_AROUSAL) / 10000.0;
    f->cap[2] = 1.0;
    f->cap[3] = 1.0;
}

/* The drive of an event mask in double, from the table of the header. */
static void aotx_affect_law_drive(unsigned int mask, double *drive)
{
    for (unsigned int j = 0u; j < AOTX_AFFECT_STATE_AXES; ++j) {
        drive[j] = 0.0;
    }
#define AOTX_AFFECT_LAW_EVENT(bit, valence, arousal) \
    if ((mask & (1u << (bit))) != 0u) { drive[0] += (double)(valence); \
                                        drive[1] += (double)(arousal); }
    AOTX_AFFECT_EVENT_TABLE(AOTX_AFFECT_LAW_EVENT)
#undef AOTX_AFFECT_LAW_EVENT
}

static short aotx_affect_law_q15(double value)
{
    double scaled = rint(value * 32768.0);
    scaled = (scaled < -32768.0) ? -32768.0 : (scaled > 32767.0) ? 32767.0 : scaled;
    return (short)scaled;
}

/* One turn of the law in double from a quantized state. The result is quantized. */
static void aotx_affect_law_step(const aotx_affect_law_figures *f,
                                 const aotx_affect_agent_state *before, unsigned int mask,
                                 const double *probe, aotx_affect_agent_state *after,
                                 short *effective, unsigned int *capped)
{
    double drive[AOTX_AFFECT_STATE_AXES];
    aotx_affect_law_drive(mask, drive);
    *capped = 0u;
    for (unsigned int j = 0u; j < AOTX_AFFECT_STATE_AXES; ++j) {
        double e = drive[j] + f->probe_gain * probe[j];
        double fast = tanh(f->decay_fast * (double)before->fast[j] / 32768.0 + f->gain_fast * e);
        double slow = tanh(f->decay_slow * (double)before->slow[j] / 32768.0 + f->gain_slow * e);
        after->fast[j] = aotx_affect_law_q15(fast);
        after->slow[j] = aotx_affect_law_q15(slow);
        int bound = (int)rint(f->cap[j] * 32768.0);
        int sum = (int)after->fast[j] + (int)after->slow[j];
        if (sum > bound) { sum = bound; *capped = 1u; }
        if (sum < -bound) { sum = -bound; *capped = 1u; }
        effective[j] = (short)((sum > 32767) ? 32767 : sum);
    }
    after->scale = (unsigned short)AOTX_AFFECT_SCALE_ONE;
    after->axes = (unsigned short)AOTX_AFFECT_DATA_AXES;
}

static int aotx_affect_law_close(short got, short want)
{
    int gap = (int)got - (int)want;
    return gap <= AOTX_AFFECT_LAW_SLACK && gap >= -AOTX_AFFECT_LAW_SLACK;
}

/* The records of one launch of the node, by agent: the trace, the state record and the
 * sequence of the state record. A missing record leaves its slot at zero. */
typedef struct aotx_affect_law_records {
    aotx_affect_trace_body trace[AOTX_SLOTS];
    aotx_affect_body state[AOTX_SLOTS];
    aotx_record_header head[AOTX_SLOTS];
    unsigned int traces, states, others;
} aotx_affect_law_records;

static void aotx_affect_law_collect(aotx_affect_test_ring *ring, aotx_affect_law_records *r)
{
    memset(r, 0, sizeof *r);
    aotx_affect_test_ring_read(ring);
    for (unsigned int i = 0u; i < AOTX_AFFECT_TEST_SLOTS; ++i) {
        const aotx_record_header *header =
            (const aotx_record_header *)(ring->records + (size_t)i * AOTX_SLOT_BYTES);
        const unsigned char *body = (const unsigned char *)header + AOTX_HEADER_BYTES;
        if (header->magic != AOTX_WIRE_MAGIC) {
            continue;
        }
        if (header->type == AOTX_REC_AFFECT_TRACE) {
            const aotx_affect_trace_body *trace = (const aotx_affect_trace_body *)body;
            if (trace->agent < AOTX_SLOTS) memcpy(&r->trace[trace->agent], trace, sizeof *trace);
            r->traces += 1u;
        } else if (header->type == AOTX_REC_AFFECT) {
            const aotx_affect_body *state = (const aotx_affect_body *)body;
            if (state->agent < AOTX_SLOTS) {
                memcpy(&r->state[state->agent], state, sizeof *state);
                memcpy(&r->head[state->agent], header, sizeof *header);
            }
            r->states += 1u;
        } else {
            r->others += 1u;
        }
    }
}

/* The fold of the state records in sequence order, from the hash the ring opened with. */
static unsigned long long aotx_affect_law_fold(const aotx_affect_law_records *r)
{
    unsigned long long hash = 0ull;
    unsigned long long low = 0ull;
    for (unsigned int n = 0u; n < r->states; ++n) {
        unsigned long long best = 0ull;
        unsigned int at = AOTX_SLOTS;
        for (unsigned int a = 0u; a < AOTX_SLOTS; ++a) {
            unsigned long long seq = r->head[a].seq;
            if (seq > low && (at == AOTX_SLOTS || seq < best)) {
                best = seq;
                at = a;
            }
        }
        if (at == AOTX_SLOTS) {
            break;
        }
        const unsigned char *bytes = (const unsigned char *)&r->state[at];
        for (unsigned int b = 0u; b < (unsigned int)sizeof(aotx_affect_body); ++b) {
            hash ^= (unsigned long long)bytes[b];
            hash *= AOTX_FNV_PRIME;
        }
        low = best;
    }
    return hash;
}

static void aotx_affect_law_seam(aotx_seam_state *seam)
{
    aotx_check_runtime(cudaMemcpyFromSymbol(seam, aotx_seam, sizeof *seam),
                       "cudaMemcpyFromSymbol");
}

/* One scripted turn over count agents from the state the table holds. The check reads the
 * table, the two records of each flagged agent and the hash against the reference. */
static void aotx_affect_law_turn(aotx_affect_test_ring *ring, unsigned int count,
                                 unsigned int loaded, const aotx_affect_law_figures *f,
                                 const char *name)
{
    aotx_affect_agent_state before[AOTX_SLOTS], after[AOTX_SLOTS];
    aotx_affect_law_records r;
    aotx_seam_state seam;
    char label[112];
    unsigned int wanted = 0u, state_right = 0u, record_right = 0u, trace_right = 0u;
    unsigned int quiet_right = 0u;
    aotx_affect_law_state_get(before);
    aotx_affect_test_ring_open(ring, 0u);
    aotx_affect_test_clear();
    aotx_affect_test_script<<<1, AOTX_SLOTS>>>(count, 1u);
    aotx_affect_turn<<<1, AOTX_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_affect_law_state_get(after);
    aotx_affect_law_collect(ring, &r);
    aotx_affect_law_seam(&seam);
    for (unsigned int a = 0u; a < count; ++a) {
        aotx_affect_test_plan p;
        aotx_affect_agent_state want;
        short effective[AOTX_AFFECT_STATE_AXES];
        unsigned int capped;
        double probe[AOTX_AFFECT_STATE_AXES] = { 0.0, 0.0, 0.0, 0.0 };
        aotx_affect_test_script_of(a, count, &p);
        if (p.flag == 0u) {
            quiet_right += (memcmp(&before[a], &after[a], sizeof after[a]) == 0
                            && r.head[a].magic == 0u) ? 1u : 0u;
            continue;
        }
        wanted += 1u;
        unsigned int mask = aotx_affect_test_reason(&p);
        /* The valence row is loaded and not a monitor; the arousal row is a monitor. */
        if (loaded != 0u && f->probe_gain != 0.0) {
            probe[0] = (double)p.reply_sum[0] / (double)p.reply_rows;
        }
        aotx_affect_law_step(f, &before[a], mask, probe, &want, effective, &capped);
        unsigned int good = 1u;
        for (unsigned int j = 0u; j < AOTX_AFFECT_STATE_AXES; ++j) {
            good &= aotx_affect_law_close(after[a].fast[j], want.fast[j]) ? 1u : 0u;
            good &= aotx_affect_law_close(after[a].slow[j], want.slow[j]) ? 1u : 0u;
        }
        good &= (after[a].scale == AOTX_AFFECT_SCALE_ONE
                 && after[a].axes == AOTX_AFFECT_DATA_AXES) ? 1u : 0u;
        state_right += good;
        const aotx_affect_body *s = &r.state[a];
        const aotx_record_header *h = &r.head[a];
        unsigned int flags = (loaded != 0u ? AOTX_AFFECT_FLAG_PROBES : 0u)
                           | (capped != 0u ? AOTX_AFFECT_FLAG_CAP : 0u);
        unsigned int rec = (h->magic == AOTX_WIRE_MAGIC && h->cls == AOTX_CLASS_A
                            && h->type == AOTX_REC_AFFECT && h->body_len == sizeof *s
                            && h->tick == AOTX_AFFECT_TEST_TICK
                            && h->writer == AOTX_WRITER_AGENT_BASE + a
                            && s->agent == a && s->turn == p.turn && s->reason == mask
                            && s->flags == flags && s->scale == after[a].scale
                            && s->axes == after[a].axes) ? 1u : 0u;
        for (unsigned int j = 0u; j < AOTX_AFFECT_STATE_AXES; ++j) {
            rec &= (s->fast[j] == after[a].fast[j] && s->slow[j] == after[a].slow[j]) ? 1u : 0u;
        }
        record_right += rec;
        const aotx_affect_trace_body *t = &r.trace[a];
        unsigned int tr = (t->agent == a && t->flags == flags && t->reason == mask) ? 1u : 0u;
        for (unsigned int j = 0u; j < AOTX_AFFECT_STATE_AXES; ++j) {
            int sum = (int)after[a].fast[j] + (int)after[a].slow[j];
            int bound = (int)rint(f->cap[j] * 32768.0);
            sum = (sum > bound) ? bound : (sum < -bound) ? -bound : sum;
            sum = (sum > 32767) ? 32767 : sum;
            tr &= (t->effective[j] == (short)sum) ? 1u : 0u;
        }
        trace_right += tr;
    }
    snprintf(label, sizeof label, "%s: the state of %u agents follows the law", name, count);
    aotx_affect_note(label, state_right == wanted, "agents", (double)state_right,
                     (double)wanted);
    snprintf(label, sizeof label, "%s: one state record holds the table at %u", name, count);
    aotx_affect_note(label, record_right == wanted && r.states == wanted, "records",
                     (double)record_right, (double)wanted);
    snprintf(label, sizeof label, "%s: the trace carries the capped state at %u", name, count);
    aotx_affect_note(label, trace_right == wanted, "traces", (double)trace_right,
                     (double)wanted);
    snprintf(label, sizeof label, "%s: the hash folds the records in order at %u", name, count);
    aotx_affect_note(label, seam.apply.state_hash == aotx_affect_law_fold(&r)
                     && seam.apply.applied_count == (unsigned long long)wanted, "records",
                     (double)seam.apply.applied_count, (double)wanted);
    if (count != wanted) {
        snprintf(label, sizeof label, "%s: an agent with the flag 0 keeps its state at %u",
                 name, count);
        aotx_affect_note(label, quiet_right == count - wanted, "agents", (double)quiet_right,
                         (double)(count - wanted));
    }
}

/* Sixteen turns with no event from a high state: every part decays toward zero along the
 * reference, and the fast part reaches zero. */
static void aotx_affect_law_decay(aotx_affect_test_ring *ring, unsigned int count,
                                  const aotx_affect_law_figures *f)
{
    aotx_affect_agent_state before[AOTX_SLOTS], after[AOTX_SLOTS];
    char label[112];
    unsigned int right = 0u, falling = 0u, zero = 0u;
    aotx_affect_law_preset(AOTX_AFFECT_LAW_HIGH, AOTX_AFFECT_LAW_HIGH, 1u);
    for (unsigned int turn = 0u; turn < AOTX_AFFECT_LAW_DECAY; ++turn) {
        aotx_affect_law_state_get(before);
        aotx_affect_test_ring_open(ring, 0u);
        aotx_affect_test_quiet<<<1, AOTX_SLOTS>>>(count);
        aotx_affect_turn<<<1, AOTX_SLOTS>>>();
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_affect_law_state_get(after);
        for (unsigned int a = 0u; a < count; ++a) {
            aotx_affect_agent_state want;
            short effective[AOTX_AFFECT_STATE_AXES];
            unsigned int capped;
            double probe[AOTX_AFFECT_STATE_AXES] = { 0.0, 0.0, 0.0, 0.0 };
            aotx_affect_law_step(f, &before[a], 0u, probe, &want, effective, &capped);
            unsigned int good = 1u, down = 1u;
            for (unsigned int j = 0u; j < AOTX_AFFECT_STATE_AXES; ++j) {
                good &= aotx_affect_law_close(after[a].fast[j], want.fast[j]) ? 1u : 0u;
                good &= aotx_affect_law_close(after[a].slow[j], want.slow[j]) ? 1u : 0u;
                down &= (abs(after[a].fast[j]) <= abs(before[a].fast[j])
                         && abs(after[a].slow[j]) <= abs(before[a].slow[j])) ? 1u : 0u;
            }
            right += good;
            falling += down;
        }
    }
    for (unsigned int a = 0u; a < count; ++a) {
        zero += (after[a].fast[0] == 0 && after[a].fast[1] == 0) ? 1u : 0u;
    }
    snprintf(label, sizeof label, "decay: %u turns with no event follow the law at %u",
             AOTX_AFFECT_LAW_DECAY, count);
    aotx_affect_note(label, right == count * AOTX_AFFECT_LAW_DECAY, "turns", (double)right,
                     (double)(count * AOTX_AFFECT_LAW_DECAY));
    snprintf(label, sizeof label, "decay: no part grows over %u turns at %u",
             AOTX_AFFECT_LAW_DECAY, count);
    aotx_affect_note(label, falling == count * AOTX_AFFECT_LAW_DECAY, "turns",
                     (double)falling, (double)(count * AOTX_AFFECT_LAW_DECAY));
    snprintf(label, sizeof label, "decay: the fast part is zero after %u turns at %u",
             AOTX_AFFECT_LAW_DECAY, count);
    aotx_affect_note(label, zero == count, "agents", (double)zero, (double)count);
    printf("decay: agent 0 after %u turns holds fast %d slow %d of %d\n",
           AOTX_AFFECT_LAW_DECAY, after[0].fast[0], after[0].slow[0], AOTX_AFFECT_LAW_HIGH);
}

/* The caps bound the effective state and not the parts. */
static void aotx_affect_law_caps(aotx_affect_test_ring *ring, unsigned int count,
                                 const aotx_affect_law_figures *defaults)
{
    aotx_affect_law_figures f = *defaults;
    aotx_affect_law_records r;
    aotx_affect_agent_state after[AOTX_SLOTS];
    char label[112];
    unsigned int right = 0u;
    f.cap[0] = 0.3;
    f.cap[1] = 0.2;
    aotx_affect_law_settings(AOTX_SET_AFFECT_CAP_VALENCE, 3000);
    /* The reset above set the arousal cap back, so the second row goes in alone. */
    {
        aotx_settings_state table;
        aotx_check_runtime(cudaMemcpyFromSymbol(&table, aotx_setting_table, sizeof table),
                           "cudaMemcpyFromSymbol");
        table.row[AOTX_SET_AFFECT_CAP_AROUSAL].value = 2000;
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_setting_table, &table, sizeof table),
                           "cudaMemcpyToSymbol");
    }
    aotx_affect_law_preset(AOTX_AFFECT_LAW_HALF, AOTX_AFFECT_LAW_HALF, 1u);
    aotx_affect_test_ring_open(ring, 0u);
    aotx_affect_test_quiet<<<1, AOTX_SLOTS>>>(count);
    aotx_affect_turn<<<1, AOTX_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_affect_law_state_get(after);
    aotx_affect_law_collect(ring, &r);
    for (unsigned int a = 0u; a < count; ++a) {
        int sign = ((a & 1u) != 0u) ? -1 : 1;
        int valence = (int)rint(f.cap[0] * 32768.0);
        int arousal = (int)rint(f.cap[1] * 32768.0);
        const aotx_affect_trace_body *t = &r.trace[a];
        unsigned int good = (t->effective[0] == (short)(sign * valence)
                             && t->effective[1] == (short)(sign * arousal)
                             && t->effective[2] == 0 && t->effective[3] == 0
                             && (t->flags & AOTX_AFFECT_FLAG_CAP) != 0u
                             && (r.state[a].flags & AOTX_AFFECT_FLAG_CAP) != 0u) ? 1u : 0u;
        /* The parts themselves stay above the cap: the sum of the decayed halves. */
        good &= (abs(after[a].fast[0]) + abs(after[a].slow[0]) > valence) ? 1u : 0u;
        right += good;
    }
    snprintf(label, sizeof label, "caps: the effective state is bound and flagged at %u",
             count);
    aotx_affect_note(label, right == count, "agents", (double)right, (double)count);
    aotx_affect_law_settings(AOTX_SETTING_NUMBER_COUNT, 0);
}

/* A scripted state record sets the state of its agent; a record past the table does not. */
static void aotx_affect_law_apply(unsigned int count)
{
    aotx_affect_agent_state after[AOTX_SLOTS];
    char label[112];
    unsigned int right = 0u;
    aotx_affect_law_preset(AOTX_AFFECT_LAW_HALF, AOTX_AFFECT_LAW_HALF, 0u);
    aotx_affect_test_apply<<<1, AOTX_SLOTS>>>(count);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_affect_law_state_get(after);
    for (unsigned int a = 0u; a < count; ++a) {
        unsigned int good = (after[a].scale == 1000u + a && after[a].axes == 2u) ? 1u : 0u;
        for (unsigned int j = 0u; j < AOTX_AFFECT_STATE_AXES; ++j) {
            good &= (after[a].fast[j] == (short)(100 * (int)a + 7 * (int)j - 300)
                     && after[a].slow[j] == (short)(-50 * (int)a + 3 * (int)j + 200)) ? 1u : 0u;
        }
        right += good;
    }
    snprintf(label, sizeof label, "apply: a state record sets the table of %u agents", count);
    aotx_affect_note(label, right == count, "agents", (double)right, (double)count);
    if (count < AOTX_SLOTS) {
        unsigned int kept = (after[count].fast[0] == AOTX_AFFECT_LAW_HALF
                             && after[count].slow[0] == AOTX_AFFECT_LAW_HALF) ? 1u : 0u;
        aotx_affect_note("apply: a record past the table changes nothing", kept == 1u,
                         "agents", (double)kept, 1.0);
    }
}

/* A replay runs no update and writes no state record. */
static void aotx_affect_law_replay(aotx_affect_test_ring *ring, unsigned int count)
{
    aotx_affect_agent_state before[AOTX_SLOTS], after[AOTX_SLOTS];
    aotx_affect_law_records r;
    aotx_seam_state seam;
    char label[112];
    aotx_affect_law_preset(AOTX_AFFECT_LAW_HALF, -AOTX_AFFECT_LAW_HALF, 1u);
    aotx_affect_law_state_get(before);
    aotx_affect_test_ring_open(ring, 1u);
    aotx_affect_test_clear();
    aotx_affect_test_script<<<1, AOTX_SLOTS>>>(count, 1u);
    aotx_affect_turn<<<1, AOTX_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_affect_law_state_get(after);
    aotx_affect_law_collect(ring, &r);
    aotx_affect_law_seam(&seam);
    unsigned int same = (memcmp(before, after, sizeof before) == 0) ? 1u : 0u;
    snprintf(label, sizeof label, "replay: no update, no record and no fold at %u", count);
    aotx_affect_note(label, same == 1u && r.states == 0u && r.traces == 0u
                     && seam.apply.state_hash == 0ull && seam.apply.applied_count == 0ull,
                     "records", (double)(r.states + r.traces), 0.0);
}

/* The open with the setting off sets the state to zero; with the setting on it keeps it. */
static void aotx_affect_law_open(unsigned int count)
{
    aotx_affect_agent_state after[AOTX_SLOTS];
    char label[112];
    unsigned int zero = 0u, kept = 0u;
    aotx_affect_law_settings(AOTX_SET_AFFECT_ON, 0);
    aotx_affect_law_preset(AOTX_AFFECT_LAW_HALF, AOTX_AFFECT_LAW_HALF, 1u);
    aotx_affect_test_open<<<1, AOTX_SLOTS>>>(count);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_affect_law_state_get(after);
    for (unsigned int a = 0u; a < count; ++a) {
        aotx_affect_agent_state none = {};
        zero += (memcmp(&after[a], &none, sizeof none) == 0) ? 1u : 0u;
    }
    aotx_affect_law_settings(AOTX_SET_AFFECT_ON, 1);
    aotx_affect_law_preset(AOTX_AFFECT_LAW_HALF, AOTX_AFFECT_LAW_HALF, 1u);
    aotx_affect_test_open<<<1, AOTX_SLOTS>>>(count);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_affect_law_state_get(after);
    for (unsigned int a = 0u; a < count; ++a) {
        kept += (abs(after[a].fast[0]) == AOTX_AFFECT_LAW_HALF) ? 1u : 0u;
    }
    snprintf(label, sizeof label, "open: the setting off sets %u agents to zero", count);
    aotx_affect_note(label, zero == count, "agents", (double)zero, (double)count);
    snprintf(label, sizeof label, "open: the setting on keeps the state of %u agents", count);
    aotx_affect_note(label, kept == count, "agents", (double)kept, (double)count);
}

/* The law cases at one count. The store holds the four rows when probes is 1. */
static void aotx_affect_test_law(aotx_affect_test_ring *ring, unsigned int count,
                                 unsigned int probes)
{
    aotx_affect_law_figures f;
    aotx_affect_law_defaults(&f);
    aotx_affect_law_settings(AOTX_SETTING_NUMBER_COUNT, 0);
    aotx_affect_law_preset(0, 0, 0u);
    aotx_affect_law_turn(ring, count, probes, &f, "one turn from zero");
    for (unsigned int turn = 1u; turn < AOTX_AFFECT_LAW_TURNS; ++turn) {
        char name[48];
        snprintf(name, sizeof name, "turn %u", turn + 1u);
        aotx_affect_law_turn(ring, count, probes, &f, name);
    }
    aotx_affect_law_decay(ring, count, &f);
    aotx_affect_law_caps(ring, count, &f);
    if (probes != 0u) {
        aotx_affect_law_figures g = f;
        g.probe_gain = 0.5;
        aotx_affect_law_settings(AOTX_SET_AFFECT_PROBE_GAIN, 5000);
        aotx_affect_law_preset(0, 0, 0u);
        aotx_affect_law_turn(ring, count, probes, &g, "probe gain");
        aotx_affect_law_settings(AOTX_SETTING_NUMBER_COUNT, 0);
    }
    aotx_affect_law_apply(count);
    aotx_affect_law_replay(ring, count);
    aotx_affect_law_open(count);
    aotx_affect_law_none();
}

#endif
