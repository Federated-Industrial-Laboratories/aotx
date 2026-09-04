/* Purpose: Check the affect couplings, the composite row and the budget scale.
 * Owns: The synthetic logits, sampler rows and composite fixture of this check.
 * Launch shape: One thread or one block for each agent.
 * Lifetime: One run of the affect check. */
#ifndef AOTX_TESTS_AFFECT_ACTUATOR_H
#define AOTX_TESTS_AFFECT_ACTUATOR_H

#include "cli/prompt.cuh"

#define AOTX_AFFECT_TEST_LOGITS 5u
#define AOTX_AFFECT_TEST_HEAT   0.8f
#define AOTX_AFFECT_TEST_TOP    2.0
#define AOTX_AFFECT_TEST_BIASED 1u
#define AOTX_AFFECT_TEST_BIAS   1.5f

/* Open and apply one sampler row for each agent. */
__global__ void aotx_affect_test_how(unsigned int count, aotx_model_how *out,
                                     unsigned int voice)
{
    unsigned int agent = threadIdx.x;
    if (agent >= count) return;
    aotx_model_how how = {};
    how.temperature = AOTX_AFFECT_TEST_HEAT;
    how.top_p = 1.0f;
    how.repeat_penalty = 1.0f;
    how.think_limit = -1;
    for (unsigned int i = 0u; i < AOTX_MODEL_STEERS; ++i) how.steer[i] = AOTX_MODEL_CONDUCT_NONE;
    how.voice = voice;
    how.voice_scale = 1.0f;
    aotx_affect_open(agent, &how);
    aotx_affect_apply_how(agent, &how);
    out[agent] = how;
}

/* The entropy of the fixture row at one temperature, in nats. The row holds the five
 * logits -2 to 2, so the largest is 2. */
static double aotx_affect_test_row_entropy(double heat)
{
    double sum = 0.0;
    double weighted = 0.0;
    for (unsigned int i = 0u; i < AOTX_AFFECT_TEST_LOGITS; ++i) {
        double one = ((double)i - 2.0) / heat;
        double chance = exp(one - AOTX_AFFECT_TEST_TOP / heat);
        sum += chance;
        weighted += chance * one;
    }
    return log(sum) + AOTX_AFFECT_TEST_TOP / heat - weighted / sum;
}

/* The probability of the biased token of the fixture row at one voice scale. */
static double aotx_affect_test_row_class(double heat, double scale)
{
    double sum = 0.0;
    double mass = 0.0;
    for (unsigned int i = 0u; i < AOTX_AFFECT_TEST_LOGITS; ++i) {
        double one = (double)i - 2.0;
        if (i == AOTX_AFFECT_TEST_BIASED) one += scale * (double)AOTX_AFFECT_TEST_BIAS;
        double chance = exp(one / heat);
        sum += chance;
        if (i == AOTX_AFFECT_TEST_BIASED) mass += chance;
    }
    return mass / sum;
}

__global__ void aotx_affect_test_apply_only(unsigned int count, aotx_model_how *out)
{
    unsigned int agent = threadIdx.x;
    if (agent >= count) return;
    aotx_model_how how = {};
    how.temperature = 0.8f; how.voice_scale = 1.0f;
    for (unsigned int i = 0u; i < AOTX_MODEL_STEERS; ++i) how.steer[i] = AOTX_MODEL_CONDUCT_NONE;
    how.voice = AOTX_MODEL_CONDUCT_NONE; how.affect = 1u;
    aotx_affect_apply_how(agent, &how); out[agent] = how;
}

/* A setting-off snapshot must keep the complete plain sampler row. */
__global__ void aotx_affect_test_off(unsigned int count, unsigned int *same)
{
    unsigned int agent = threadIdx.x;
    if (agent >= count) return;
    aotx_model_how plain = {};
    plain.temperature = 0.73f; plain.top_k = 17u; plain.top_p = 0.81f;
    plain.min_p = 0.03f; plain.repeat_penalty = 1.07f; plain.repeat_window = 23u;
    plain.presence_penalty = 0.2f; plain.frequency_penalty = -0.1f;
    plain.seed = 91ull; plain.think_limit = 11; plain.voice = AOTX_MODEL_CONDUCT_NONE;
    plain.voice_scale = 1.0f;
    for (unsigned int i = 0u; i < AOTX_MODEL_STEERS; ++i) plain.steer[i] = AOTX_MODEL_CONDUCT_NONE;
    aotx_model_how got = plain;
    aotx_affect_open(agent, &got); aotx_affect_apply_how(agent, &got);
    const unsigned char *a = (const unsigned char *)&plain;
    const unsigned char *b = (const unsigned char *)&got;
    unsigned int equal = 1u;
    for (unsigned int i = 0u; i < sizeof plain; ++i) equal &= a[i] == b[i];
    same[agent] = equal;
}

static void aotx_affect_test_actuator_settings(unsigned int on, long long steer,
                                                long long budget)
{
    aotx_settings_state table;
    aotx_affect_test_settings<<<1, 1>>>(AOTX_SETTING_NUMBER_COUNT, 0);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpyFromSymbol(&table, aotx_setting_table, sizeof table),
                       "cudaMemcpyFromSymbol");
    table.row[AOTX_SET_AFFECT_ON].value = on;
    table.row[AOTX_SET_AFFECT_TEMPERATURE_GAIN].value = 5000;
    table.row[AOTX_SET_AFFECT_VOICE_GAIN].value = 5000;
    table.row[AOTX_SET_AFFECT_STEER_GAIN].value = steer;
    table.row[AOTX_SET_AFFECT_BUDGET].value = budget;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_setting_table, &table, sizeof table),
                       "cudaMemcpyToSymbol");
}

static void aotx_affect_test_state(float valence, float arousal)
{
    aotx_affect_agent_state state[AOTX_SLOTS];
    memset(state, 0, sizeof state);
    for (unsigned int a = 0u; a < AOTX_SLOTS; ++a) {
        state[a].fast[0] = (short)rintf(valence * 32768.0f);
        state[a].fast[1] = (short)rintf(arousal * 32768.0f);
        state[a].scale = AOTX_AFFECT_SCALE_ONE;
        state[a].axes = AOTX_AFFECT_DATA_AXES;
    }
    aotx_affect_law_state_set(state);
}

/* Read the pick entropy at five state values. */
static void aotx_affect_test_entropy(aotx_affect_test_ring *ring, unsigned int count)
{
    float host[AOTX_SLOTS * AOTX_AFFECT_TEST_LOGITS];
    unsigned int agent[AOTX_SLOTS];
    float entropy[5], temperature[5], voice[5], base[5], moved[5];
    float arousal[5] = { -0.8f, -0.4f, 0.0f, 0.4f, 0.8f };
    aotx_model_how *how = (aotx_model_how *)aotx_affect_take(count * sizeof *how);
    float *head = (float *)aotx_affect_take((unsigned long long)count
                                            * AOTX_AFFECT_TEST_LOGITS * sizeof(float));
    unsigned int *device_agent = (unsigned int *)aotx_affect_take(count * sizeof *device_agent);
    int *token = (int *)aotx_affect_take(count * sizeof *token);
    unsigned int *draw = (unsigned int *)aotx_affect_take(count * sizeof *draw);
    for (unsigned int a = 0u; a < count; ++a) {
        agent[a] = a;
        for (unsigned int i = 0u; i < AOTX_AFFECT_TEST_LOGITS; ++i)
            host[a * AOTX_AFFECT_TEST_LOGITS + i] = (float)i - 2.0f;
    }
    aotx_check_runtime(cudaMemcpy(head, host, count * AOTX_AFFECT_TEST_LOGITS * sizeof(float),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(device_agent, agent, count * sizeof *agent,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_affect_test_actuator_settings(1u, 0, 2500);
    for (unsigned int s = 0u; s < 5u; ++s) {
        aotx_affect_test_state(0.5f, arousal[s]); aotx_affect_test_clear();
        aotx_affect_test_ring_open(ring, 0u);
        aotx_affect_test_how<<<1, AOTX_SLOTS>>>(count, how, AOTX_MODEL_CONDUCT_NONE);
        aotx_model_desc desc = {}; aotx_model_work work = {}; aotx_model_run run = {};
        desc.role = AOTX_AFFECT_TEST_ROLE; desc.vocab = AOTX_AFFECT_TEST_LOGITS;
        work.head = head; run.agent = device_agent; run.token = token; run.draw = draw;
        run.how = how; run.seqs = count; run.rows = count; run.telemetry = 1u;
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_model, &desc, sizeof desc,
                           (size_t)AOTX_AFFECT_TEST_ROLE * sizeof desc), "cudaMemcpyToSymbol");
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_space, &work, sizeof work,
                           (size_t)AOTX_AFFECT_TEST_ROLE * sizeof work), "cudaMemcpyToSymbol");
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_call, &run, sizeof run,
                           (size_t)AOTX_AFFECT_TEST_ROLE * sizeof run), "cudaMemcpyToSymbol");
        aotx_model_pick<<<count, AOTX_MODEL_ROW_THREADS>>>(AOTX_AFFECT_TEST_ROLE);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_affect_sums sums[AOTX_SLOTS]; aotx_affect_test_sums(sums);
        entropy[s] = sums[0].entropy_sum;
        base[s] = sums[0].entropy_base_sum;
        moved[s] = sums[0].class_sum;
        aotx_model_how row;
        aotx_check_runtime(cudaMemcpy(&row, how, sizeof row, cudaMemcpyDeviceToHost),
                           "cudaMemcpy");
        temperature[s] = row.temperature;
        voice[s] = row.voice_scale;
    }
    int monotone = entropy[0] < entropy[1] && entropy[1] < entropy[2]
                && entropy[2] < entropy[3] && entropy[3] < entropy[4];
    char label[96]; snprintf(label, sizeof label, "pick entropy rises over five states at %u", count);
    aotx_affect_note(label, monotone, "entropy", entropy[4] - entropy[0], 0.0);
    snprintf(label, sizeof label, "zero arousal keeps base temperature at %u", count);
    aotx_affect_note(label, fabsf(temperature[2] - 0.8f) < 1.0e-6f,
                     "temperature", temperature[2], 0.8);
    snprintf(label, sizeof label, "voice scale multiplies the valence gain at %u", count);
    aotx_affect_note(label, fabsf(voice[2] - 1.25f) < 1.0e-6f,
                     "scale", voice[2], 1.25);
    /* The entropy shift of each state, against the host entropy of the same row at the
     * applied temperature and at the base temperature of 0.8. */
    int shifted = 1;
    double worst = 0.0;
    for (unsigned int s = 0u; s < 5u; ++s) {
        double want = aotx_affect_test_row_entropy((double)temperature[s])
                    - aotx_affect_test_row_entropy((double)AOTX_AFFECT_TEST_HEAT);
        double got = (double)entropy[s] - (double)base[s];
        worst = fmax(worst, fabs(got - want));
        shifted &= (fabs(got - want) < 1.0e-4) ? 1 : 0;
    }
    snprintf(label, sizeof label, "the entropy shift follows the temperature at %u", count);
    aotx_affect_note(label, shifted, "nats", worst, 0.0);
    snprintf(label, sizeof label, "zero arousal gives no entropy shift at %u", count);
    aotx_affect_note(label, fabs((double)entropy[2] - (double)base[2]) < 1.0e-5,
                     "nats", (double)entropy[2] - (double)base[2], 0.0);
    int quiet = 1;
    for (unsigned int s = 0u; s < 5u; ++s) quiet &= (moved[s] == 0.0f) ? 1 : 0;
    snprintf(label, sizeof label, "no voice profile gives no class shift at %u", count);
    aotx_affect_note(label, quiet, "shift", moved[2], 0.0);
    cudaFree(how); cudaFree(head); cudaFree(device_agent); cudaFree(token); cudaFree(draw);
    aotx_affect_test_model();
}

static unsigned int aotx_affect_test_voice_at = AOTX_MODEL_CONDUCT_NONE;

/* Register one voice profile that biases one token of the fixture row. */
static void aotx_affect_test_voice_open(void)
{
    unsigned int voices = 0u;
    const unsigned int token = AOTX_AFFECT_TEST_BIASED;
    const float bias = AOTX_AFFECT_TEST_BIAS;
    if (aotx_affect_test_voice_at != AOTX_MODEL_CONDUCT_NONE) return;
    aotx_check_runtime(cudaMemcpyFromSymbol(&voices, aotx_conduct, sizeof voices,
                                            offsetof(aotx_conduct_table, voices)),
                       "cudaMemcpyFromSymbol");
    if (aotx_conduct_register_voice("bias", &token, &bias, 1u) == 0) {
        aotx_affect_test_voice_at = voices;
    }
}

/* The class frequency shift of the picks: at a voice gain that scales the bias, and at a
 * voice gain of zero. The state keeps zero arousal, so the temperature stays at the base
 * and the host recomputation reads one temperature. */
static void aotx_affect_test_voice(aotx_affect_test_ring *ring, unsigned int count)
{
    float host[AOTX_SLOTS * AOTX_AFFECT_TEST_LOGITS];
    unsigned int agent[AOTX_SLOTS];
    long long gains[2] = { 5000, 0 };
    float moved[2] = { 0.0f, 0.0f };
    float scale[2] = { 1.0f, 1.0f };
    char label[96];
    aotx_affect_test_voice_open();
    if (aotx_affect_test_voice_at == AOTX_MODEL_CONDUCT_NONE) {
        aotx_affect_note("the voice profile of the class case registers", 0, "profiles",
                         0.0, 1.0);
        return;
    }
    aotx_model_how *how = (aotx_model_how *)aotx_affect_take(count * sizeof *how);
    float *head = (float *)aotx_affect_take((unsigned long long)count
                                            * AOTX_AFFECT_TEST_LOGITS * sizeof(float));
    unsigned int *device_agent = (unsigned int *)aotx_affect_take(count * sizeof *device_agent);
    int *token = (int *)aotx_affect_take(count * sizeof *token);
    unsigned int *draw = (unsigned int *)aotx_affect_take(count * sizeof *draw);
    for (unsigned int a = 0u; a < count; ++a) {
        agent[a] = a;
        for (unsigned int i = 0u; i < AOTX_AFFECT_TEST_LOGITS; ++i)
            host[a * AOTX_AFFECT_TEST_LOGITS + i] = (float)i - 2.0f;
    }
    aotx_check_runtime(cudaMemcpy(head, host, count * AOTX_AFFECT_TEST_LOGITS * sizeof(float),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(device_agent, agent, count * sizeof *agent,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    for (unsigned int g = 0u; g < 2u; ++g) {
        aotx_settings_state table;
        aotx_affect_test_actuator_settings(1u, 0, 2500);
        /* The voice gain goes in beside the other settings. A settings kernel resets the
         * whole table, which would clear the affect switch of this case. */
        aotx_check_runtime(cudaMemcpyFromSymbol(&table, aotx_setting_table, sizeof table),
                           "cudaMemcpyFromSymbol");
        table.row[AOTX_SET_AFFECT_VOICE_GAIN].value = gains[g];
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_setting_table, &table, sizeof table),
                           "cudaMemcpyToSymbol");
        aotx_affect_test_state(0.5f, 0.0f);
        aotx_affect_test_clear();
        aotx_affect_test_ring_open(ring, 0u);
        aotx_affect_test_how<<<1, AOTX_SLOTS>>>(count, how, aotx_affect_test_voice_at);
        aotx_model_desc desc = {}; aotx_model_work work = {}; aotx_model_run run = {};
        desc.role = AOTX_AFFECT_TEST_ROLE; desc.vocab = AOTX_AFFECT_TEST_LOGITS;
        work.head = head; run.agent = device_agent; run.token = token; run.draw = draw;
        run.how = how; run.seqs = count; run.rows = count; run.telemetry = 1u;
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_model, &desc, sizeof desc,
                           (size_t)AOTX_AFFECT_TEST_ROLE * sizeof desc), "cudaMemcpyToSymbol");
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_space, &work, sizeof work,
                           (size_t)AOTX_AFFECT_TEST_ROLE * sizeof work), "cudaMemcpyToSymbol");
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_call, &run, sizeof run,
                           (size_t)AOTX_AFFECT_TEST_ROLE * sizeof run), "cudaMemcpyToSymbol");
        aotx_model_pick<<<count, AOTX_MODEL_ROW_THREADS>>>(AOTX_AFFECT_TEST_ROLE);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_affect_sums sums[AOTX_SLOTS]; aotx_affect_test_sums(sums);
        unsigned int every = 0u;
        for (unsigned int a = 0u; a < count; ++a)
            every += (sums[a].class_sum == sums[0].class_sum) ? 1u : 0u;
        moved[g] = sums[0].class_sum;
        aotx_model_how row;
        aotx_check_runtime(cudaMemcpy(&row, how, sizeof row, cudaMemcpyDeviceToHost),
                           "cudaMemcpy");
        scale[g] = row.voice_scale;
        snprintf(label, sizeof label, "every agent gives the same class shift at %u", count);
        aotx_affect_note(label, every == count, "agents", (double)every, (double)count);
    }
    double want = aotx_affect_test_row_class((double)AOTX_AFFECT_TEST_HEAT, (double)scale[0])
                - aotx_affect_test_row_class((double)AOTX_AFFECT_TEST_HEAT, 1.0);
    snprintf(label, sizeof label, "the class shift follows the voice bias at %u", count);
    aotx_affect_note(label, fabs((double)moved[0] - want) < 1.0e-4 && want > 0.01, "shift",
                     (double)moved[0], want);
    snprintf(label, sizeof label, "a zero voice gain gives a zero class shift at %u", count);
    aotx_affect_note(label, moved[1] == 0.0f && fabsf(scale[1] - 1.0f) < 1.0e-6f, "shift",
                     (double)moved[1], 0.0);
    cudaFree(how); cudaFree(head); cudaFree(device_agent); cudaFree(token); cudaFree(draw);
    aotx_affect_test_model();
}

static void aotx_affect_test_plain(unsigned int count)
{
    unsigned int *device = (unsigned int *)aotx_affect_take(count * sizeof(unsigned int));
    unsigned int same[AOTX_SLOTS];
    aotx_affect_test_actuator_settings(0u, 10000, 2500);
    aotx_affect_test_off<<<1, AOTX_SLOTS>>>(count, device);
    aotx_check_runtime(cudaMemcpy(same, device, count * sizeof *same, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    unsigned int right = 0u; for (unsigned int a = 0u; a < count; ++a) right += same[a];
    char label[96]; snprintf(label, sizeof label, "setting off keeps the plain row at %u", count);
    aotx_affect_note(label, right == count, "rows", right, count); cudaFree(device);
}

static void aotx_affect_test_actuator_snapshot(unsigned int count)
{
    aotx_affect_test_actuator_settings(1u, 10000, 2500);
    aotx_affect_test_state(0.5f, 0.5f);
    aotx_affect_test_open<<<1, AOTX_SLOTS>>>(count);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_settings_state table;
    aotx_check_runtime(cudaMemcpyFromSymbol(&table, aotx_setting_table, sizeof table),
                       "cudaMemcpyFromSymbol");
    table.row[AOTX_SET_AFFECT_TEMPERATURE_GAIN].value = -10000;
    table.row[AOTX_SET_AFFECT_VOICE_GAIN].value = -10000;
    table.row[AOTX_SET_AFFECT_STEER_GAIN].value = 0;
    table.row[AOTX_SET_AFFECT_BUDGET].value = 40000;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_setting_table, &table, sizeof table),
                       "cudaMemcpyToSymbol");
    aotx_model_how *device = (aotx_model_how *)aotx_affect_take(count * sizeof *device);
    aotx_affect_test_apply_only<<<1, AOTX_SLOTS>>>(count, device);
    aotx_model_how how; aotx_affect_law law;
    aotx_check_runtime(cudaMemcpy(&how, device, sizeof how, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_check_runtime(cudaMemcpyFromSymbol(&law, aotx_affect_laws, sizeof law),
                       "cudaMemcpyFromSymbol");
    char label[96]; snprintf(label, sizeof label, "an open sequence keeps all actuator settings at %u", count);
    int same = fabsf(how.temperature - 1.0f) < 1.0e-6f
            && fabsf(how.voice_scale - 1.25f) < 1.0e-6f
            && law.steer_gain == 1.0f && law.budget == 0.25f;
    printf("actuator snapshot: temperature %.9g, voice %.9g, steer %.9g, budget %.9g\n",
           (double)how.temperature, (double)how.voice_scale, (double)law.steer_gain,
           (double)law.budget);
    aotx_affect_note(label, same, "settings", same, 1.0); cudaFree(device);
}

static void aotx_affect_test_composite_loader(aotx_affect_test_store *store)
{
    aotx_affect_composite_desc table;
    const float *basis = 0; float *rows = 0;
    int wrote = aotx_affect_test_composite_store(store, 1u);
    int loaded = aotx_affect_load_store(store->dir);
    aotx_check_runtime(cudaMemcpyFromSymbol(&table, aotx_affect_composite_table, sizeof table),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(&basis, aotx_affect_composite, sizeof basis),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(&rows, aotx_affect_steer, sizeof rows),
                       "cudaMemcpyFromSymbol");
    aotx_affect_note("both marks load the composite", wrote == 0 && loaded == 0
                     && table.trusted == 1u && table.hidden == AOTX_AFFECT_TEST_HIDDEN
                     && table.layer_count == 1u && basis != 0 && rows != 0, "marks",
                     table.trusted, 1.0);
    aotx_affect_test_calibration(store, 0u); aotx_affect_load_store(store->dir);
    aotx_check_runtime(cudaMemcpyFromSymbol(&table, aotx_affect_composite_table, sizeof table),
                       "cudaMemcpyFromSymbol");
    aotx_affect_test_actuator_settings(1u, 10000, 2500);
    aotx_affect_test_open<<<1, 1>>>(1u); aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_affect_law law; aotx_check_runtime(cudaMemcpyFromSymbol(&law, aotx_affect_laws,
                                           sizeof law), "cudaMemcpyFromSymbol");
    aotx_model_how *device = (aotx_model_how *)aotx_affect_take(sizeof *device), how;
    aotx_affect_test_apply_only<<<1, 1>>>(1u, device);
    aotx_check_runtime(cudaMemcpy(&how, device, sizeof how, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_affect_note("a line without both marks gives zero steer gain",
                     table.trusted == 0u && law.steer_gain == 0.0f
                     && how.steer[AOTX_MODEL_CONDUCT_AFFECT] == AOTX_MODEL_CONDUCT_NONE,
                     "gain", law.steer_gain, 0.0);
    cudaFree(device);
    aotx_affect_test_calibration(store, 1u); aotx_affect_load_store(store->dir);
}

static void aotx_affect_test_composite(aotx_affect_test_ring *ring, unsigned int count)
{
    aotx_say_state say = {}; unsigned int counts[AOTX_SLOTS] = {};
    for (unsigned int a = 0u; a < count; ++a) { say.slot[a].wanted = 1u; counts[a] = 1u; }
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_say, &say, sizeof say), "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_say_count, counts, sizeof counts), "cudaMemcpyToSymbol");
    aotx_affect_test_actuator_settings(1u, 10000, 2500);
    aotx_affect_test_state(0.5f, 0.999969482f);
    aotx_affect_build<<<AOTX_SLOTS, 256u>>>(); aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_affect_composite_desc table; float *device = 0; aotx_affect_agent_state state[AOTX_SLOTS];
    aotx_check_runtime(cudaMemcpyFromSymbol(&table, aotx_affect_composite_table, sizeof table), "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(&device, aotx_affect_steer, sizeof device), "cudaMemcpyFromSymbol");
    aotx_affect_law_state_get(state);
    size_t cells = (size_t)table.layer_count * table.hidden;
    float *got = (float *)malloc(count * cells * sizeof(float));
    aotx_check_runtime(cudaMemcpy(got, device, count * cells * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
    float direction[2][AOTX_AFFECT_TEST_HIDDEN];
    aotx_affect_test_direction(0u, table.hidden, direction[0]); aotx_affect_test_direction(1u, table.hidden, direction[1]);
    float d0 = 0.5f, d1 = 32767.0f / 32768.0f;
    float q = 4.0f * d0 * d0 + d1 * d1; float u = sqrtf(0.5f / q);
    unsigned int right = 0u;
    for (unsigned int a = 0u; a < count; ++a) {
        int good = state[a].scale == (unsigned short)rintf(u * 65535.0f)
                && (state[a].actuator_flags & 6u) == 6u;
        for (unsigned int x = 0u; x < table.hidden; ++x)
            good &= fabsf(got[(size_t)a * cells + x]
                          - u * (d0 * direction[0][x] + d1 * direction[1][x])) < 1.0e-5f;
        right += good ? 1u : 0u;
    }
    char label[96]; snprintf(label, sizeof label, "composite row and budget scale at %u", count);
    aotx_affect_note(label, right == count, "rows", right, count);
    float kl = 0.5f * u * u * q; snprintf(label, sizeof label, "applied KL stays in budget at %u", count);
    aotx_affect_note(label, kl <= 0.275f && kl >= 0.225f, "nats", kl, 0.25);
    float *resid = (float *)aotx_affect_take((unsigned long long)count * table.hidden
                                             * sizeof(float));
    unsigned int offset[AOTX_SLOTS + 1u], agent[AOTX_SLOTS];
    for (unsigned int a = 0u; a < count; ++a) { offset[a] = a; agent[a] = a; }
    offset[count] = count;
    unsigned int *device_offset = (unsigned int *)aotx_affect_take((count + 1u) * sizeof *device_offset);
    unsigned int *device_agent = (unsigned int *)aotx_affect_take(count * sizeof *device_agent);
    aotx_model_how *device_how = (aotx_model_how *)aotx_affect_take(count * sizeof *device_how);
    aotx_check_runtime(cudaMemcpy(device_offset, offset, (count + 1u) * sizeof *offset,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(device_agent, agent, count * sizeof *agent,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_affect_test_how<<<1, AOTX_SLOTS>>>(count, device_how,
                                            AOTX_MODEL_CONDUCT_NONE);
    aotx_model_desc desc = {}; aotx_model_work work = {}; aotx_model_run run = {};
    desc.role = AOTX_AFFECT_TEST_ROLE; desc.hidden = table.hidden;
    desc.layers = AOTX_AFFECT_TEST_LAYERS; work.resid = resid;
    run.offset = device_offset; run.agent = device_agent; run.how = device_how;
    run.seqs = count; run.tokens = count;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model, &desc, sizeof desc,
                       (size_t)AOTX_AFFECT_TEST_ROLE * sizeof desc), "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_space, &work, sizeof work,
                       (size_t)AOTX_AFFECT_TEST_ROLE * sizeof work), "cudaMemcpyToSymbol");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_call, &run, sizeof run,
                       (size_t)AOTX_AFFECT_TEST_ROLE * sizeof run), "cudaMemcpyToSymbol");
    aotx_model_conduct<<<AOTX_DECODE_WAVE, AOTX_MODEL_ROW_THREADS>>>(
        AOTX_AFFECT_TEST_ROLE, AOTX_AFFECT_TEST_LAYER);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    float *applied = (float *)malloc(count * table.hidden * sizeof(float));
    aotx_check_runtime(cudaMemcpy(applied, resid, count * table.hidden * sizeof(float),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    unsigned int resolved = 0u;
    for (unsigned int a = 0u; a < count; ++a) {
        int good = 1;
        for (unsigned int x = 0u; x < table.hidden; ++x)
            good &= fabsf(applied[(size_t)a * table.hidden + x]
                          - got[(size_t)a * cells + x]) < 1.0e-5f;
        resolved += good ? 1u : 0u;
    }
    snprintf(label, sizeof label, "conduct resolves the affect slot at %u", count);
    aotx_affect_note(label, resolved == count, "rows", resolved, count);
    free(applied); cudaFree(resid); cudaFree(device_offset); cudaFree(device_agent);
    cudaFree(device_how);
    aotx_affect_test_ring_open(ring, 0u); aotx_affect_test_quiet<<<1, AOTX_SLOTS>>>(count);
    aotx_affect_turn<<<1, AOTX_SLOTS>>>(); aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_affect_law_records record; aotx_affect_law_collect(ring, &record);
    unsigned int carried = 0u; for (unsigned int a = 0u; a < count; ++a)
        carried += record.state[a].scale == state[a].scale
                && (record.state[a].flags & 6u) == 6u ? 1u : 0u;
    snprintf(label, sizeof label, "state records carry the budget scale at %u", count);
    aotx_affect_note(label, carried == count, "records", carried, count);
    unsigned int spent = 0u; for (unsigned int a = 0u; a < count; ++a)
        spent += (fabsf(record.trace[a].budget_spent - kl) < 1.0e-5f) ? 1u : 0u;
    snprintf(label, sizeof label, "the trace carries the budget spent at %u", count);
    aotx_affect_note(label, spent == count, "records", spent, count);
    free(got);
    /* A budget above the dose: the scale stays at one, the budget flag stays clear and the
     * whole quadratic form is spent. */
    aotx_affect_test_actuator_settings(1u, 10000, 40000);
    aotx_affect_test_state(0.5f, 0.999969482f);
    aotx_affect_build<<<AOTX_SLOTS, 256u>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_affect_law_state_get(state);
    unsigned int inside = 0u; for (unsigned int a = 0u; a < count; ++a)
        inside += (state[a].scale == (unsigned short)AOTX_AFFECT_SCALE_ONE
                   && (state[a].actuator_flags & AOTX_AFFECT_FLAG_BUDGET) == 0u
                   && fabsf(state[a].budget_spent - 0.5f * q) < 1.0e-5f) ? 1u : 0u;
    snprintf(label, sizeof label, "a budget above the dose spends the whole dose at %u",
             count);
    aotx_affect_note(label, inside == count, "agents", inside, count);
}

#endif
