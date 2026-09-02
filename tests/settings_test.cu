/* Purpose: Check the settings table, the record path, the refusals and the control page.
 * Owns: The record probe, the setting bodies and the counts of the cases.
 * Launch shape: One thread applies a line; the pump makes the ticks.
 * Lifetime: One run of the test program. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>

#include "boot/check.h"
#include "bus/bus.cuh"
#include "cli/cli.cuh"
#include "mem/mem.cuh"
#include "sched/sched.cuh"
#include "seam/seam.cuh"
#include "settings/settings.cuh"

#include "seam_feed.h"

/* Records the probe keeps of one case. */
#define AOTX_TEST_FOUND   2048u

static unsigned int aotx_test_applied;
static unsigned int aotx_test_failed;

static void aotx_test_check(int ok, const char *what)
{
    aotx_test_applied += 1u;
    if (!ok) {
        aotx_test_failed += 1u;
        printf("settings: FAILED %s\n", what);
    }
}

/* One record the probe found. */
typedef struct aotx_test_record {
    unsigned long long seq;
    unsigned int writer;
    unsigned int cls;
    unsigned int flags;
    unsigned int length;
    unsigned char body[AOTX_BODY_BYTES];
} aotx_test_record;

__global__ void aotx_test_gather(unsigned int type, aotx_test_record *out, unsigned int max,
                                 unsigned int *count)
{
    unsigned long long tail = aotx_seam.dev.tail;
    unsigned long long stride = (unsigned long long)(gridDim.x * blockDim.x);
    for (unsigned long long seq = 1ull + (unsigned long long)(blockIdx.x * blockDim.x
                                                              + threadIdx.x);
         seq <= tail; seq += stride) {
        const volatile aotx_record_header *header = aotx_cli_slot(seq);
        if (!aotx_cli_holds(header, seq, type)) {
            continue;
        }
        unsigned int at = atomicAdd(count, 1u);
        if (at >= max) {
            continue;
        }
        unsigned int length = header->body_len;
        if (length > AOTX_BODY_BYTES) {
            length = AOTX_BODY_BYTES;
        }
        out[at].seq = seq;
        out[at].writer = header->writer;
        out[at].cls = header->cls;
        out[at].flags = header->flags;
        out[at].length = length;
        const volatile unsigned char *from = (const volatile unsigned char *)header
                                           + AOTX_HEADER_BYTES;
        for (unsigned int b = 0u; b < length; ++b) {
            out[at].body[b] = from[b];
        }
    }
}

static int aotx_test_order(const void *left, const void *right)
{
    const aotx_test_record *a = (const aotx_test_record *)left;
    const aotx_test_record *b = (const aotx_test_record *)right;
    return (a->seq < b->seq) ? -1 : ((a->seq > b->seq) ? 1 : 0);
}

static unsigned int aotx_test_records(unsigned int type, aotx_test_record *out,
                                      unsigned int max)
{
    aotx_test_record *device = NULL;
    unsigned int *counter = NULL;
    unsigned int count = 0u;
    aotx_check_runtime(cudaMalloc(&device, (size_t)max * sizeof *device), "cudaMalloc");
    aotx_check_runtime(cudaMalloc(&counter, sizeof *counter), "cudaMalloc");
    aotx_check_runtime(cudaMemset(counter, 0, sizeof *counter), "cudaMemset");
    aotx_test_gather<<<64, 128>>>(type, device, max, counter);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(&count, counter, sizeof count, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    if (count > max) {
        count = max;
    }
    aotx_check_runtime(cudaMemcpy(out, device, (size_t)count * sizeof *out,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    cudaFree(device);
    cudaFree(counter);
    qsort(out, count, sizeof *out, aotx_test_order);
    return count;
}

/* Fill the table with the defaults again, so every case starts from the same rows. */
__global__ void aotx_test_clear(void)
{
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        aotx_settings_reset();
        aotx_settings_publish();
    }
}

/* Parse one command line, as the apply step does. */
__global__ void aotx_test_line(const unsigned char *text, unsigned int length)
{
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        aotx_cli_line(text, length, aotx_time_tick);
    }
}

static void aotx_test_command(const char *line)
{
    unsigned char *device = NULL;
    unsigned int length = (unsigned int)strlen(line);
    aotx_check_runtime(cudaMalloc(&device, AOTX_BODY_BYTES), "cudaMalloc");
    aotx_check_runtime(cudaMemcpy(device, line, length, cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    aotx_test_line<<<1, 1>>>(device, length);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    cudaFree(device);
}

static aotx_settings_state aotx_test_table(void)
{
    aotx_settings_state table;
    aotx_check_runtime(cudaMemcpyFromSymbol(&table, aotx_setting_table, sizeof table),
                       "cudaMemcpyFromSymbol");
    return table;
}

/* The name, the side and the range of a number key, on the host side. The lists come from
 * keys.h, so the check states no figure of its own. */
static const char *aotx_test_name(unsigned int index)
{
    switch (index) {
#define AOTX_SETTING_ONE(symbol, name, side, effect, ...) case symbol: return name;
    AOTX_SETTING_NUMBERS(AOTX_SETTING_ONE)
#undef AOTX_SETTING_ONE
    default: return "-";
    }
}

static unsigned int aotx_test_side(unsigned int index)
{
    switch (index) {
#define AOTX_SETTING_ONE(symbol, name, side, effect, ...) \
    case symbol: return AOTX_SETTING_SIDE_##side;
    AOTX_SETTING_NUMBERS(AOTX_SETTING_ONE)
#undef AOTX_SETTING_ONE
    default: return 0u;
    }
}

static long long aotx_test_least(unsigned int index)
{
    switch (index) {
#define AOTX_SETTING_ONE(symbol, name, side, effect, value, least, ...) \
    case symbol: return (long long)(least);
    AOTX_SETTING_NUMBERS(AOTX_SETTING_ONE)
#undef AOTX_SETTING_ONE
    default: return 0;
    }
}

static long long aotx_test_most(unsigned int index)
{
    switch (index) {
#define AOTX_SETTING_ONE(symbol, name, side, effect, value, least, most, ...) \
    case symbol: return (long long)(most);
    AOTX_SETTING_NUMBERS(AOTX_SETTING_ONE)
#undef AOTX_SETTING_ONE
    default: return 0;
    }
}

static int aotx_test_scale(unsigned int index)
{
    switch (index) {
#define AOTX_SETTING_ONE(symbol, name, side, effect, value, least, most, scale) \
    case symbol: return (scale);
    AOTX_SETTING_NUMBERS(AOTX_SETTING_ONE)
#undef AOTX_SETTING_ONE
    default: return AOTX_SETTING_SCALE_ONE;
    }
}

/* Build one setting body. */
static void aotx_test_body(aotx_setting_body *body, const char *key, long long value,
                           unsigned int scale)
{
    memset(body, 0, sizeof *body);
    body->value = value;
    body->scale = scale;
    body->key_len = (unsigned int)strlen(key);
    memcpy(body->key, key, body->key_len);
}

/* The fold of the state hash, as the apply makes it. */
static unsigned long long aotx_test_fold(unsigned long long hash, const void *bytes,
                                         unsigned int count)
{
    const unsigned char *at = (const unsigned char *)bytes;
    for (unsigned int i = 0u; i < count; ++i) {
        hash ^= (unsigned long long)at[i];
        hash *= AOTX_FNV_PRIME;
    }
    return hash;
}

/* Run ticks until the apply consumed every record of the inbound ring. */
static void aotx_test_drain(aotx_pump *pump, aotx_seam_rings *rings)
{
    const volatile aotx_inbound_preamble *preamble =
        (const volatile aotx_inbound_preamble *)rings->inbound_map;
    for (unsigned int i = 0u; i < 64u; ++i) {
        aotx_pump_tick(pump);
        if (preamble->consumed >= preamble->head) {
            return;
        }
    }
}

/* Every row holds the default of its key after the boot of the table. */
static void aotx_test_case_defaults(void)
{
    aotx_test_clear<<<1, 1>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_settings_state table = aotx_test_table();
    unsigned int wrong = 0u;
    unsigned int device_keys = 0u;
    for (unsigned int i = 0u; i < (unsigned int)AOTX_SETTING_NUMBER_COUNT; ++i) {
        wrong += (table.row[i].value != aotx_settings_default(i)) ? 1u : 0u;
        wrong += (table.row[i].changed != 0ull) ? 1u : 0u;
        device_keys += (aotx_test_side(i) == AOTX_SETTING_SIDE_DEVICE) ? 1u : 0u;
    }
    aotx_test_check(wrong == 0u, "every row holds the default of its key and no tick");
    aotx_test_check(device_keys > 0u, "the list names settings the device reads");
    printf("settings: %u keys, %u of them read by the device\n",
           (unsigned int)AOTX_SETTING_NUMBER_COUNT, device_keys);
}

/* Apply a batch of setting records through the inbound ring and the apply node. Every
 * record names a key the device reads and a value inside its range, and no two records of
 * one key carry the same value. */
static void aotx_test_case_batch(aotx_pump *pump, aotx_seam_rings *rings,
                                 unsigned long long boot_id, unsigned int count)
{
    aotx_pump_report report;
    aotx_setting_body *bodies =
        (aotx_setting_body *)calloc(count, sizeof *bodies);
    long long *want = (long long *)calloc(AOTX_SETTING_NUMBER_COUNT, sizeof(long long));

    aotx_test_clear<<<1, 1>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_settings_state before_table = aotx_test_table();
    for (unsigned int i = 0u; i < (unsigned int)AOTX_SETTING_NUMBER_COUNT; ++i) {
        want[i] = before_table.row[i].value;
    }
    aotx_pump_read(&report);
    unsigned long long hash = report.state_hash;
    unsigned long long applied = report.applied;

    /* The device keys, in the order of the list. A batch larger than the list walks it
     * again with another value each time. */
    unsigned int keys[AOTX_SETTING_NUMBER_COUNT];
    unsigned int key_count = 0u;
    for (unsigned int i = 0u; i < (unsigned int)AOTX_SETTING_NUMBER_COUNT; ++i) {
        if (aotx_test_side(i) == AOTX_SETTING_SIDE_DEVICE) {
            keys[key_count++] = i;
        }
    }
    for (unsigned int i = 0u; i < count; ++i) {
        unsigned int index = keys[i % key_count];
        long long least = aotx_test_least(index);
        long long most = aotx_test_most(index);
        long long value = least + (long long)(i % 7u) + (long long)(i / key_count);
        if (value > most) {
            value = most;
        }
        aotx_test_body(&bodies[i], aotx_test_name(index), value,
                       (unsigned int)aotx_test_scale(index));
        want[index] = value;
        hash = aotx_test_fold(hash, &bodies[i], (unsigned int)sizeof bodies[i]);
    }
    aotx_test_feed_records(rings, AOTX_REC_SETTING, AOTX_CLASS_A, AOTX_WRITER_FEEDER, 0u,
                           bodies, (unsigned int)sizeof *bodies, count, boot_id);
    aotx_test_drain(pump, rings);

    aotx_settings_state after = aotx_test_table();
    aotx_pump_read(&report);
    unsigned int wrong = 0u;
    for (unsigned int i = 0u; i < (unsigned int)AOTX_SETTING_NUMBER_COUNT; ++i) {
        wrong += (after.row[i].value != want[i]) ? 1u : 0u;
    }
    aotx_test_check(wrong == 0u, "every row holds the value of the last record of its key");
    aotx_test_check(after.applied == count, "the table counted every record it took");
    aotx_test_check(after.refused == 0u, "no record of the batch was refused");
    aotx_test_check(report.applied == applied + (unsigned long long)count,
                    "the apply counted every class A record");
    aotx_test_check(report.state_hash == hash,
                    "the state hash folded every setting body in order");
    printf("settings: the batch of %u records left hash %llx and %u rows changed\n",
           count, report.state_hash, key_count);
    free(bodies);
    free(want);
}

/* The record the apply writes again carries the class, the writer and the flags a class A
 * record carries. A record that came from a replay names the restore writer. */
static void aotx_test_case_journal(aotx_pump *pump, aotx_seam_rings *rings,
                                   unsigned long long boot_id)
{
    aotx_test_record *found =
        (aotx_test_record *)malloc(AOTX_TEST_FOUND * sizeof *found);
    aotx_setting_body body;
    unsigned int count = 0u;
    unsigned int live = 0u;
    unsigned int replayed = 0u;

    aotx_test_clear<<<1, 1>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_body(&body, "tick.period_ms", 12, AOTX_SETTING_SCALE_ONE);
    aotx_test_feed_records(rings, AOTX_REC_SETTING, AOTX_CLASS_A, AOTX_WRITER_FEEDER, 0u,
                           &body, (unsigned int)sizeof body, 1u, boot_id);
    aotx_test_drain(pump, rings);
    aotx_test_body(&body, "tick.period_ms", 13, AOTX_SETTING_SCALE_ONE);
    aotx_test_feed_records(rings, AOTX_REC_SETTING, AOTX_CLASS_A, AOTX_WRITER_FEEDER,
                           AOTX_FLAG_REPLAYED, &body, (unsigned int)sizeof body, 1u,
                           boot_id);
    aotx_test_drain(pump, rings);

    count = aotx_test_records(AOTX_REC_SETTING, found, AOTX_TEST_FOUND);
    for (unsigned int i = 0u; i < count; ++i) {
        if (found[i].cls != AOTX_CLASS_A
            || found[i].length < (unsigned int)sizeof(aotx_setting_body)) {
            continue;
        }
        if (found[i].writer == AOTX_WRITER_FEEDER && found[i].flags == 0u) {
            live += 1u;
        }
        if (found[i].writer == AOTX_WRITER_RESTORE
            && (found[i].flags & AOTX_FLAG_REPLAYED) != 0u) {
            replayed += 1u;
        }
    }
    aotx_test_check(live >= 1u,
                    "the record the apply writes again is class A and names the feeder");
    aotx_test_check(replayed >= 1u,
                    "a replayed record is class A and names the restore writer");
    aotx_test_check(aotx_test_table().row[AOTX_SET_TICK_PERIOD_MS].value == 13,
                    "the last record of the key holds the row");
    printf("settings: %u setting records in the ring, %u live and %u replayed\n",
           count, live, replayed);
    free(found);
}

/* An unknown key, a value under the least and a value over the most are refused. Each one
 * writes one console line and one bus note, and the table does not change. */
static void aotx_test_case_refusals(aotx_pump *pump, aotx_seam_rings *rings,
                                    unsigned long long boot_id)
{
    aotx_test_record *found =
        (aotx_test_record *)malloc(AOTX_TEST_FOUND * sizeof *found);
    aotx_setting_body bodies[4];
    long long least = aotx_test_least(AOTX_SET_TICK_PERIOD_MS);
    long long most = aotx_test_most(AOTX_SET_TICK_PERIOD_MS);

    aotx_test_clear<<<1, 1>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int console_before = aotx_test_records(AOTX_REC_CONSOLE, found,
                                                    AOTX_TEST_FOUND);
    unsigned int bus_before = aotx_test_records(AOTX_REC_BUS, found, AOTX_TEST_FOUND);

    aotx_test_body(&bodies[0], "tick.period_days", 1, AOTX_SETTING_SCALE_ONE);
    aotx_test_body(&bodies[1], "tick.period_ms", least - 1, AOTX_SETTING_SCALE_ONE);
    aotx_test_body(&bodies[2], "tick.period_ms", most + 1, AOTX_SETTING_SCALE_ONE);
    /* A key the boot glue reads and the device does not. */
    aotx_test_body(&bodies[3], "window.on", 1, AOTX_SETTING_SCALE_ONE);
    aotx_test_feed_records(rings, AOTX_REC_SETTING, AOTX_CLASS_A, AOTX_WRITER_FEEDER, 0u,
                           bodies, (unsigned int)sizeof bodies[0], 4u, boot_id);
    aotx_test_drain(pump, rings);

    aotx_settings_state table = aotx_test_table();
    unsigned int console_after = aotx_test_records(AOTX_REC_CONSOLE, found,
                                                   AOTX_TEST_FOUND);
    unsigned int bus_after = aotx_test_records(AOTX_REC_BUS, found, AOTX_TEST_FOUND);

    aotx_test_check(table.refused == 4u, "the table refused every one of the four records");
    aotx_test_check(table.applied == 0u, "no refused record changed a row");
    aotx_test_check(table.row[AOTX_SET_TICK_PERIOD_MS].value
                    == aotx_settings_default(AOTX_SET_TICK_PERIOD_MS),
                    "the row of a refused value keeps the value it had");
    aotx_test_check(console_after >= console_before + 4u,
                    "each refusal wrote one console line");
    aotx_test_check(bus_after >= bus_before + 4u, "each refusal wrote one bus note");
    printf("settings: the four refusals gave %u console lines and %u bus notes\n",
           console_after - console_before, bus_after - bus_before);
    free(found);
}

/* The set command builds a class A record, folds it into the hash and applies it. The
 * settings command writes one line for each key the device reads. The control page holds
 * the two pump values after the tick that follows. */
static void aotx_test_case_command(aotx_pump *pump)
{
    aotx_pump_report report;
    aotx_test_record *found =
        (aotx_test_record *)malloc(AOTX_TEST_FOUND * sizeof *found);
    aotx_setting_body body;

    aotx_test_clear<<<1, 1>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_pump_read(&report);
    unsigned long long hash = report.state_hash;
    unsigned long long applied = report.applied;
    unsigned int before = aotx_test_records(AOTX_REC_SETTING, found, AOTX_TEST_FOUND);

    aotx_test_command("set tick.period_ms 20");
    /* The record of a set line is written by the tick commit node, after every applied
     * line of the tick. The fold order is therefore the order of the journal. */
    aotx_test_check(aotx_test_table().row[AOTX_SET_TICK_PERIOD_MS].value
                    == aotx_settings_default(AOTX_SET_TICK_PERIOD_MS),
                    "a set line changes no row before the tick commits");
    aotx_pump_tick(pump);
    aotx_test_body(&body, "tick.period_ms", 20, AOTX_SETTING_SCALE_ONE);
    hash = aotx_test_fold(hash, &body, (unsigned int)sizeof body);
    aotx_pump_read(&report);
    unsigned int after = aotx_test_records(AOTX_REC_SETTING, found, AOTX_TEST_FOUND);
    aotx_settings_state table = aotx_test_table();

    aotx_test_check(table.row[AOTX_SET_TICK_PERIOD_MS].value == 20,
                    "the set command gave the row its value");
    aotx_test_check(after == before + 1u, "the set command wrote one setting record");
    aotx_test_check(report.state_hash == hash,
                    "the set command folded its record into the state hash");
    aotx_test_check(report.applied == applied + 1ull,
                    "the set command counted its record as applied");
    aotx_test_check(found[after - 1u].writer == AOTX_WRITER_CONSOLE
                    && found[after - 1u].cls == AOTX_CLASS_A,
                    "the record of the set command is class A and names the console");

    /* A number with decimals lands in the scaled unit of its key. */
    aotx_test_command("set sample.temperature 0.35");
    aotx_pump_tick(pump);
    table = aotx_test_table();
    aotx_test_check(table.row[AOTX_SET_TEMPERATURE].value == 3500,
                    "a value with decimals lands in the scaled unit");

    /* The console takes what the file takes: a leading plus, and decimals that are all
     * zero for a whole-number key. */
    aotx_test_command("set tick.period_ms +60.00");
    aotx_pump_tick(pump);
    table = aotx_test_table();
    aotx_test_check(table.row[AOTX_SET_TICK_PERIOD_MS].value == 60,
                    "a plus and zero decimals land as the whole number");
    aotx_test_command("set tick.period_ms 20");
    aotx_pump_tick(pump);
    table = aotx_test_table();

    /* Two refusals at the console: a key that is not known and a value out of range. */
    unsigned int refused = table.refused;
    aotx_test_command("set tick.period_years 3");
    aotx_test_command("set tick.period_ms 100000");
    aotx_pump_tick(pump);
    table = aotx_test_table();
    aotx_test_check(table.row[AOTX_SET_TICK_PERIOD_MS].value == 20,
                    "a refused set line leaves the row as it was");
    aotx_test_check(table.refused >= refused + 1u, "a refused set line is counted");

    /* The settings command writes one line for each key the device reads, and a head. */
    unsigned int console_before = aotx_test_records(AOTX_REC_CONSOLE, found,
                                                    AOTX_TEST_FOUND);
    aotx_test_command("settings");
    unsigned int console_after = aotx_test_records(AOTX_REC_CONSOLE, found,
                                                   AOTX_TEST_FOUND);
    unsigned int device_keys = 0u;
    for (unsigned int i = 0u; i < (unsigned int)AOTX_SETTING_NUMBER_COUNT; ++i) {
        device_keys += (aotx_test_side(i) == AOTX_SETTING_SIDE_DEVICE) ? 1u : 0u;
    }
    aotx_test_check(console_after >= console_before + device_keys + 1u,
                    "the settings command writes a head and one line for each key");

    /* The tick commit node publishes the two pump values, so the page holds them. */
    aotx_test_check(aotx_settings_period_ns() == 20ull * 1000000ull,
                    "the control page holds the period the set command gave");
    aotx_test_check(aotx_settings_budget_ns()
                    == (unsigned long long)aotx_settings_default(AOTX_SET_DECODE_BUDGET_MS)
                       * 1000000ull,
                    "the control page holds the decode budget of the table");
    printf("settings: the control page holds period %llu ns and budget %llu ns\n",
           aotx_settings_period_ns(), aotx_settings_budget_ns());
    free(found);
}

#ifdef AOTX_AFFECT
typedef struct aotx_test_affect_setting {
    unsigned int index;
    const char *value;
    long long scaled;
} aotx_test_affect_setting;

/* Each optional setting takes a set line and writes the matching class A record. */
static void aotx_test_case_affect_commands(aotx_pump *pump)
{
    static const aotx_test_affect_setting cases[] = {
        { AOTX_SET_AFFECT_ON, "1", 1 },
        { AOTX_SET_QUALITY_ON, "1", 1 },
        { AOTX_SET_AFFECT_PROBE_GAIN, "0.25", 2500 },
        { AOTX_SET_AFFECT_DECAY_FAST, "0.4", 4000 },
        { AOTX_SET_AFFECT_DECAY_SLOW, "0.8", 8000 },
        { AOTX_SET_AFFECT_GAIN_FAST, "1.5", 15000 },
        { AOTX_SET_AFFECT_GAIN_SLOW, "0.2", 2000 },
        { AOTX_SET_AFFECT_CAP_VALENCE, "0.75", 7500 },
        { AOTX_SET_AFFECT_CAP_AROUSAL, "0.5", 5000 },
        { AOTX_SET_AFFECT_TEMPERATURE_GAIN, "-0.25", -2500 },
        { AOTX_SET_AFFECT_VOICE_GAIN, "0.3", 3000 },
        { AOTX_SET_AFFECT_STEER_GAIN, "0.4", 4000 },
        { AOTX_SET_AFFECT_BUDGET, "0.5", 5000 }
    };
    aotx_test_record *found =
        (aotx_test_record *)malloc(AOTX_TEST_FOUND * sizeof *found);
    unsigned int before = aotx_test_records(AOTX_REC_SETTING, found, AOTX_TEST_FOUND);
    unsigned int wrong = 0u;
    aotx_test_clear<<<1, 1>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    for (unsigned int i = 0u; i < sizeof cases / sizeof cases[0]; ++i) {
        char line[128];
        snprintf(line, sizeof line, "set %s %s", aotx_test_name(cases[i].index),
                 cases[i].value);
        aotx_test_command(line);
        aotx_pump_tick(pump);
        wrong += (aotx_test_table().row[cases[i].index].value != cases[i].scaled) ? 1u : 0u;
    }
    unsigned int after = aotx_test_records(AOTX_REC_SETTING, found, AOTX_TEST_FOUND);
    for (unsigned int i = before; i < after; ++i) {
        if (found[i].writer != AOTX_WRITER_CONSOLE || found[i].cls != AOTX_CLASS_A
            || found[i].length != sizeof(aotx_setting_body)) {
            wrong += 1u;
        }
    }
    if (after == before + sizeof cases / sizeof cases[0]) {
        for (unsigned int i = 0u; i < sizeof cases / sizeof cases[0]; ++i) {
            aotx_setting_body want;
            aotx_test_body(&want, aotx_test_name(cases[i].index), cases[i].scaled,
                           (unsigned int)aotx_test_scale(cases[i].index));
            if (memcmp(found[before + i].body, &want, sizeof want) != 0) {
                wrong += 1u;
            }
        }
    }
    aotx_test_check(wrong == 0u,
                    "each optional set line changes its row and writes its class A body");
    aotx_test_check(after == before + sizeof cases / sizeof cases[0],
                    "each optional set line writes one setting record");
    printf("settings: %u optional set lines wrote %u setting records\n",
           (unsigned int)(sizeof cases / sizeof cases[0]), after - before);
    free(found);
}
#endif

/* A replay of the journal sends every line again. The setting record of a set line stands
 * in the journal beside that line. The command therefore writes nothing while a replay
 * runs, and the record makes the change one time. */
static void aotx_test_case_replay(aotx_pump *pump)
{
    aotx_pump_report report;
    aotx_test_record *found =
        (aotx_test_record *)malloc(AOTX_TEST_FOUND * sizeof *found);

    aotx_test_clear<<<1, 1>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_pump_read(&report);
    unsigned long long hash = report.state_hash;
    unsigned long long applied = report.applied;
    unsigned int before = aotx_test_records(AOTX_REC_SETTING, found, AOTX_TEST_FOUND);

    aotx_seam_set_replaying(1);
    aotx_test_command("set tick.period_ms 30");
    aotx_seam_set_replaying(0);
    aotx_pump_tick(pump);

    unsigned int after = aotx_test_records(AOTX_REC_SETTING, found, AOTX_TEST_FOUND);
    aotx_settings_state table = aotx_test_table();
    aotx_pump_read(&report);
    aotx_test_check(after == before, "a set line writes no record while a replay runs");
    aotx_test_check(table.row[AOTX_SET_TICK_PERIOD_MS].value
                    == aotx_settings_default(AOTX_SET_TICK_PERIOD_MS),
                    "a set line changes no row while a replay runs");
    aotx_test_check(report.state_hash == hash && report.applied == applied,
                    "a set line folds nothing into the state hash while a replay runs");
    printf("settings: a set line under a replay wrote %u records and left hash %llx\n",
           after - before, report.state_hash);
    free(found);
}

int main(void)
{
    CUdevice device;
    CUcontext context;
    aotx_mem_map map;
    aotx_seam_rings rings;
    aotx_pump pump;
    unsigned long long boot_id = 0x5E771465ull;

    aotx_check_driver(cuInit(0), "cuInit");
    aotx_check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    aotx_check_driver(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    aotx_check_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    if (aotx_mem_reserve(&map) != 0 || aotx_seam_open(&rings, boot_id) != 0) {
        printf("settings: the memory map or the rings did not open\n");
        return 1;
    }
    aotx_seam_bind(&rings, map.ring, map.ring_bytes, boot_id);
    aotx_seam_bind_bulk(&rings, map.scratch, AOTX_BULK_STAGE_BYTES);
    aotx_seam_note_boot<<<1, 1>>>(0ull, 0ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    if (aotx_settings_page_open() != 0) {
        printf("settings: the control page did not open\n");
        return 1;
    }
    if (aotx_pump_build(&pump, 0ull, 1u) != 0) {
        printf("settings: the tick graph did not build\n");
        return 1;
    }

    aotx_test_case_defaults();
    aotx_test_case_batch(&pump, &rings, boot_id, 1u);
    aotx_test_case_batch(&pump, &rings, boot_id, AOTX_SLOTS);
    aotx_test_case_journal(&pump, &rings, boot_id);
    aotx_test_case_refusals(&pump, &rings, boot_id);
    aotx_test_case_command(&pump);
#ifdef AOTX_AFFECT
    aotx_test_case_affect_commands(&pump);
#endif
    aotx_test_case_replay(&pump);

    aotx_pump_close(&pump);
    aotx_settings_page_close();
    aotx_seam_close(&rings);
    aotx_mem_release(&map);
    printf("settings: %u cases applied, %u passed, %u failed\n", aotx_test_applied,
           aotx_test_applied - aotx_test_failed, aotx_test_failed);
    if (aotx_test_applied == 0u) {
        printf("settings: no case ran\n");
        return 1;
    }
    return (aotx_test_failed == 0u) ? 0 : 1;
}
