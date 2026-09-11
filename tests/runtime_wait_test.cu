/* Purpose: Check first-checkpoint waiting with late acknowledgments and terminal conditions.
 * Owns: Distinct startup handles, a controlled clock and checkpoint transport fields.
 * Launch shape: Host control at one and 64 handles; pump and child polls have fixed test inputs.
 * Lifetime: Each case ends on acknowledgment, cancellation, writer exit or disk error. */
#include "boot/boot.cuh"
#include "cognitive/checkpoint.cuh"
#include <stdio.h>
#include <time.h>

static unsigned checks, failures, mode, ticks;
static int expected_writer, active;
static long long elapsed;
static aotx_checkpoint_ring *transport;
static void check(bool ok, const char *message) {
    ++checks;
    if (!ok) { ++failures; fprintf(stderr, "FAIL: %s\n", message); }
}
extern "C" int __real_clock_gettime(clockid_t clock, struct timespec *value);
extern "C" int __wrap_clock_gettime(clockid_t clock, struct timespec *value) {
    if (!active) return __real_clock_gettime(clock, value);
    value->tv_sec = elapsed / 1000000000ll; value->tv_nsec = elapsed % 1000000000ll;
    return 0;
}
extern "C" void __wrap__Z14aotx_pump_tickP9aotx_pump(aotx_pump *pump) {
    (void)pump; ++ticks; elapsed += 10000000000ll;
    if (mode == 0 && ticks == 30) transport->consumed = 1;
    if (mode == 3 && ticks == 3) transport->error = 1;
}
extern "C" void __wrap__Z14aotx_pump_paceP9aotx_pump(aotx_pump *pump) { (void)pump; }
extern "C" int __wrap__Z14aotx_seam_polliPiS_(int pid, int *ended, int *status) {
    check(pid == expected_writer, "the wait checks its own writer");
    *ended = mode == 2 && ticks == 3; *status = *ended ? 17 : 0;
    return 0;
}
static int stopped(void) { return (mode == 1 && ticks == 4) || ticks == 100; }
static void batch(unsigned n) {
    for (unsigned i = 0; i < n; ++i) for (mode = 0; mode < 6; ++mode) {
        aotx_checkpoint_ring ring = {}; ring.boot = 500 + i; ring.generation = 7 + i;
        transport = &ring; ticks = 0; elapsed = 0; expected_writer = 1000 + (int)i;
        aotx_boot_children children = {}; children.drain = mode == 4 ? 0 : expected_writer;
        aotx_boot_options options = {}; options.ccir = mode == 5 ? NULL : "identity.aotxccir";
        aotx_seam_rings rings = {}; rings.checkpoint_map = (unsigned char *)&ring;
        aotx_pump pump = {}; active = 1;
        int rc = aotx_boot_runtime_ready(&options, &rings, &pump, stopped, &children);
        active = 0;
        const unsigned expected_ticks[] = {30, 4, 3, 3, 0, 0};
        check(rc == ((mode == 0 || mode == 5) ? 0 : 1), "startup returns the required terminal status");
        check(ticks == expected_ticks[mode], "the wait reaches its acknowledgment or terminal condition");
        check(mode != 0 || elapsed > 180000000000ll, "late healthy acknowledgment exceeds 180 seconds");
        check(mode != 2 || !children.drain, "a reaped writer is removed from the child handles");
        check(mode == 0 || !ring.consumed, "a refusal cannot create a durable acknowledgment");
    }
}
int main(void) {
    batch(1); batch(64);
    printf("runtime wait: %u checks, %u failures\n", checks, failures);
    return failures || checks < 390 ? 1 : 0;
}
