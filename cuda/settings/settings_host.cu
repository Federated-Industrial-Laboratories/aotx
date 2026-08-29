/* Purpose: Make the page the pump reads and give its device address to the table.
 * Owns: The pinned control page.
 * Launch shape: Host glue; one one-thread kernel fills the table at the open.
 * Lifetime: From the open before the first tick to the close at exit. */
#include <cuda_runtime.h>
#include <stddef.h>

#include "boot/check.h"
#include "settings/settings.cuh"

/* The host address of the page, and the address the device writes. One page holds the
 * values the glue reads, so the pump takes one acquire load a tick. */
static aotx_settings_page *aotx_settings_host_page = 0;

int aotx_settings_page_open(void)
{
    if (aotx_settings_host_page != 0) {
        return 0;
    }
    void *host = 0;
    if (cudaHostAlloc(&host, sizeof(aotx_settings_page),
                      cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        return 1;
    }
    aotx_settings_host_page = (aotx_settings_page *)host;
    aotx_settings_host_page->period_ns = 0ull;
    aotx_settings_host_page->budget_ns = 0ull;
    aotx_settings_host_page->mirror_hz = 0ull;

    void *device = 0;
    if (cudaHostGetDevicePointer(&device, host, 0) != cudaSuccess) {
        cudaFreeHost(host);
        aotx_settings_host_page = 0;
        return 1;
    }
    aotx_settings_page *at = (aotx_settings_page *)device;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_settings_control, &at, sizeof at),
                       "cudaMemcpyToSymbol");

    /* The table takes the default of every key and publishes the control page, so the pump
     * reads a period from the first tick. */
    aotx_settings_boot<<<1, 1>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    return 0;
}

void aotx_settings_page_close(void)
{
    if (aotx_settings_host_page == 0) {
        return;
    }
    aotx_settings_page *none = 0;
    cudaMemcpyToSymbol(aotx_settings_control, &none, sizeof none);
    cudaFreeHost(aotx_settings_host_page);
    aotx_settings_host_page = 0;
}

/* The default of one number key. A caller that reads the page before the first tick, or a
 * run with no page, takes this value. */
long long aotx_settings_default(unsigned int index)
{
    switch (index) {
#define AOTX_SETTING_ONE(symbol, name, side, effect, value, ...) \
    case symbol: return (long long)(value);
    AOTX_SETTING_NUMBERS(AOTX_SETTING_ONE)
#undef AOTX_SETTING_ONE
    default: return 0;
    }
}

/* An acquire load of the field the device writes with a release store. The load orders the
 * read of the budget after it. The budget is stored before the period. A read that lands
 * between the two stores gives the new budget with the old period, and the next tick
 * gives both. */
static unsigned long long aotx_settings_acquire(const unsigned long long *at)
{
    return __atomic_load_n(at, __ATOMIC_ACQUIRE);
}

unsigned long long aotx_settings_period_ns(void)
{
    unsigned long long got = (aotx_settings_host_page != 0)
                           ? aotx_settings_acquire(&aotx_settings_host_page->period_ns)
                           : 0ull;
    if (got != 0ull) {
        return got;
    }
    return (unsigned long long)aotx_settings_default(AOTX_SET_TICK_PERIOD_MS)
         * 1000000ull;
}

unsigned long long aotx_settings_budget_ns(void)
{
    unsigned long long period = (aotx_settings_host_page != 0)
                              ? aotx_settings_acquire(&aotx_settings_host_page->period_ns)
                              : 0ull;
    if (period != 0ull) {
        return aotx_settings_host_page->budget_ns;
    }
    return (unsigned long long)aotx_settings_default(AOTX_SET_DECODE_BUDGET_MS)
         * 1000000ull;
}

unsigned long long aotx_settings_mirror_hz(void)
{
    unsigned long long period = (aotx_settings_host_page != 0)
                              ? aotx_settings_acquire(&aotx_settings_host_page->period_ns)
                              : 0ull;
    if (period != 0ull) {
        return aotx_settings_host_page->mirror_hz;
    }
    return (unsigned long long)aotx_settings_default(AOTX_SET_MIRROR_HZ);
}
