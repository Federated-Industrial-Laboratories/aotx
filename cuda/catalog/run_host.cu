/* Purpose: Run one module over a synthetic batch and read its figures.
 * Owns: Nothing; the batch is the batch of the tick and the verdict is a device global.
 * Launch shape: Host glue only; the kernels of the check judge every figure.
 * Lifetime: One run of the check program. */
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

#include "boot/check.h"
#include "catalog/check.cuh"
#include "catalog/check_host.h"
#include "settings/keys.h"
#include "tool/module_host.h"

extern "C" {
#include "disk/settings/settings.h"
}

/* The budget of one launch, in microseconds. A module node stands inside one tick, and a
 * tick with no decode holds the period of the tick. The figure therefore comes from the
 * default of the setting tick.period_ms and not from a constant of this file. */
static unsigned int aotx_check_budget_us(void)
{
    aotx_settings table;
    aotx_settings_defaults(&table);
    return (unsigned int)(table.number[AOTX_SET_TICK_PERIOD_MS] * 1000);
}

/* Run the module over a batch of a count of rows and judge what it wrote. */
static void aotx_check_batch(unsigned int node, unsigned int entry, unsigned int rows)
{
    aotx_check_verdict verdict;
    cudaEvent_t start;
    cudaEvent_t stop;
    float took = 0.0f;
    char line[128];

    aotx_check_fill<<<AOTX_SLOTS, 1>>>(node, entry, rows);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaEventCreate(&start), "cudaEventCreate");
    aotx_check_runtime(cudaEventCreate(&stop), "cudaEventCreate");
    aotx_check_runtime(cudaEventRecord(start, 0), "cudaEventRecord");
    aotx_tool_module_launch(entry);
    aotx_check_runtime(cudaEventRecord(stop, 0), "cudaEventRecord");
    aotx_check_runtime(cudaEventSynchronize(stop), "cudaEventSynchronize");
    aotx_check_runtime(cudaEventElapsedTime(&took, start, stop), "cudaEventElapsedTime");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    aotx_check_judge<<<AOTX_SLOTS, 1>>>(rows);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpyFromSymbol(&verdict, aotx_check_out, sizeof verdict),
                       "cudaMemcpyFromSymbol");

    unsigned int micro = (unsigned int)(took * 1000.0f);
    snprintf(line, sizeof line, "at %u rows the module answered rows:", rows);
    aotx_check_say(verdict.done == rows, line, verdict.done);
    snprintf(line, sizeof line, "at %u rows a status the contract refuses:", rows);
    aotx_check_say(verdict.status_bad == 0u, line, verdict.status_bad);
    snprintf(line, sizeof line, "at %u rows a length over the bound:", rows);
    aotx_check_say(verdict.over == 0u, line, verdict.over);
    snprintf(line, sizeof line, "at %u rows an untaken row that was written:", rows);
    aotx_check_say(verdict.untaken == 0u, line, verdict.untaken);
    snprintf(line, sizeof line, "at %u rows the longest result of %u bytes:", rows,
             (unsigned int)AOTX_TOOL_RESULT_BYTES);
    aotx_check_say(verdict.longest <= (unsigned int)AOTX_TOOL_RESULT_BYTES, line,
                   verdict.longest);
    unsigned int budget = aotx_check_budget_us();
    snprintf(line, sizeof line,
             "at %u rows the launch of the tick period of %u microseconds took:", rows,
             budget);
    aotx_check_say(micro <= budget, line, micro);
}

/* The device arm: load the module by digest, read its figures and run the batches. */
void aotx_check_device(const char *dir, unsigned int entry, unsigned int rows,
                              aotx_check_entry *held, aotx_check_entry *on)
{
    unsigned char digest[32];
    char path[1024];
    aotx_tool_module_row row;
    int regs = 0;
    int local = 0;
    int threads = 0;
    int ptx = 0;
    int arch = 0;

    if (aotx_tool_module_plan_row(0u, &row) != 0) {
        aotx_check_say(0, "the plan holds a row for the module:", 0ull);
        return;
    }
    snprintf(path, sizeof path, "%s/%s", dir, row.file);
    if (aotx_tool_module_digest(path, digest) != 0) {
        aotx_check_say(0, "the module file opens and hashes:", 0ull);
        return;
    }
    /* The module file is known now, so the import comes again with its digest, as the
     * feeder publishes it. The sha256 line of the manifest is judged against that digest,
     * and an import of the same name takes the entry that stands. */
    aotx_check_import_dir(dir, held, on, digest, 2u);
    aotx_check_say(held->state == AOTX_CATALOG_INSTALLED,
                   "the manifest and the digest of the module file:",
                   (unsigned long long)held->state);
    if (held->state != AOTX_CATALOG_INSTALLED) {
        printf("check: the reason is: %s\n", held->reason);
        return;
    }

    unsigned int made = aotx_tool_module_open();
    aotx_check_say(made == 1u, "modules the driver holds:", made);
    if (made != 1u) {
        return;
    }
    int node = aotx_tool_module_place(entry);
    if (node < 0 || aotx_tool_module_figures(entry, &regs, &local, &threads, &ptx,
                                             &arch) != 0) {
        aotx_check_say(0, "the figures of the kernel:", 0ull);
        return;
    }
    aotx_check_say(local == 0, "bytes of local memory, which must be none:",
                   (unsigned long long)local);
    aotx_check_say(regs > 0, "registers the kernel keeps:", (unsigned long long)regs);
    aotx_check_say(threads >= (int)AOTX_TOOL_MODULE_THREADS,
                   "threads of a block the kernel takes:", (unsigned long long)threads);
    /* The target line of the module text says what the module was written for. The binary
     * version says what the driver made for this card. A module of a target above the card
     * does not load, so the target is the figure a stranger reads. */
    int target = aotx_tool_module_target(entry);
    aotx_check_say(target > 0 && target <= (int)AOTX_ARCH,
                   "the target line of the module text names sm_:",
                   (unsigned long long)target);
    aotx_check_say(arch <= (int)AOTX_ARCH,
                   "the architecture the driver made for this card:",
                   (unsigned long long)arch);
    aotx_check_say(ptx > 0, "the version of the module text, times ten:",
                   (unsigned long long)ptx);
    if (rows != 0u) {
        aotx_check_batch((unsigned int)node, entry, rows);
        return;
    }
    aotx_check_batch((unsigned int)node, entry, 1u);
    aotx_check_batch((unsigned int)node, entry, AOTX_SLOTS);
}

