/* Purpose: Load the refusal phrases and capture the quality turn node.
 * Owns: No device allocation; the phrase table is a device symbol.
 * Launch shape: Host glue only; the graph holds one block for each agent.
 * Lifetime: The phrase table lasts for the whole run. */
#include <cuda_runtime.h>

#include <stdio.h>
#include <string.h>

#include "boot/check.h"
#include "quality/quality.cuh"

static int aotx_quality_phrase_line(aotx_quality_phrase_table *table, const char *line)
{
    size_t length = strcspn(line, "\r\n");
    if (length == 0u) return 0;
    if (table->count >= AOTX_QUALITY_PHRASES || length > AOTX_QUALITY_PHRASE_BYTES) {
        fprintf(stderr, "the refusal phrase file has too many phrases or a long phrase\n");
        return 1;
    }
    memcpy(table->text[table->count], line, length);
    table->length[table->count] = (unsigned int)length;
    table->count += 1u;
    return 0;
}

int aotx_quality_load(const char *path)
{
    aotx_quality_phrase_table table;
    char line[256];
    memset(&table, 0, sizeof table);
    FILE *in = (path != 0) ? fopen(path, "r") : 0;
    if (in == 0) {
        fprintf(stderr, "the refusal phrase file does not open\n");
        return 1;
    }
    int bad = 0;
    while (!bad && fgets(line, sizeof line, in) != 0) {
        bad = aotx_quality_phrase_line(&table, line);
    }
    if (ferror(in) != 0) bad = 1;
    fclose(in);
    if (bad || table.count == 0u) return 1;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_quality_phrases, &table, sizeof table),
                       "cudaMemcpyToSymbol");
    return 0;
}

int aotx_quality_open_host(void)
{
    return aotx_quality_load(AOTX_QUALITY_PHRASE_FILE);
}

int aotx_quality_capture(void *stream)
{
    cudaStream_t on = (cudaStream_t)stream;
    aotx_quality_turn<<<AOTX_SLOTS, AOTX_QUALITY_PHRASES, 0, on>>>();
    return 0;
}
