/* Purpose: Load the refusal phrases and capture the quality turn node.
 * Owns: No device allocation; the phrase table is a device symbol.
 * Launch shape: Host glue only; the graph holds one block for each agent.
 * Lifetime: The phrase table lasts for the whole run. */
#include <cuda_runtime.h>

#include <stdio.h>
#include "disk/runtime/assets.h"
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

static int aotx_quality_take(FILE *in)
{
    aotx_quality_phrase_table table;
    char line[256];
    memset(&table, 0, sizeof table);
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

int aotx_quality_load(const char *path)
{
    return aotx_quality_take(path ? fopen(path, "r") : NULL);
}

/* The phrases come from the model store when it holds them, else from the file the build
 * names. A store with neither loads no phrase: the refusal figure then stays 0, and one
 * line says so. A file that does not read as a phrase list refuses the load. */
int aotx_quality_load_store(const char *dir)
{
    aotx_quality_phrase_table none;
    FILE *in = aotx_asset_stream(dir, "quality/refusal-phrases.txt");
    if (in != 0) return aotx_quality_take(in);
    if (aotx_asset_is_runtime(dir)) {
        fprintf(stderr, "the runtime file has no refusal phrase asset\n");
        return 1;
    }
    in = fopen(AOTX_QUALITY_PHRASE_FILE, "r");
    if (in != 0) {
        fclose(in);
        return aotx_quality_load(AOTX_QUALITY_PHRASE_FILE);
    }
    memset(&none, 0, sizeof none);
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_quality_phrases, &none, sizeof none),
                       "cudaMemcpyToSymbol");
    fprintf(stderr, "quality: no refusal phrase file in %s/quality, the refusal figure stays 0\n",
            dir);
    return 0;
}

int aotx_quality_capture(void *stream)
{
    cudaStream_t on = (cudaStream_t)stream;
    aotx_quality_turn<<<AOTX_SLOTS, AOTX_QUALITY_PHRASES, 0, on>>>();
    return 0;
}
