/* Purpose: Find and read the module file of one entry, and take the digest and the target.
 * Owns: The two directories of the run: the module root and the journal.
 * Launch shape: Host glue only; no kernel stands in this file.
 * Lifetime: From the first load to the close of the run.
 *
 * The loader takes three routes to a module file, in this order. The first is the path the
 * head of the import carried, which holds 63 bytes and fails for a longer one. The second
 * is the root of the run and the name of the module, which is the name of its directory.
 * The third is the table the feeder writes in the journal, which holds the directory of
 * every import against its number. */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tool/module.cuh"
#include "tool/module_host.h"

#include <signal.h>
#include <stdint.h>
extern "C" {
#include "disk/wire/diskwire.h"
}

/* The directory of the module directories of the run. The head of an import carries the
 * tail of the path of the directory it came from, and that field holds 63 bytes. A longer
 * path therefore does not open. The loader then looks for the module below this root, by
 * the name of the module, which is the name of its directory. */
static char aotx_tool_module_dir[AOTX_MODULE_ROOT_BYTES];

void aotx_tool_module_root(const char *dir)
{
    if (dir == NULL) {
        aotx_tool_module_dir[0] = '\0';
        return;
    }
    snprintf(aotx_tool_module_dir, sizeof aotx_tool_module_dir, "%s", dir);
}

/* The journal directory of the run. The feeder writes a table there with one row for each
 * import: its number, its name and the directory it came from. */
static char aotx_tool_module_book[AOTX_MODULE_ROOT_BYTES];

void aotx_tool_module_journal(const char *dir)
{
    if (dir == NULL) {
        aotx_tool_module_book[0] = '\0';
        return;
    }
    snprintf(aotx_tool_module_book, sizeof aotx_tool_module_book, "%s/%s", dir,
             "modules.jsonl");
}

/* The directory of one import, from the table the feeder writes. The read of a file is
 * glue, as the read of the models manifest is. The return is 0 when the row is found. */
static int aotx_tool_module_from_table(unsigned int import, char *out, size_t bytes)
{
    char line[1024];
    uint64_t number = 0u;
    if (aotx_tool_module_book[0] == '\0') {
        return 1;
    }
    FILE *at = fopen(aotx_tool_module_book, "rb");
    if (at == NULL) {
        return 1;
    }
    int found = 1;
    while (found != 0 && fgets(line, (int)sizeof line, at) != NULL) {
        if (aotx_json_number(line, "\"import\":", &number) != 0
            && (unsigned int)number == import
            && aotx_json_text(line, "\"dir\":\"", out, bytes) != 0) {
            found = 0;
        }
    }
    fclose(at);
    return found;
}

/* The architecture of the target line of a module text, or zero. */
int aotx_tool_module_target_of(const char *text)
{
    const char *at = strstr(text, ".target sm_");
    int made = 0;
    if (at == NULL) {
        return 0;
    }
    at += 11;
    while (*at >= '0' && *at <= '9') {
        made = made * 10 + (*at - '0');
        at += 1;
    }
    return made;
}

/* Read a whole file. The caller frees the bytes. */
static char *aotx_tool_module_file(const char *path, size_t *bytes)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *text = (size >= 0) ? (char *)malloc((size_t)size + 1u) : NULL;
    if (text == NULL || fread(text, 1u, (size_t)size, file) != (size_t)size) {
        free(text);
        fclose(file);
        return NULL;
    }
    text[size] = '\0';
    *bytes = (size_t)size;
    fclose(file);
    return text;
}


char *aotx_tool_module_text(const aotx_tool_module_row *row, size_t *bytes, char *path,
                            size_t path_bytes)
{
    char *text = NULL;
    if (row->path[0] != '\0') {
        snprintf(path, path_bytes, "%s/%s", row->path, row->file);
    } else {
        snprintf(path, path_bytes, "%s", row->file);
    }
    text = aotx_tool_module_file(path, bytes);
    if (text == NULL && aotx_tool_module_dir[0] != '\0') {
        snprintf(path, path_bytes, "%s/%s/%s", aotx_tool_module_dir, row->name, row->file);
        text = aotx_tool_module_file(path, bytes);
    }
    if (text == NULL) {
        char dir[AOTX_MODULE_ROOT_BYTES];
        if (aotx_tool_module_from_table(row->import, dir, sizeof dir) == 0) {
            snprintf(path, path_bytes, "%s/%s", dir, row->file);
            text = aotx_tool_module_file(path, bytes);
        }
    }
    return text;
}

int aotx_tool_module_digest(const char *path, unsigned char digest[32])
{
    aotx_sha256 state;
    size_t bytes = 0u;
    char *text = aotx_tool_module_file(path, &bytes);
    if (text == NULL) {
        return 1;
    }
    aotx_sha256_init(&state);
    aotx_sha256_update(&state, text, bytes);
    aotx_sha256_final(&state, digest);
    free(text);
    return 0;
}
