/* Purpose: Read the chain of turns and the requests file that the drain wrote.
 * Owns: The line buffer of one walk.
 * Threading: One thread; the reader writes the standard output.
 * Lifetime: The call. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/journal/chain.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The first line of a chain has no line before it, so its digest field is 64 zeros. */
#define AOTX_CHAIN_FIRST "0000000000000000000000000000000000000000000000000000000000000000"

/* Writes the digest of one line, with the end byte of the line in it. The drain computes
 * the digest the same way, so a reader that reads the file as bytes gets the same value. */
static void line_digest(const char *line, size_t bytes, char *out)
{
    unsigned char digest[AOTX_SHA256_DIGEST];
    aotx_sha256 state;
    aotx_sha256_init(&state);
    aotx_sha256_update(&state, line, bytes);
    aotx_sha256_final(&state, digest);
    aotx_sha256_text(digest, out);
}

/* Prints one turn. A field the line does not hold prints as a question mark, so a line
 * that another program wrote is still readable. */
static void print_turn(const char *line, uint64_t at)
{
    char finish[32];
    char tool[32];
    char input[32];
    char output[32];
    uint64_t agent = 0;
    uint64_t turn = 0;
    uint64_t tokens = 0;
    uint64_t request = 0;
    aotx_json_number(line, "\"agent\":", &agent);
    aotx_json_number(line, "\"turn\":", &turn);
    aotx_json_number(line, "\"tokens\":", &tokens);
    aotx_json_number(line, "\"request\":", &request);
    if (!aotx_json_text(line, "\"input_hash\":\"", input, sizeof(input))) {
        snprintf(input, sizeof(input), "?");
    }
    if (!aotx_json_text(line, "\"output_hash\":\"", output, sizeof(output))) {
        snprintf(output, sizeof(output), "?");
    }
    if (!aotx_json_text(line, "\"finish\":\"", finish, sizeof(finish))) {
        snprintf(finish, sizeof(finish), "?");
    }
    if (!aotx_json_text(line, "\"tool\":\"", tool, sizeof(tool))) {
        snprintf(tool, sizeof(tool), "?");
    }
    printf("line=%llu agent=%llu turn=%llu input=%s output=%s tokens=%llu finish=%s"
           " tool=%s request=%llu\n",
           (unsigned long long)at, (unsigned long long)agent, (unsigned long long)turn,
           input, output, (unsigned long long)tokens, finish, tool,
           (unsigned long long)request);
}

int aotx_chain_check(const char *path, aotx_chain_report *out)
{
    char want[65];
    char got[65];
    char *line = NULL;
    size_t cap = 0;
    ssize_t n;
    FILE *f = fopen(path, "r");
    memset(out, 0, sizeof(*out));
    if (f == NULL) {
        return -1;
    }
    snprintf(want, sizeof(want), "%s", AOTX_CHAIN_FIRST);
    while ((n = getline(&line, &cap, f)) > 0) {
        out->lines++;
        if (!aotx_json_text(line, "\"prev\":\"", got, sizeof(got))) {
            got[0] = '\0';
        }
        if (out->at == 0 && strcmp(want, got) != 0) {
            out->at = out->lines;
            snprintf(out->want, sizeof(out->want), "%s", want);
            snprintf(out->got, sizeof(out->got), "%s", got);
        }
        if (line[n - 1] != '\n' && out->at == 0) {
            /* A last line with no end byte is a line a crash cut. The chain cannot go on
             * from it, because the digest of a line holds its end byte. */
            out->at = out->lines;
            snprintf(out->want, sizeof(out->want), "a line with an end byte");
            snprintf(out->got, sizeof(out->got), "a line the disk cut");
        }
        out->turns++;
        print_turn(line, out->lines);
        line_digest(line, (size_t)n, want);
    }
    free(line);
    fclose(f);
    return (out->at == 0) ? 0 : 1;
}

int aotx_requests_print(const char *path, uint64_t *lines, uint64_t *bad)
{
    char tool[32];
    char auth[32];
    char arg[AOTX_TOOL_ARG_BYTES + 1];
    char *line = NULL;
    size_t cap = 0;
    ssize_t n;
    FILE *f = fopen(path, "r");
    *lines = 0;
    *bad = 0;
    if (f == NULL) {
        return -1;
    }
    while ((n = getline(&line, &cap, f)) > 0) {
        uint64_t request = 0;
        uint64_t agent = 0;
        uint64_t turn = 0;
        uint64_t deadline = 0;
        uint64_t tick = 0;
        (void)n;
        (*lines)++;
        if (!aotx_json_number(line, "\"request\":", &request) ||
            !aotx_json_text(line, "\"tool\":\"", tool, sizeof(tool)) ||
            !aotx_json_text(line, "\"arg\":\"", arg, sizeof(arg)) ||
            !aotx_json_text(line, "\"auth\":\"", auth, sizeof(auth))) {
            (*bad)++;
            continue;
        }
        aotx_json_number(line, "\"agent\":", &agent);
        aotx_json_number(line, "\"turn\":", &turn);
        aotx_json_number(line, "\"deadline\":", &deadline);
        aotx_json_number(line, "\"tick\":", &tick);
        printf("request=%llu agent=%llu turn=%llu tool=%s auth=%s deadline=%llu tick=%llu"
               " arg=%s\n",
               (unsigned long long)request, (unsigned long long)agent,
               (unsigned long long)turn, tool, auth, (unsigned long long)deadline,
               (unsigned long long)tick, arg);
    }
    free(line);
    fclose(f);
    return 0;
}
