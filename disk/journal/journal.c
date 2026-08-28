/* Purpose: Print the records of a journal as text, so a reader can compare two runs.
 * Owns: The block buffer of one walk.
 * Threading: One thread; the program reads files and writes the standard output.
 * Lifetime: From the start of the walk to the exit of the program. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/restore/scan.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define AOTX_BLOCK_MAX (16u * 1024u * 1024u)

typedef struct print_state {
    uint64_t tokens;   /* token records printed */
    uint64_t records;  /* records read */
    uint64_t shorts;   /* token records whose body is too short to read */
} print_state;

/* Prints one line for each token record of a block. The first four fields are the token
 * itself, so a comparison of two runs can cut the line after them. The last two fields
 * state what the run did with the token. The sampled field names the flag bit of a token
 * that the model made. The replayed field marks a token that a restore applied again.
 * Returns 0 to go on with the walk. */
static int print_block(void *ctx, const unsigned char *block, uint64_t index)
{
    print_state *s = (print_state *)ctx;
    const aotx_block_header *bh = (const aotx_block_header *)block;
    uint32_t i;
    (void)index;
    for (i = 0; i < bh->record_count; i++) {
        const aotx_record_header *h = aotx_block_record(block, i);
        aotx_token_body body;
        s->records++;
        if (h->type != AOTX_REC_TOKEN) {
            continue;
        }
        if (h->body_len < sizeof(body)) {
            s->shorts++;
            continue;
        }
        memcpy(&body, aotx_record_body(h), sizeof(body));
        printf("slot=%u position=%u token=%u flags=0x%04x"
               " seed=%016llx draw=%llu role=%u tick=%llu seq=%llu sampled=%d replayed=%d\n",
               body.slot, body.position, body.token, body.flags,
               (unsigned long long)body.seed, (unsigned long long)body.draw, body.role,
               (unsigned long long)h->tick, (unsigned long long)h->seq,
               ((body.flags & AOTX_TOKEN_SAMPLED) != 0) ? 1 : 0,
               ((h->flags & AOTX_FLAG_REPLAYED) != 0) ? 1 : 0);
        s->tokens++;
    }
    return 0;
}

static void usage(void)
{
    fprintf(stderr, "usage: aotx_journal tokens <dir> [--boot <id>]\n");
    fprintf(stderr, "  <dir>   a boot directory, or a journal directory that holds boot"
                    " directories\n");
    fprintf(stderr, "  --boot  the boot identity, 16 hexadecimal digits; with no identity"
                    " the newest boot is read\n");
}

/* Gives the directory to walk. A directory that holds segments is a boot directory and is
 * read as it stands. A directory that holds none is a journal, and the boot identity, or
 * the newest complete boot, names the directory in it. Returns 0, or an exit code. */
static int choose_dir(const char *dir, const char *boot, unsigned char *buffer, char *out,
                      size_t out_bytes)
{
    char names[1][AOTX_NAME_BYTES];
    aotx_journal_scan scan;
    if (aotx_segment_list(dir, names, 1) > 0) {
        snprintf(out, out_bytes, "%s", dir);
        return 0;
    }
    if (boot != NULL) {
        snprintf(out, out_bytes, "%s/%s", dir, boot);
        if (aotx_segment_list(out, names, 1) <= 0) {
            fprintf(stderr, "journal: the boot %s in %s holds no segment\n", boot, dir);
            return AOTX_EXIT_NOJOURNAL;
        }
        return 0;
    }
    if (aotx_journal_latest(dir, buffer, AOTX_BLOCK_MAX, &scan) != 0) {
        fprintf(stderr, "journal: no journal in %s holds a complete tick\n", dir);
        return AOTX_EXIT_NOJOURNAL;
    }
    snprintf(out, out_bytes, "%s", scan.dir);
    return 0;
}

int main(int argc, char **argv)
{
    print_state s;
    unsigned char *buffer;
    char dir[AOTX_PATH_BYTES];
    const char *boot = NULL;
    uint64_t blocks = 0;
    int torn = 0;
    int status;
    int i;

    if (argc < 3 || strcmp(argv[1], "tokens") != 0) {
        usage();
        return AOTX_EXIT_FAULT;
    }
    for (i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--boot") == 0 && i + 1 < argc) {
            boot = argv[++i];
        } else {
            usage();
            return AOTX_EXIT_FAULT;
        }
    }

    memset(&s, 0, sizeof(s));
    buffer = (unsigned char *)malloc(AOTX_BLOCK_MAX);
    if (buffer == NULL) {
        fprintf(stderr, "journal: the block buffer does not fit in memory\n");
        return AOTX_EXIT_FAULT;
    }
    status = choose_dir(argv[2], boot, buffer, dir, sizeof(dir));
    if (status != 0) {
        free(buffer);
        return status;
    }
    if (aotx_journal_walk(dir, buffer, AOTX_BLOCK_MAX, print_block, &s, &blocks, &torn) != 0) {
        fprintf(stderr, "journal: the walk of %s did not finish\n", dir);
        free(buffer);
        return AOTX_EXIT_FAULT;
    }
    fflush(stdout);
    fprintf(stderr, "journal: %s blocks %llu records %llu tokens %llu short %llu torn %d\n",
            dir, (unsigned long long)blocks, (unsigned long long)s.records,
            (unsigned long long)s.tokens, (unsigned long long)s.shorts, torn);
    free(buffer);
    return AOTX_EXIT_OK;
}
