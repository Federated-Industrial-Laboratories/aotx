/* Purpose: Read the journal segments of a boot and find the last complete tick.
 * Owns: Nothing; the caller holds the result structure and the block buffer.
 * Threading: One thread; the scan reads files and holds no state between calls.
 * Lifetime: The call. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/restore/scan.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define AOTX_SEGMENT_MAX 4096

int aotx_journal_walk(const char *dir, unsigned char *buffer, uint32_t bytes,
                      aotx_block_fn fn, void *ctx, uint64_t *blocks, int *torn)
{
    char (*names)[AOTX_NAME_BYTES];
    char path[AOTX_PATH_BYTES + 32];
    int count;
    int i;
    int rc = 0;
    int done = 0;
    *blocks = 0;
    *torn = 0;
    names = (char (*)[AOTX_NAME_BYTES])malloc((size_t)AOTX_SEGMENT_MAX * AOTX_NAME_BYTES);
    if (names == NULL) {
        return -1;
    }
    count = aotx_segment_list(dir, names, AOTX_SEGMENT_MAX);
    if (count < 0) {
        free(names);
        return -1;
    }
    for (i = 0; i < count && *torn == 0 && rc == 0 && done == 0; i++) {
        aotx_segment_reader reader;
        snprintf(path, sizeof(path), "%s/%s", dir, names[i]);
        if (aotx_segment_reader_open(&reader, path) != 0) {
            rc = -1;
            break;
        }
        for (;;) {
            uint32_t got = 0;
            int frame = aotx_segment_get(&reader, buffer, bytes, &got);
            const char *reason = "";
            if (frame == AOTX_FRAME_END) {
                break;
            }
            if (frame == AOTX_FRAME_TORN) {
                /* A frame that fails the checksum, or a frame that the file cuts short,
                 * ends the part of the journal that can be read. */
                *torn = 1;
                break;
            }
            if (aotx_block_valid(buffer, got, &reason) != 0) {
                *torn = 1;
                break;
            }
            if (fn != NULL) {
                int step = fn(ctx, buffer, *blocks);
                if (step < 0) {
                    rc = -1;
                    break;
                }
                if (step > 0) {
                    (*blocks)++;
                    done = 1;
                    break;
                }
            }
            (*blocks)++;
        }
        aotx_segment_reader_close(&reader);
    }
    free(names);
    return rc;
}

static int read_block(void *ctx, const unsigned char *block, uint64_t index)
{
    aotx_journal_scan *s = (aotx_journal_scan *)ctx;
    const aotx_block_header *bh = (const aotx_block_header *)block;
    uint32_t i;
    for (i = 0; i < bh->record_count; i++) {
        const aotx_record_header *h = aotx_block_record(block, i);
        const unsigned char *body = aotx_record_body(h);
        if (h->type == AOTX_REC_BOOT && h->body_len >= sizeof(aotx_boot_body)) {
            aotx_boot_body boot;
            memcpy(&boot, body, sizeof(boot));
            s->boot_wall_ns = boot.wall_ns;
        } else if (h->type == AOTX_REC_RESTORE && h->body_len >= sizeof(aotx_restore_body)) {
            aotx_restore_body r;
            memcpy(&r, body, sizeof(r));
            s->restore_hash = r.state_hash;
            s->restore_of = r.restored_boot_id;
            s->has_restore = 1;
        }
    }
    if (bh->record_count > 0) {
        const aotx_record_header *last = aotx_block_record(block, bh->record_count - 1);
        if (last->type == AOTX_REC_TICK_COMMIT && last->body_len >= sizeof(aotx_commit_body)) {
            aotx_commit_body commit;
            memcpy(&commit, aotx_record_body(last), sizeof(commit));
            s->commits++;
            s->last_tick = bh->tick;
            s->last_block = index;
            s->state_hash = commit.state_hash;
        }
    }
    return 0;
}

int aotx_journal_read(const char *dir, uint64_t boot_id, unsigned char *buffer, uint32_t bytes,
                      aotx_journal_scan *out)
{
    memset(out, 0, sizeof(*out));
    out->boot_id = boot_id;
    if (strlen(dir) + 1 > AOTX_PATH_BYTES) {
        return -1;
    }
    memcpy(out->dir, dir, strlen(dir) + 1);
    return aotx_journal_walk(dir, buffer, bytes, read_block, out, &out->blocks, &out->torn);
}

/* A boot directory name is the boot identity in sixteen hexadecimal digits. */
static int boot_name(const char *name, uint64_t *boot_id)
{
    int i;
    if (strlen(name) != 16) {
        return 0;
    }
    for (i = 0; i < 16; i++) {
        char c = name[i];
        int ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        if (!ok) {
            return 0;
        }
    }
    *boot_id = strtoull(name, NULL, 16);
    return 1;
}

int aotx_journal_latest(const char *journal, unsigned char *buffer, uint32_t bytes,
                        aotx_journal_scan *out)
{
    struct dirent **found = NULL;
    int total;
    int i;
    int have = 0;
    total = scandir(journal, &found, NULL, alphasort);
    if (total < 0) {
        return -1;
    }
    for (i = 0; i < total; i++) {
        aotx_journal_scan one;
        struct stat st;
        char path[AOTX_PATH_BYTES + 32];
        uint64_t boot_id = 0;
        if (!boot_name(found[i]->d_name, &boot_id)) {
            free(found[i]);
            continue;
        }
        snprintf(path, sizeof(path), "%s/%s", journal, found[i]->d_name);
        free(found[i]);
        if (aotx_journal_read(path, boot_id, buffer, bytes, &one) != 0 || one.commits == 0) {
            continue;
        }
        /* The boot record wall clock orders the boots. A boot with no boot record falls
         * back to the change time of its directory. */
        one.rank = one.boot_wall_ns;
        if (one.rank == 0 && stat(path, &st) == 0) {
            one.rank = (uint64_t)st.st_mtime * 1000000000u;
        }
        if (!have || one.rank > out->rank) {
            *out = one;
            have = 1;
        }
    }
    for (; i < total; i++) {
        free(found[i]);
    }
    free(found);
    return have ? 0 : -1;
}
