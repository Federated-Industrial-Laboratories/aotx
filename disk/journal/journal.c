/* Purpose: Print the records of a journal as text, so a reader can compare two runs.
 * Owns: The block buffer of one walk.
 * Threading: One thread; the program reads files and writes the standard output.
 * Lifetime: From the start of the walk to the exit of the program. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/journal/chain.h"
#include "disk/restore/scan.h"
#include "disk/settings/settings.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define AOTX_BLOCK_MAX (16u * 1024u * 1024u)

/* 64 hexadecimal characters and one end byte. */
#define AOTX_HEX_TEXT 65

/* Gives the name of a module kind. */
static const char *kind_name(uint32_t kind)
{
    static const char *names[3] = { "skill", "role", "tool" };
    return (kind >= 1u && kind <= 3u) ? names[kind - 1u] : "other";
}

/* The record type that one walk prints. */
#define AOTX_PRINT_TOKENS   0
#define AOTX_PRINT_SETTINGS 1
#define AOTX_PRINT_MODULES  2

typedef struct print_state {
    uint64_t tokens;    /* token records printed */
    uint64_t settings;  /* setting records printed */
    uint64_t modules;   /* import heads and remove records printed */
    uint64_t records;   /* records read */
    uint64_t shorts;    /* records whose body is too short to read */
    uint64_t strays;    /* parts that name an import which is not the open one */
    int      mode;      /* the record type that the walk prints */
    int      open;      /* one while a head waits for the parts that belong to it */
    uint32_t parts;     /* parts counted under the open head */
    aotx_import_head head; /* the open head */
    uint64_t head_tick;
    uint64_t head_seq;
    uint16_t head_flags;
} print_state;

/* Gives the name of the record type that one mode prints. */
static const char *aotx_print_name(int mode)
{
    static const char *names[3] = { "tokens", "settings", "modules" };
    return (mode >= 0 && mode <= 2) ? names[mode] : "records";
}

/* Gives the count of the records that one mode printed. */
static uint64_t aotx_print_count(const print_state *s)
{
    if (s->mode == AOTX_PRINT_SETTINGS) {
        return s->settings;
    }
    return (s->mode == AOTX_PRINT_MODULES) ? s->modules : s->tokens;
}

/* Writes a text of a record with every byte that a terminal acts on made a mark. Every
 * field that comes from a record goes through this before it reaches the output. */
static void safe_text(const char *in, size_t in_bytes, char *out)
{
    size_t i;
    for (i = 0; i + 1u < in_bytes && in[i] != '\0'; i++) {
        unsigned char b = (unsigned char)in[i];
        out[i] = (b < 0x20u || b >= 0x7fu) ? '?' : (char)b;
    }
    out[i] = '\0';
}

/* Prints one line for one token record. The first four fields are the token itself, so a
 * comparison of two runs can cut the line after them. The last two fields state what the
 * run did with the token. The sampled field names the flag bit of a token that the model
 * made. The replayed field marks a token that a restore applied again. */
static void print_token(const aotx_record_header *h, print_state *s)
{
    aotx_token_body body;
    if (h->body_len < sizeof(body)) {
        s->shorts++;
        return;
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

/* Prints one line for one setting record. The line names the tick, the writer, the key and
 * the value in the unit the operator writes. The replayed field marks a record that a
 * restore applied again. A reader can then prove the order of a restored run. */
static void print_setting(const aotx_record_header *h, print_state *s)
{
    aotx_setting_body body;
    char key[AOTX_SETTING_WIRE_KEY_BYTES + 1];
    char value[32];
    uint32_t len;
    uint32_t i;
    if (h->body_len < sizeof(body)) {
        s->shorts++;
        return;
    }
    memcpy(&body, aotx_record_body(h), sizeof(body));
    len = body.key_len;
    if (len > AOTX_SETTING_WIRE_KEY_BYTES) {
        len = AOTX_SETTING_WIRE_KEY_BYTES;
    }
    for (i = 0; i < len; i++) {
        unsigned char b = (unsigned char)body.key[i];
        /* The key comes from a record, so a byte that a terminal acts on becomes a mark. */
        key[i] = (b < 0x20u || b >= 0x7fu) ? '?' : (char)b;
    }
    key[len] = '\0';
    aotx_settings_format(body.value, (int)body.scale, value, sizeof(value));
    printf("tick=%llu writer=%u key=%s value=%s seq=%llu replayed=%d\n",
           (unsigned long long)h->tick, h->writer, key, value,
           (unsigned long long)h->seq, ((h->flags & AOTX_FLAG_REPLAYED) != 0) ? 1 : 0);
    s->settings++;
}

/* Prints the head that waits, with the count of the parts that came after it. The parts of
 * one import follow their head in the journal, so one open head is enough. A part that
 * names another import is counted as a stray. */
static void flush_head(print_state *s)
{
    char digest[AOTX_HEX_TEXT];
    char name[AOTX_IMPORT_NAME_BYTES];
    char path[AOTX_IMPORT_PATH_BYTES];
    if (s->open == 0) {
        return;
    }
    aotx_sha256_text(s->head.digest, digest);
    safe_text(s->head.name, sizeof(s->head.name), name);
    safe_text(s->head.path, sizeof(s->head.path), path);
    printf("tick=%llu import=%u kind=%s name=%s files=%u bytes=%llu parts=%u digest=%s"
           " path=%s seq=%llu replayed=%d\n",
           (unsigned long long)s->head_tick, s->head.import, kind_name(s->head.kind),
           name, s->head.files,
           (unsigned long long)((uint64_t)s->head.file_bytes[0] + s->head.file_bytes[1]),
           s->parts, digest, path, (unsigned long long)s->head_seq,
           ((s->head_flags & AOTX_FLAG_REPLAYED) != 0) ? 1 : 0);
    s->open = 0;
    s->parts = 0;
    s->modules++;
}

/* Takes one import record. A head closes the head before it and opens its own; a part
 * counts under the open head. */
static void print_import(const aotx_record_header *h, print_state *s)
{
    aotx_import_head head;
    if (h->body_len < sizeof(head)) {
        s->shorts++;
        return;
    }
    memcpy(&head, aotx_record_body(h), sizeof(head));
    if (head.part != 0) {
        if (s->open != 0 && head.import == s->head.import) {
            s->parts++;
        } else {
            s->strays++;
        }
        return;
    }
    flush_head(s);
    s->head = head;
    s->head_tick = h->tick;
    s->head_seq = h->seq;
    s->head_flags = h->flags;
    s->parts = 0;
    s->open = 1;
}

/* Prints one line for a module that leaves the catalog. */
static void print_remove(const aotx_record_header *h, print_state *s)
{
    aotx_remove_body gone;
    char name[AOTX_IMPORT_NAME_BYTES];
    if (h->body_len < sizeof(gone)) {
        s->shorts++;
        return;
    }
    flush_head(s);
    memcpy(&gone, aotx_record_body(h), sizeof(gone));
    safe_text(gone.name, sizeof(gone.name), name);
    printf("tick=%llu removed name=%s seq=%llu replayed=%d\n",
           (unsigned long long)h->tick, name, (unsigned long long)h->seq,
           ((h->flags & AOTX_FLAG_REPLAYED) != 0) ? 1 : 0);
    s->modules++;
}

/* Prints one line for each record of a block that the mode names. Returns 0 to go on with
 * the walk. */
static int print_block(void *ctx, const unsigned char *block, uint64_t index)
{
    print_state *s = (print_state *)ctx;
    const aotx_block_header *bh = (const aotx_block_header *)block;
    uint32_t i;
    (void)index;
    for (i = 0; i < bh->record_count; i++) {
        const aotx_record_header *h = aotx_block_record(block, i);
        s->records++;
        if (s->mode == AOTX_PRINT_SETTINGS) {
            if (h->type == AOTX_REC_SETTING) {
                print_setting(h, s);
            }
        } else if (s->mode == AOTX_PRINT_MODULES) {
            if (h->type == AOTX_REC_IMPORT) {
                print_import(h, s);
            } else if (h->type == AOTX_REC_REMOVE) {
                print_remove(h, s);
            }
        } else if (h->type == AOTX_REC_TOKEN) {
            print_token(h, s);
        }
    }
    return 0;
}

static void usage(void)
{
    fprintf(stderr, "usage: aotx_journal tokens|settings|modules|manifest|requests <dir>"
                    " [--boot <id>]\n");
    fprintf(stderr, "  tokens    the token records of a run\n");
    fprintf(stderr, "  settings  the setting records of a run\n");
    fprintf(stderr, "  modules   the modules that the run imported and removed\n");
    fprintf(stderr, "  manifest  the turns of a run, with the digest chain verified\n");
    fprintf(stderr, "  requests  the tool requests of a journal\n");
    fprintf(stderr, "  <dir>   a boot directory, or a journal directory that holds boot"
                    " directories\n");
    fprintf(stderr, "  --boot  the boot identity, 16 hexadecimal digits; with no identity"
                    " the newest boot is read\n");
}

/* Reports whether a path names a directory. */
static int is_dir(const char *path)
{
    struct stat info;
    return (stat(path, &info) == 0 && S_ISDIR(info.st_mode)) ? 1 : 0;
}

/* Gives the directory that holds the chain files. A journal holds them in a directory of
 * its own; a caller that names that directory gets it back. */
static void chain_dir(const char *dir, char *out, size_t out_bytes)
{
    char with[AOTX_PATH_BYTES + 16];
    snprintf(with, sizeof(with), "%s/manifest", dir);
    snprintf(out, out_bytes, "%s", is_dir(with) ? with : dir);
}

/* Verifies one chain file and reports it. Returns 0 when the chain holds, or 1. */
static int check_one(const char *dir, const char *name)
{
    aotx_chain_report report;
    char path[AOTX_PATH_BYTES + 96];
    int status;
    snprintf(path, sizeof(path), "%.500s/%.80s", dir, name);
    status = aotx_chain_check(path, &report);
    if (status < 0) {
        fprintf(stderr, "journal: the chain file %s does not read\n", path);
        return 1;
    }
    if (status == 0) {
        fprintf(stderr, "journal: %s turns %llu chain holds\n", path,
                (unsigned long long)report.turns);
        return 0;
    }
    fprintf(stderr, "journal: %s turns %llu chain breaks at line %llu, the line names %s"
                    " and the line before it gives %s\n",
            path, (unsigned long long)report.turns, (unsigned long long)report.at,
            (report.got[0] != '\0') ? report.got : "no digest", report.want);
    return 1;
}

/* Verifies the chain of one boot, or of every boot the directory holds. Returns an exit
 * code. */
static int run_manifest(const char *dir, const char *boot)
{
    char where[AOTX_PATH_BYTES + 16];
    struct dirent **found = NULL;
    int broken = 0;
    int files = 0;
    int total;
    int i;
    chain_dir(dir, where, sizeof(where));
    if (boot != NULL) {
        char name[80];
        snprintf(name, sizeof(name), "%.60s.jsonl", boot);
        return (check_one(where, name) == 0) ? AOTX_EXIT_OK : AOTX_EXIT_FAULT;
    }
    total = scandir(where, &found, NULL, alphasort);
    if (total < 0) {
        fprintf(stderr, "journal: the directory %s does not read\n", where);
        return AOTX_EXIT_NOJOURNAL;
    }
    for (i = 0; i < total; i++) {
        size_t len = strlen(found[i]->d_name);
        if (len > 6 && strcmp(found[i]->d_name + len - 6, ".jsonl") == 0 && len < 72) {
            broken += check_one(where, found[i]->d_name);
            files++;
        }
        free(found[i]);
    }
    free(found);
    if (files == 0) {
        fprintf(stderr, "journal: no chain file is in %s\n", where);
        return AOTX_EXIT_NOJOURNAL;
    }
    fprintf(stderr, "journal: chain files %d, broken %d\n", files, broken);
    return (broken == 0) ? AOTX_EXIT_OK : AOTX_EXIT_FAULT;
}

/* Prints the requests of a journal. Returns an exit code. */
static int run_requests(const char *dir)
{
    char path[AOTX_PATH_BYTES + 32];
    uint64_t lines = 0;
    uint64_t bad = 0;
    snprintf(path, sizeof(path), "%s/requests.jsonl", dir);
    if (!is_dir(dir)) {
        snprintf(path, sizeof(path), "%s", dir);
    }
    if (aotx_requests_print(path, &lines, &bad) != 0) {
        fprintf(stderr, "journal: the requests file %s does not read\n", path);
        return AOTX_EXIT_NOJOURNAL;
    }
    fflush(stdout);
    fprintf(stderr, "journal: %s requests %llu, lines not read %llu\n", path,
            (unsigned long long)lines, (unsigned long long)bad);
    return (bad == 0) ? AOTX_EXIT_OK : AOTX_EXIT_FAULT;
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

    if (argc < 3) {
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
    if (strcmp(argv[1], "manifest") == 0) {
        return run_manifest(argv[2], boot);
    }
    if (strcmp(argv[1], "requests") == 0) {
        return run_requests(argv[2]);
    }
    if (strcmp(argv[1], "tokens") != 0 && strcmp(argv[1], "settings") != 0 &&
        strcmp(argv[1], "modules") != 0) {
        usage();
        return AOTX_EXIT_FAULT;
    }

    memset(&s, 0, sizeof(s));
    if (strcmp(argv[1], "settings") == 0) {
        s.mode = AOTX_PRINT_SETTINGS;
    } else if (strcmp(argv[1], "modules") == 0) {
        s.mode = AOTX_PRINT_MODULES;
    } else {
        s.mode = AOTX_PRINT_TOKENS;
    }
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
    /* The last head of the walk holds no part after it, so the walk closes it here. */
    flush_head(&s);
    fflush(stdout);
    fprintf(stderr, "journal: %s blocks %llu records %llu %s %llu short %llu stray %llu"
                    " torn %d\n",
            dir, (unsigned long long)blocks, (unsigned long long)s.records,
            aotx_print_name(s.mode), (unsigned long long)aotx_print_count(&s),
            (unsigned long long)s.shorts, (unsigned long long)s.strays, torn);
    free(buffer);
    return AOTX_EXIT_OK;
}
