/* Purpose: Read one module directory and publish it as one import of class A records.
 * Owns: The bytes of the two files under read, and the import count of one run.
 * Threading: One thread; the feeder is the only caller and the only producer of the ring.
 * Lifetime: From the first import to the close of the feeder. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/feed/import.h"
#include "disk/feed/modules.h"

#include <dirent.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* The buffer that the digest of a module file reads through. The digest holds no file in
 * memory, so a large module file costs the feeder these bytes and no more. */
#define AOTX_IMPORT_DIGEST_BUFFER 4096u

/* Reports whether the name is one to sixty-three bytes of lower case letters, digits and
 * underscores. The name is the identity of the module in the catalog. */
static int name_ok(const char *name)
{
    size_t len = strlen(name);
    size_t i;
    if (len < 1u || len > (size_t)(AOTX_IMPORT_NAME_BYTES - 1)) {
        return 0;
    }
    for (i = 0; i < len; i++) {
        char c = name[i];
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_')) {
            return 0;
        }
    }
    return 1;
}

/* Writes the last component of a path. A path that ends with one or more slashes gives the
 * component before them, so a path with an end slash names the same module. The buffer is
 * wider than a name of the catalog. A name that is too long thus stays too long here, and
 * the name rule refuses it. */
static void last_component(const char *path, char *out, size_t out_bytes)
{
    size_t end = strlen(path);
    size_t start;
    while (end > 0 && path[end - 1u] == '/') {
        end--;
    }
    start = end;
    while (start > 0 && path[start - 1u] != '/') {
        start--;
    }
    if (end - start >= out_bytes) {
        end = start + out_bytes - 1u;
    }
    memcpy(out, path + start, end - start);
    out[end - start] = '\0';
}

/* Writes the tail of a path, which the console shows beside the name of the module. The
 * input is the whole path with every dot and every link taken out, so the head carries an
 * absolute path whenever one fits. The glue of a restore opens the module file of a device
 * tool from that path. A longer absolute path keeps its tail only. */
static void path_tail(const char *path, char *out, size_t out_bytes)
{
    char whole[PATH_MAX];
    const char *at = (realpath(path, whole) != NULL) ? whole : path;
    size_t len = strlen(at);
    size_t take = out_bytes - 1u;
    snprintf(out, out_bytes, "%s", (len > take) ? at + (len - take) : at);
}

/* Gives the value of one key of a manifest. A manifest holds one key and one value a
 * line, with the key in lower case and the value to the end of the line. There is no
 * quotation and no escape. A line whose first byte after the spaces is a number sign is a
 * comment. A key that comes more than one time gives the value of its last line, which is
 * the rule of the settings reader. Returns 1 when the key is there. */
static int manifest_value(const unsigned char *text, uint32_t len, const char *key,
                          char *out, size_t out_bytes)
{
    size_t key_len = strlen(key);
    uint32_t at = 0;
    int found = 0;
    while (at < len) {
        uint32_t end = at;
        uint32_t first;
        uint32_t colon;
        uint32_t value;
        uint32_t stop;
        while (end < len && text[end] != '\n') {
            end++;
        }
        first = at;
        while (first < end && (text[first] == ' ' || text[first] == '\t')) {
            first++;
        }
        colon = first;
        while (colon < end && text[colon] != ':') {
            colon++;
        }
        if (first < end && text[first] != '#' && colon < end) {
            uint32_t key_end = colon;
            while (key_end > first && (text[key_end - 1u] == ' ' || text[key_end - 1u] == '\t')) {
                key_end--;
            }
            if ((size_t)(key_end - first) == key_len &&
                memcmp(text + first, key, key_len) == 0) {
                value = colon + 1u;
                while (value < end && (text[value] == ' ' || text[value] == '\t')) {
                    value++;
                }
                stop = end;
                while (stop > value && (text[stop - 1u] == ' ' || text[stop - 1u] == '\t' ||
                                        text[stop - 1u] == '\r')) {
                    stop--;
                }
                if ((size_t)(stop - value) < out_bytes) {
                    memcpy(out, text + value, stop - value);
                    out[stop - value] = '\0';
                    found = 1;
                }
            }
        }
        at = end + 1u;
    }
    return found;
}

/* Gives the kind of a module, or zero when the word names none. */
static uint32_t kind_of(const char *word)
{
    if (strcmp(word, "skill") == 0) {
        return AOTX_MODULE_SKILL;
    }
    if (strcmp(word, "role") == 0) {
        return AOTX_MODULE_ROLE;
    }
    if (strcmp(word, "tool") == 0) {
        return AOTX_MODULE_TOOL;
    }
    return 0;
}

/* Opens one file of a module directory. The file stays under the directory. A name that
 * starts at the root of the file system is refused. A name that holds a component of two
 * dots is refused. Returns the descriptor, or -1. */
static int open_in(int dir_fd, const char *name, const char **reason)
{
    aotx_walk walk;
    uint32_t status = 0;
    if (name[0] == '/') {
        *reason = "a file of the manifest starts at the root of the file system";
        return -1;
    }
    return aotx_path_walk(&walk, dir_fd, name, AOTX_WALK_NO_UP, &status, reason);
}

/* Opens the module directory. A path that starts at the root of the file system walks from
 * that root, and every other path walks from the working directory. The walk refuses a
 * symbolic link at any component. The operator names this directory, so no root is above
 * it and a component of two dots is permitted here. Returns the descriptor. */
static int open_module_dir(const char *path, const char **reason)
{
    aotx_walk walk;
    uint32_t status = 0;
    int absolute = (path[0] == '/');
    int base_fd = open(absolute ? "/" : ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    int fd;
    if (base_fd < 0) {
        *reason = "the directory the path starts at does not open";
        return -1;
    }
    fd = aotx_path_walk(&walk, base_fd, absolute ? path + 1 : path, AOTX_WALK_DIR, &status,
                        reason);
    close(base_fd);
    return fd;
}

/* Reads one whole file into the buffer. Returns the byte count, -1 when the file does not
 * read, or -2 when the file is longer than the bound of the feeder. */
static long read_file(int fd, unsigned char *out)
{
    uint32_t got = 0;
    unsigned char more;
    while (got < AOTX_IMPORT_CAP) {
        ssize_t n = read(fd, out + got, (size_t)(AOTX_IMPORT_CAP - got));
        if (n < 0) {
            return -1;
        }
        if (n == 0) {
            return (long)got;
        }
        got += (uint32_t)n;
    }
    /* One more byte proves that the file is longer than the bound and not equal to it. */
    return (read(fd, &more, 1) > 0) ? -2 : (long)got;
}

/* The reason of a refusal that names one file. The text lives beside the program, because
 * a caller reads the reason after this function returns. */
static char refuse_text[128];

/* Reads one open file into the buffer of one file of the import, and closes it. Returns 0,
 * or 1 with the reason. */
static int read_into(int fd, const char *what, unsigned char *out, uint32_t *bytes,
                     const char **reason)
{
    long got = read_file(fd, out);
    close(fd);
    if (got == -2) {
        /* The bound comes from the constant, so the reason cannot state a figure that the
         * bound does not hold. */
        snprintf(refuse_text, sizeof(refuse_text),
                 "the %s file is longer than the bound of %u bytes", what,
                 (unsigned)AOTX_IMPORT_CAP);
        *reason = refuse_text;
        return 1;
    }
    if (got < 0) {
        snprintf(refuse_text, sizeof(refuse_text), "the %s file does not read", what);
        *reason = refuse_text;
        return 1;
    }
    *bytes = (uint32_t)got;
    return 0;
}

/* Opens one file of the module directory and reads it. The reason of the walk names the
 * cause, so a link and a file that is not there do not read the same. Returns 0, or 1. */
static int take_file(int dir_fd, const char *name, const char *what, unsigned char *out,
                     uint32_t *bytes, const char **reason)
{
    const char *cause = "";
    int fd = open_in(dir_fd, name, &cause);
    if (fd < 0) {
        /* A file that is not there is the common cause, and it makes a whole clause of its
         * own. Every other cause comes after the name of the file it belongs to. */
        if (cause == aotx_walk_absent) {
            snprintf(refuse_text, sizeof(refuse_text), "the %s file is not there", what);
        } else {
            snprintf(refuse_text, sizeof(refuse_text), "the %s file: %.80s", what, cause);
        }
        *reason = refuse_text;
        return 1;
    }
    return read_into(fd, what, out, bytes, reason);
}

/* Computes the digest of the module file of a device tool. The bytes of that file are not
 * published. The driver loads the code from the file and the head carries the digest. A
 * restore can then refuse a module file that changed. Returns 0, or 1 with the reason. */
static int take_digest(int dir_fd, const char *name, unsigned char digest[32],
                       const char **reason)
{
    unsigned char buffer[AOTX_IMPORT_DIGEST_BUFFER];
    aotx_sha256 state;
    struct stat info;
    const char *cause = "";
    int fd = open_in(dir_fd, name, &cause);
    if (fd < 0) {
        if (cause == aotx_walk_absent) {
            *reason = "the module file is not there";
        } else {
            snprintf(refuse_text, sizeof(refuse_text), "the module file: %.80s", cause);
            *reason = refuse_text;
        }
        return 1;
    }
    if (fstat(fd, &info) != 0 || !S_ISREG(info.st_mode)) {
        close(fd);
        *reason = "the module file is not a regular file";
        return 1;
    }
    aotx_sha256_init(&state);
    if (aotx_sha256_read(fd, 0, (uint64_t)info.st_size, buffer, sizeof(buffer), &state) != 0) {
        close(fd);
        *reason = "the module file does not read";
        return 1;
    }
    close(fd);
    aotx_sha256_final(&state, digest);
    return 0;
}

/* Fills the head of one import from the files of the module directory. Returns 0, or 1
 * with the reason. */
static int build_head(aotx_import *s, int dir_fd, aotx_import_head *head, uint32_t *bytes,
                      const char **reason)
{
    char word[64];
    char file_name[AOTX_WALK_BYTES];
    const char *cause = "";
    int fd = open_in(dir_fd, AOTX_IMPORT_MANIFEST, &cause);
    if (fd < 0 && cause != aotx_walk_absent) {
        /* A manifest that is there and does not open names its own defect. Only a manifest
         * that is not there sends the reader to the skill file. */
        *reason = cause;
        return 1;
    }
    if (fd < 0) {
        /* A skill directory of the open shape holds a skill file and no manifest. The head
         * states file 0 empty and file 1 the skill file, and the device reads the head of
         * two keys that the file carries. */
        fd = open_in(dir_fd, AOTX_IMPORT_SKILL, &cause);
        if (fd < 0) {
            /* A skill file that is not there is the one case that names the directory. A
             * link, or a file that does not open, names itself. */
            *reason = (cause == aotx_walk_absent)
                      ? "the directory holds no manifest and no skill file" : cause;
            return 1;
        }
        if (read_into(fd, "skill", s->bytes[1], &bytes[1], reason) != 0) {
            return 1;
        }
        head->kind = AOTX_MODULE_SKILL;
        return 0;
    }
    if (read_into(fd, "manifest", s->bytes[0], &bytes[0], reason) != 0) {
        return 1;
    }
    if (!manifest_value(s->bytes[0], bytes[0], "kind", word, sizeof(word))) {
        *reason = "the manifest names no kind";
        return 1;
    }
    head->kind = kind_of(word);
    if (head->kind == 0) {
        *reason = "the manifest names a kind that is not a skill, a role or a tool";
        return 1;
    }
    /* The body key names the text of a skill or of a role. A manifest with no body key
     * publishes the manifest alone, and the device states what it needs and does not get. */
    if (manifest_value(s->bytes[0], bytes[0], "body", file_name, sizeof(file_name)) &&
        take_file(dir_fd, file_name, "body", s->bytes[1], &bytes[1], reason) != 0) {
        return 1;
    }
    if (head->kind == AOTX_MODULE_TOOL &&
        manifest_value(s->bytes[0], bytes[0], "side", word, sizeof(word)) &&
        strcmp(word, "device") == 0) {
        if (!manifest_value(s->bytes[0], bytes[0], "module", file_name, sizeof(file_name))) {
            *reason = "the manifest of a device tool names no module file";
            return 1;
        }
        if (take_digest(dir_fd, file_name, head->digest, reason) != 0) {
            return 1;
        }
    }
    return 0;
}

/* Gives the seconds of a whole number value, or zero when the text names none. */
static uint32_t seconds_of(const char *text)
{
    uint32_t value = 0;
    size_t i;
    for (i = 0; text[i] != '\0'; i++) {
        if (text[i] < '0' || text[i] > '9' || value > 100000u) {
            return 0;
        }
        value = value * 10u + (uint32_t)(text[i] - '0');
    }
    return value;
}

/* Appends one line of the module table for one import. The file takes a line for every
 * kind. The highest import number of the journal thus stands in it, and no two modules of
 * one journal take one number.
 *
 * The table in memory takes a row for a tool of side host alone. That row gives the feeder
 * the directory, the program and the timeout. A request for that tool then runs the
 * program the operator installed.
 *
 * The manifest key of the authorization keeps the spelling the device reader takes, since
 * the two must read one file. */
static void note_import(aotx_import *s, const aotx_import_head *head, const char *path,
                        uint32_t manifest_bytes)
{
    static const char *const kinds[4] = { "none", "skill", "role", "tool" };
    char word[32];
    char program[AOTX_MODULE_PROGRAM];
    char full[PATH_MAX];
    const char *side = "none";
    uint32_t timeout = 0;
    uint32_t authorize = 0;
    if (s->table == NULL) {
        return;
    }
    program[0] = '\0';
    if (head->kind == AOTX_MODULE_TOOL) {
        /* The device reads a tool with no side key as a device tool. The feeder reads it
         * the same way, so the two hold one rule. */
        side = "device";
        if (manifest_value(s->bytes[0], manifest_bytes, "side", word, sizeof(word)) &&
            strcmp(word, "host") == 0) {
            side = "host";
        }
        if (manifest_value(s->bytes[0], manifest_bytes, "program", program,
                           sizeof(program)) == 0) {
            /* A host tool with no program still takes a row. A call to it then states the
             * cause, and does not state that the run holds no such tool. */
            program[0] = '\0';
        }
        if (manifest_value(s->bytes[0], manifest_bytes, "timeout", word, sizeof(word))) {
            timeout = seconds_of(word);
        }
        if (manifest_value(s->bytes[0], manifest_bytes, "authorise", word, sizeof(word))) {
            authorize = (strcmp(word, "always") == 0) ? 1u : 0u;
        }
    }
    if (realpath(path, full) == NULL) {
        return;
    }
    aotx_modules_add(s->table, head->import, head->name,
                     kinds[(head->kind <= 3u) ? head->kind : 0u], side, full, program,
                     timeout, authorize);
}

/* Publishes one record of an import. Returns 0, or -1 when the ring closed or the stop
 * flag went to one. */
static int put(aotx_import *s, const aotx_inbound_ring *ring,
               const volatile sig_atomic_t *stop, const void *body, uint32_t len)
{
    aotx_record_header h;
    memset(&h, 0, sizeof(h));
    /* The inbound preamble carries no boot identity, so the field stays zero. The device
     * stamps its own boot identity when it writes the record to the journal. */
    h.writer = AOTX_WRITER_FEEDER;
    h.cls = AOTX_CLASS_A;
    h.type = AOTX_REC_IMPORT;
    h.body_len = len;
    if (aotx_inbound_wait(ring, stop) != 0) {
        return -1;
    }
    aotx_inbound_put(ring, &h, body);
    s->records++;
    return 0;
}

/* Publishes the head and then the parts of each file, offset by offset. The part numbers
 * are contiguous over the two files, so a gap in them is a lost record. */
static int publish(aotx_import *s, const aotx_inbound_ring *ring,
                   const volatile sig_atomic_t *stop, const aotx_import_head *head,
                   const uint32_t *bytes)
{
    aotx_import_part part;
    uint32_t index = 1;
    uint32_t f;
    if (put(s, ring, stop, head, (uint32_t)sizeof(*head)) != 0) {
        return -1;
    }
    for (f = 0; f < AOTX_IMPORT_FILES; f++) {
        uint32_t at;
        for (at = 0; at < bytes[f]; at += AOTX_IMPORT_TEXT_BYTES) {
            uint32_t take = bytes[f] - at;
            if (take > AOTX_IMPORT_TEXT_BYTES) {
                take = AOTX_IMPORT_TEXT_BYTES;
            }
            memset(&part, 0, sizeof(part));
            part.import = head->import;
            part.part = index++;
            part.file = f;
            part.offset = at;
            part.length = take;
            memcpy(part.text, s->bytes[f] + at, take);
            if (put(s, ring, stop, &part, (uint32_t)sizeof(part)) != 0) {
                return -1;
            }
        }
    }
    s->imports++;
    return 0;
}

int aotx_import_dir(aotx_import *s, const char *path, const aotx_inbound_ring *ring,
                    const volatile sig_atomic_t *stop, const char **reason)
{
    aotx_import_head head;
    uint32_t bytes[AOTX_IMPORT_FILES];
    char name[AOTX_IMPORT_NAME_BYTES * 2];
    int dir_fd;
    int refused;

    *reason = "";
    memset(&head, 0, sizeof(head));
    memset(bytes, 0, sizeof(bytes));
    last_component(path, name, sizeof(name));
    if (!name_ok(name)) {
        *reason = "the name is not one to sixty-three bytes of lower case letters, digits"
                  " and underscores";
        s->refusals++;
        return 1;
    }
    dir_fd = open_module_dir(path, reason);
    if (dir_fd < 0) {
        s->refusals++;
        return 1;
    }
    refused = build_head(s, dir_fd, &head, bytes, reason);
    close(dir_fd);
    if (refused != 0) {
        s->refusals++;
        return 1;
    }
    /* A file of no bytes carries no text, so the count states the files that carry bytes.
     * The place of a file in the head is fixed: 0 the manifest and 1 the body. */
    head.files = ((bytes[0] > 0) ? 1u : 0u) + ((bytes[1] > 0) ? 1u : 0u);
    head.file_bytes[0] = bytes[0];
    head.file_bytes[1] = bytes[1];
    head.import = ++s->number;
    head.part = 0;
    snprintf(head.name, sizeof(head.name), "%s", name);
    path_tail(path, head.path, sizeof(head.path));
    if (publish(s, ring, stop, &head, bytes) != 0) {
        return -1;
    }
    /* The line goes in after the records, so the table names no module that the device did
     * not get. */
    note_import(s, &head, path, bytes[0]);
    return 0;
}

/* Publishes the line that states a refused import. An import that an operator names is a
 * command at run time, so its refusal must reach the console and the message file. The
 * device parser prints this line, and the line of the standard error stays for a run with
 * no console. A successful import needs no such line, because the catalog states the
 * commit.
 *
 * A line that is too long gives the tail of the path and the whole reason. The operator
 * knows the path that was named, and the reason is what the operator does not know. */
static int refuse_line(aotx_import *s, const aotx_inbound_ring *ring,
                       const volatile sig_atomic_t *stop, const char *path,
                       const char *reason)
{
    aotx_record_header h;
    char line[AOTX_BODY_BYTES + 1];
    size_t room = sizeof(line) - 1u;
    size_t fixed = strlen("import  refused: ") + strlen(reason);
    const char *at = path;
    int used;
    if (fixed < room && fixed + strlen(path) > room) {
        at = path + strlen(path) - (room - fixed);
    }
    used = snprintf(line, sizeof(line), "import %s refused: %s", at, reason);
    if (used < 0) {
        return 0;
    }
    memset(&h, 0, sizeof(h));
    h.writer = AOTX_WRITER_FEEDER;
    h.cls = AOTX_CLASS_A;
    h.type = AOTX_REC_INPUT_LINE;
    h.body_len = ((size_t)used < room) ? (uint32_t)used : (uint32_t)room;
    if (aotx_inbound_wait(ring, stop) != 0) {
        return -1;
    }
    aotx_inbound_put(ring, &h, line);
    s->lines++;
    return 0;
}

int aotx_import_take(aotx_import *s, const char *path, const aotx_inbound_ring *ring,
                     const volatile sig_atomic_t *stop)
{
    const char *reason = "";
    int status = aotx_import_dir(s, path, ring, stop, &reason);
    if (status > 0) {
        fprintf(stderr, "import: %s: %s\n", path, reason);
        return refuse_line(s, ring, stop, path, reason);
    }
    return (status < 0) ? -1 : 0;
}

int aotx_import_tree(aotx_import *s, const char *dir, const aotx_inbound_ring *ring,
                     const volatile sig_atomic_t *stop)
{
    struct dirent **found = NULL;
    char path[AOTX_WALK_BYTES];
    int total = scandir(dir, &found, NULL, alphasort);
    int rc = 0;
    int i;
    if (total < 0) {
        fprintf(stderr, "import: %s: the directory does not read\n", dir);
        return 0;
    }
    for (i = 0; i < total; i++) {
        const char *name = found[i]->d_name;
        /* A name that starts with a dot is not a module directory. */
        if (rc == 0 && name[0] != '.') {
            const char *reason = "";
            int status;
            snprintf(path, sizeof(path), "%s/%s", dir, name);
            status = aotx_import_dir(s, path, ring, stop, &reason);
            if (status > 0) {
                fprintf(stderr, "import: %s: %s\n", path, reason);
            } else if (status < 0) {
                rc = -1;
            }
        }
        free(found[i]);
    }
    free(found);
    return rc;
}

int aotx_import_line(const unsigned char *line, uint32_t len, char *out, size_t out_bytes)
{
    static const char word[] = "import";
    uint32_t at = (uint32_t)(sizeof(word) - 1u);
    uint32_t end = len;
    if (len <= at || memcmp(line, word, at) != 0) {
        return 0;
    }
    if (line[at] != ' ' && line[at] != '\t') {
        return 0;
    }
    while (at < len && (line[at] == ' ' || line[at] == '\t')) {
        at++;
    }
    while (end > at && (line[end - 1u] == ' ' || line[end - 1u] == '\t' ||
                        line[end - 1u] == '\r')) {
        end--;
    }
    /* A line that names no path is not an import line. It goes to the device, which states
     * what the command needs. */
    if (end == at || (size_t)(end - at) >= out_bytes) {
        return 0;
    }
    memcpy(out, line + at, end - at);
    out[end - at] = '\0';
    return 1;
}
