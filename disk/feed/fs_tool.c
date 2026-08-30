/* Purpose: Run the host tools of a request under the allowed root and publish the replies.
 * Owns: The descriptor of the root, the descriptor of the requests file, and the table of
 *       requests that were executed.
 * Threading: One thread; the feeder is the only caller and the only producer of the ring.
 * Lifetime: From the open of the root to the close of the feeder. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/feed/import.h"
#include "disk/feed/modules.h"
#include "disk/feed/run_tool.h"

#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* The allowed root is the security boundary of the system. A path that leaves the root is
 * refused. A path that goes through a symbolic link is refused. A path that names anything
 * other than a regular file is refused. The reason goes back to the agent.
 *
 * The authorization boundary is the requests file. The drain writes a line only for a
 * request that needs no authorization or that the operator granted. This file executes
 * what that file holds and nothing else. A tool the operator refused thus has no path to a
 * program or to a file. A tool that still waits for the operator has none either. */

/* The entries one directory listing holds, and the bytes of one entry name. A directory
 * with more entries states the cut, as a listing over the cap does. */
#define AOTX_LIST_ENTRIES 256u
#define AOTX_LIST_NAME    256u

/* The digest line is the first line of every file read result. */
#define AOTX_DIGEST_HEAD 73u

/* The reason of a refusal that names a word of the request. The text lives beside the
 * program, because a caller reads the reason after the function returns. */
static char refuse_text[192];

/* Reports whether the table already holds the identity. */
static int already(const aotx_fs_tool *t, uint32_t request)
{
    uint32_t i;
    for (i = 0; i < AOTX_FS_SEEN; i++) {
        if (t->seen[i] == (uint64_t)request + 1u) {
            return 1;
        }
    }
    return 0;
}

static void remember(aotx_fs_tool *t, uint32_t request)
{
    t->seen[t->seen_at % AOTX_FS_SEEN] = (uint64_t)request + 1u;
    t->seen_at++;
}

/* ---- the reply ---- */

int aotx_fs_put_part(aotx_fs_tool *t, const aotx_inbound_ring *ring,
                     const volatile sig_atomic_t *stop, uint32_t agent, uint32_t request,
                     uint32_t status, uint32_t part, uint32_t parts, const void *data,
                     uint32_t len)
{
    aotx_record_header h;
    aotx_tool_reply_body body;
    memset(&h, 0, sizeof(h));
    memset(&body, 0, sizeof(body));
    h.writer = AOTX_WRITER_FEEDER;
    h.cls = AOTX_CLASS_A;
    h.type = AOTX_REC_TOOL_REPLY;
    h.body_len = (uint32_t)sizeof(body);
    body.agent = agent;
    body.request = request;
    body.status = status;
    body.part = part;
    body.parts = parts;
    body.len = (len > AOTX_TOOL_REPLY_BYTES) ? AOTX_TOOL_REPLY_BYTES : len;
    if (body.len > 0) {
        memcpy(body.bytes, data, body.len);
    }
    if (aotx_inbound_wait(ring, stop) != 0) {
        return -1;
    }
    aotx_inbound_put(ring, &h, &body);
    t->replies++;
    return 0;
}

int aotx_fs_put_reason(aotx_fs_tool *t, const aotx_inbound_ring *ring,
                       const volatile sig_atomic_t *stop, uint32_t agent, uint32_t request,
                       uint32_t status, const char *reason)
{
    if (status == AOTX_TOOL_REFUSED) {
        t->refusals++;
    } else {
        t->errors++;
    }
    return aotx_fs_put_part(t, ring, stop, agent, request, status, 0u, 1u, reason,
                            (uint32_t)strlen(reason));
}

int aotx_fs_put_bytes(aotx_fs_tool *t, const aotx_inbound_ring *ring,
                      const volatile sig_atomic_t *stop, uint32_t agent, uint32_t request,
                      const unsigned char *data, uint32_t len, const char *reason)
{
    uint32_t content = (len + AOTX_TOOL_REPLY_BYTES - 1u) / AOTX_TOOL_REPLY_BYTES;
    uint32_t parts;
    uint32_t i;
    if (content == 0 && reason == NULL) {
        /* A result of no bytes is one part that carries no byte. The reply thus has a
         * part and the device sees a whole reply. */
        content = 1;
    }
    parts = content + ((reason != NULL) ? 1u : 0u);
    for (i = 0; i < content; i++) {
        uint32_t at = i * AOTX_TOOL_REPLY_BYTES;
        uint32_t take = len - at;
        if (take > AOTX_TOOL_REPLY_BYTES) {
            take = AOTX_TOOL_REPLY_BYTES;
        }
        if (aotx_fs_put_part(t, ring, stop, agent, request, AOTX_TOOL_OK, i, parts,
                             data + at, take) != 0) {
            return -1;
        }
    }
    if (reason != NULL) {
        t->errors++;
        return aotx_fs_put_part(t, ring, stop, agent, request, AOTX_TOOL_ERROR, parts - 1u,
                                parts, reason, (uint32_t)strlen(reason));
    }
    return 0;
}

/* ---- the tools that read ---- */

/* Opens one file under the allowed root. The root is the boundary. A path that starts at
 * the root of the file system is refused before the walk. A path that holds a component of
 * two dots is refused there too. Returns the descriptor, or -1 with the status and the
 * reason. */
static int open_under(int root_fd, const char *path, unsigned flags, uint32_t *status,
                      const char **reason)
{
    aotx_walk walk;
    *status = AOTX_TOOL_REFUSED;
    if (path[0] == '\0') {
        *reason = "the path is empty";
        return -1;
    }
    if (path[0] == '/') {
        *reason = "the path starts at the root of the file system";
        return -1;
    }
    if (strlen(path) > AOTX_TOOL_ARG_BYTES) {
        *reason = "the path is too long";
        return -1;
    }
    return aotx_path_walk(&walk, root_fd, path, AOTX_WALK_NO_UP | flags, status, reason);
}

/* Reads one file under the root and publishes its reply. Returns 0 or -1. */
static int read_file(aotx_fs_tool *t, const aotx_inbound_ring *ring,
                     const volatile sig_atomic_t *stop, uint32_t agent, uint32_t request,
                     const char *path)
{
    struct stat info;
    const char *reason = "";
    uint32_t status = AOTX_TOOL_OK;
    uint32_t got = 0;
    uint32_t content_cap = AOTX_FS_CAP - AOTX_DIGEST_HEAD;
    aotx_sha256 digest;
    unsigned char raw[AOTX_SHA256_DIGEST];
    char text[AOTX_SHA256_DIGEST * 2u + 1u];
    char head[AOTX_DIGEST_HEAD + 1u];
    int cut = 0;
    int fd = open_under(t->root_fd, path, 0u, &status, &reason);
    if (fd < 0) {
        return aotx_fs_put_reason(t, ring, stop, agent, request, status, reason);
    }
    if (fstat(fd, &info) != 0 || !S_ISREG(info.st_mode)) {
        close(fd);
        return aotx_fs_put_reason(t, ring, stop, agent, request, AOTX_TOOL_REFUSED,
                                  "the path does not name a regular file");
    }
    aotx_sha256_init(&digest);
    while (got < content_cap) {
        ssize_t n = read(fd, t->bytes + AOTX_DIGEST_HEAD + got,
                         (size_t)(content_cap - got));
        if (n < 0) {
            close(fd);
            return aotx_fs_put_reason(t, ring, stop, agent, request, AOTX_TOOL_ERROR,
                                      "the file does not read");
        }
        if (n == 0) {
            break;
        }
        aotx_sha256_update(&digest, t->bytes + AOTX_DIGEST_HEAD + got, (size_t)n);
        got += (uint32_t)n;
    }
    if (got == content_cap) {
        unsigned char more;
        ssize_t n = read(fd, &more, 1);
        cut = (n > 0);
    }
    close(fd);
    aotx_sha256_final(&digest, raw);
    aotx_sha256_text(raw, text);
    snprintf(head, sizeof(head), "sha256: %s\n", text);
    memcpy(t->bytes, head, AOTX_DIGEST_HEAD);
    if (cut) {
        /* The count comes from the cap itself, so the reason cannot state a figure that
         * the cap does not hold. */
        snprintf(refuse_text, sizeof(refuse_text),
                 "the file is longer than the cap and the reply holds the first %u bytes",
                 (unsigned)content_cap);
    }
    return aotx_fs_put_bytes(t, ring, stop, agent, request, t->bytes,
                             got + AOTX_DIGEST_HEAD,
                             cut ? refuse_text : NULL);
}

/* Gives the size, the modification time and the digest of one file, with no file bytes. */
static int stat_file(aotx_fs_tool *t, const aotx_inbound_ring *ring,
                     const volatile sig_atomic_t *stop, uint32_t agent, uint32_t request,
                     const char *path)
{
    struct stat info;
    const char *reason = "";
    uint32_t status = AOTX_TOOL_OK;
    aotx_sha256 digest;
    unsigned char raw[AOTX_SHA256_DIGEST];
    char text[AOTX_SHA256_DIGEST * 2u + 1u];
    int wrote;
    int fd = open_under(t->root_fd, path, 0u, &status, &reason);
    if (fd < 0) {
        return aotx_fs_put_reason(t, ring, stop, agent, request, status, reason);
    }
    if (fstat(fd, &info) != 0 || !S_ISREG(info.st_mode)) {
        close(fd);
        return aotx_fs_put_reason(t, ring, stop, agent, request, AOTX_TOOL_REFUSED,
                                  "the path does not name a regular file");
    }
    aotx_sha256_init(&digest);
    if (aotx_sha256_read(fd, 0u, (uint64_t)info.st_size, t->bytes,
                         sizeof(t->bytes), &digest) != 0) {
        close(fd);
        return aotx_fs_put_reason(t, ring, stop, agent, request, AOTX_TOOL_ERROR,
                                  "the file does not read for its digest");
    }
    close(fd);
    aotx_sha256_final(&digest, raw);
    aotx_sha256_text(raw, text);
    wrote = snprintf((char *)t->bytes, sizeof(t->bytes),
                     "bytes: %llu\nmodified: %lld.%09ld\nsha256: %s\n",
                     (unsigned long long)info.st_size,
                     (long long)info.st_mtim.tv_sec, info.st_mtim.tv_nsec, text);
    if (wrote < 0 || (size_t)wrote >= sizeof(t->bytes)) {
        return aotx_fs_put_reason(t, ring, stop, agent, request, AOTX_TOOL_ERROR,
                                  "the file state reply does not fit");
    }
    return aotx_fs_put_bytes(t, ring, stop, agent, request, t->bytes,
                             (uint32_t)wrote, NULL);
}

/* The names of one directory. The table lives beside the program, because a listing of
 * many entries must not stand on the stack of the poll loop. */
static char list_names[AOTX_LIST_ENTRIES][AOTX_LIST_NAME];

/* Puts two names in order, so a listing is sorted by name and two runs give one order. */
static int name_order(const void *a, const void *b)
{
    return strcmp((const char *)a, (const char *)b);
}

/* Gives the word for the kind of one entry. */
static const char *kind_word(mode_t mode)
{
    if (S_ISREG(mode)) {
        return "file";
    }
    if (S_ISDIR(mode)) {
        return "directory";
    }
    if (S_ISLNK(mode)) {
        return "link";
    }
    return "other";
}

/* Lists the entries of one directory under the root. Each entry gives one line with the
 * name, the kind and the byte count. Returns 0 or -1. */
static int list_dir(aotx_fs_tool *t, const aotx_inbound_ring *ring,
                    const volatile sig_atomic_t *stop, uint32_t agent, uint32_t request,
                    const char *path)
{
    const char *reason = "";
    uint32_t status = AOTX_TOOL_OK;
    uint32_t count = 0;
    uint32_t used = 0;
    uint32_t i;
    int over = 0;
    DIR *dir;
    struct dirent *entry;
    int fd;
    if (path[0] == '\0' || strcmp(path, ".") == 0) {
        /* The root itself is a directory an agent may list, and the walk of no component
         * names no file. The root is already open. */
        fd = dup(t->root_fd);
        if (fd < 0) {
            return aotx_fs_put_reason(t, ring, stop, agent, request, AOTX_TOOL_ERROR,
                                      "the root does not open again");
        }
    } else {
        fd = open_under(t->root_fd, path, AOTX_WALK_DIR, &status, &reason);
    }
    if (fd < 0) {
        return aotx_fs_put_reason(t, ring, stop, agent, request, status, reason);
    }
    dir = fdopendir(fd);
    if (dir == NULL) {
        close(fd);
        return aotx_fs_put_reason(t, ring, stop, agent, request, AOTX_TOOL_ERROR,
                                  "the directory does not read");
    }
    while ((entry = readdir(dir)) != NULL) {
        /* The name of the directory and the name of the one above it are not entries.
         * The walk also refuses a path that names them. */
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        if (count >= AOTX_LIST_ENTRIES || strlen(entry->d_name) >= AOTX_LIST_NAME) {
            over = 1;
            continue;
        }
        snprintf(list_names[count], AOTX_LIST_NAME, "%s", entry->d_name);
        count++;
    }
    qsort(list_names, count, AOTX_LIST_NAME, name_order);
    for (i = 0; i < count; i++) {
        struct stat info;
        char line[AOTX_LIST_NAME + 64];
        int line_len;
        unsigned long long bytes = 0;
        const char *word = "other";
        if (fstatat(dirfd(dir), list_names[i], &info, AT_SYMLINK_NOFOLLOW) == 0) {
            word = kind_word(info.st_mode);
            bytes = (unsigned long long)info.st_size;
        }
        line_len = snprintf(line, sizeof(line), "%s %s %llu\n", list_names[i], word, bytes);
        if (line_len < 0) {
            continue;
        }
        if (used + (uint32_t)line_len > AOTX_FS_CAP) {
            over = 1;
            break;
        }
        memcpy(t->bytes + used, line, (size_t)line_len);
        used += (uint32_t)line_len;
    }
    closedir(dir);
    if (over) {
        snprintf(refuse_text, sizeof(refuse_text),
                 "the directory holds more entries than the cap of %u bytes takes and the"
                 " reply holds the first of them", (unsigned)AOTX_FS_CAP);
    }
    return aotx_fs_put_bytes(t, ring, stop, agent, request, t->bytes, used,
                             over ? refuse_text : NULL);
}

/* ---- the tools that run a program ---- */

/* Starts the built-in tool that runs a command line. The working directory is the allowed
 * root. Returns 0, or -1 when the ring closed. */
static int run_command(aotx_fs_tool *t, const aotx_inbound_ring *ring,
                       const volatile sig_atomic_t *stop, uint32_t agent, uint32_t request,
                       const char *command, const char *line)
{
    char *args[4];
    char shell[] = AOTX_RUN_SHELL;
    char flag[] = "-c";
    const char *reason = "";
    args[0] = shell;
    args[1] = flag;
    args[2] = (char *)command;
    args[3] = NULL;
    if (aotx_run_start(t->kids, agent, request, t->root_fd, AOTX_RUN_SHELL, args, "run",
                       line, t->timeout, &reason) != 0) {
        return aotx_fs_put_reason(t, ring, stop, agent, request, AOTX_TOOL_ERROR, reason);
    }
    return 0;
}

/* Starts the program of one host tool of the catalog. The working directory is the module
 * directory. Returns 0, or -1 when the ring closed. */
static int run_module(aotx_fs_tool *t, const aotx_inbound_ring *ring,
                      const volatile sig_atomic_t *stop, uint32_t agent, uint32_t request,
                      const aotx_module_row *row, const char *line)
{
    char *args[2];
    char program[AOTX_MODULE_PROGRAM];
    const char *reason = "";
    aotx_walk walk;
    uint32_t status = AOTX_TOOL_OK;
    int dir_fd;
    int check_fd;
    if (row->program[0] == '\0') {
        return aotx_fs_put_reason(t, ring, stop, agent, request, AOTX_TOOL_ERROR,
                                  "the manifest of the tool names no program");
    }
    if (row->program[0] == '/') {
        return aotx_fs_put_reason(t, ring, stop, agent, request, AOTX_TOOL_REFUSED,
                                  "the program starts at the root of the file system");
    }
    dir_fd = open(row->dir, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dir_fd < 0) {
        return aotx_fs_put_reason(t, ring, stop, agent, request, AOTX_TOOL_ERROR,
                                  "the module directory does not open");
    }
    /* The program stays under the module directory. The walk applies the same rule that
     * every file operation of the feeder applies, so a link out of the directory fails. */
    check_fd = aotx_path_walk(&walk, dir_fd, row->program, AOTX_WALK_NO_UP, &status, &reason);
    if (check_fd < 0) {
        close(dir_fd);
        return aotx_fs_put_reason(t, ring, stop, agent, request, status, reason);
    }
    close(check_fd);
    snprintf(program, sizeof(program), "%s", row->program);
    args[0] = program;
    args[1] = NULL;
    if (aotx_run_start(t->kids, agent, request, dir_fd, program, args, row->name, line,
                       row->timeout, &reason) != 0) {
        close(dir_fd);
        return aotx_fs_put_reason(t, ring, stop, agent, request, AOTX_TOOL_ERROR, reason);
    }
    close(dir_fd);
    return 0;
}

/* ---- one line of the requests file ---- */

/* Splits the arguments of a built-in tool and gives the value of one key. A key the call
 * does not carry is a refusal that names the key. Returns 1, or 0 when the tool answered
 * the request with a reason. */
static int take_args(aotx_fs_tool *t, const aotx_inbound_ring *ring,
                     const volatile sig_atomic_t *stop, uint32_t agent, uint32_t request,
                     aotx_args *a, const char *arg, const char *const *keys, uint32_t count,
                     int *rc)
{
    const char *reason = "";
    uint32_t i;
    if (!aotx_args_split(a, arg, keys, count, &reason)) {
        *rc = aotx_fs_put_reason(t, ring, stop, agent, request, AOTX_TOOL_REFUSED, reason);
        return 0;
    }
    for (i = 0; i < count; i++) {
        if (aotx_args_value(a, keys[i]) == NULL) {
            snprintf(refuse_text, sizeof(refuse_text),
                     "the call carries no argument under the key %.64s", keys[i]);
            *rc = aotx_fs_put_reason(t, ring, stop, agent, request, AOTX_TOOL_REFUSED,
                                     refuse_text);
            return 0;
        }
    }
    return 1;
}

/* Runs the tool that the line names. Returns 0 or -1. */
static int take_tool(aotx_fs_tool *t, const aotx_inbound_ring *ring,
                     const volatile sig_atomic_t *stop, uint32_t agent, uint32_t request,
                     const char *tool, uint32_t number, const char *arg)
{
    static const char *const one_path[1] = { "path" };
    static const char *const write_keys[2] = { "path", "text" };
    static const char *const update_keys[3] = { "path", "old", "new" };
    static const char *const run_keys[1] = { "command" };
    const aotx_module_row *row;
    aotx_args a;
    int rc = 0;
    if (strcmp(tool, "fs_read") == 0) {
        if (!take_args(t, ring, stop, agent, request, &a, arg, one_path, 1u, &rc)) {
            return rc;
        }
        return read_file(t, ring, stop, agent, request, aotx_args_value(&a, "path"));
    }
    if (strcmp(tool, "fs_list") == 0) {
        if (!take_args(t, ring, stop, agent, request, &a, arg, one_path, 1u, &rc)) {
            return rc;
        }
        return list_dir(t, ring, stop, agent, request, aotx_args_value(&a, "path"));
    }
    if (strcmp(tool, "fs_stat") == 0) {
        if (!take_args(t, ring, stop, agent, request, &a, arg, one_path, 1u, &rc)) {
            return rc;
        }
        return stat_file(t, ring, stop, agent, request, aotx_args_value(&a, "path"));
    }
    if (strcmp(tool, "fs_write") == 0) {
        if (!take_args(t, ring, stop, agent, request, &a, arg, write_keys, 2u, &rc)) {
            return rc;
        }
        return aotx_fs_write_file(t, ring, stop, agent, request, aotx_args_value(&a, "path"),
                                  aotx_args_value(&a, "text"));
    }
    if (strcmp(tool, "fs_update") == 0) {
        if (!take_args(t, ring, stop, agent, request, &a, arg, update_keys, 3u, &rc)) {
            return rc;
        }
        return aotx_fs_update_file(t, ring, stop, agent, request, aotx_args_value(&a, "path"),
                                   aotx_args_value(&a, "old"), aotx_args_value(&a, "new"));
    }
    if (strcmp(tool, "run") == 0) {
        if (!take_args(t, ring, stop, agent, request, &a, arg, run_keys, 1u, &rc)) {
            return rc;
        }
        return run_command(t, ring, stop, agent, request, aotx_args_value(&a, "command"),
                           t->line);
    }
    /* A tool of the catalog is named by the line or found under its number. The table
     * holds the directory, the program and the timeout of every host tool of the run. */
    row = (t->table != NULL) ? aotx_modules_name(t->table, tool) : NULL;
    if (row == NULL && t->table != NULL && number >= AOTX_TOOL_MODULE_BASE) {
        row = aotx_modules_number(t->table, number);
    }
    if (row != NULL) {
        return run_module(t, ring, stop, agent, request, row, t->line);
    }
    snprintf(refuse_text, sizeof(refuse_text),
             "the tool %.64s is not a host tool of this run", tool);
    return aotx_fs_put_reason(t, ring, stop, agent, request, AOTX_TOOL_ERROR, refuse_text);
}

/* Takes one line of the requests file. Returns 0 or -1. */
static int take_line(aotx_fs_tool *t, struct aotx_import *imports,
                     const aotx_inbound_ring *ring, const volatile sig_atomic_t *stop)
{
    char tool[AOTX_IMPORT_NAME_BYTES];
    char arg[AOTX_TOOL_ARG_BYTES + 1];
    uint64_t request = 0;
    uint64_t agent = 0;
    uint64_t number = 0;
    if (!aotx_json_number(t->line, "\"request\":", &request) || request == 0 ||
        !aotx_json_number(t->line, "\"agent\":", &agent) ||
        !aotx_json_text(t->line, "\"tool\":\"", tool, sizeof(tool)) ||
        !aotx_json_text(t->line, "\"arg\":\"", arg, sizeof(arg))) {
        /* A line the reader cannot read names no request, so no reply can name it. */
        t->errors++;
        return 0;
    }
    if (!aotx_json_number(t->line, "\"number\":", &number)) {
        number = 0;
    }
    if (already(t, (uint32_t)request)) {
        /* One request is executed one time. A tool has an effect on the host, and a second
         * run of it is a second effect. */
        t->again++;
        return 0;
    }
    remember(t, (uint32_t)request);
    t->taken++;
    if (strcmp(tool, "import") == 0) {
        /* A surface the feeder does not read gives an import this way. The directory
         * goes out as it goes out for a line of the standard input. The import is the
         * reply, so the line makes no reply record. The table of identities above keeps
         * the feeder from reading one directory twice. */
        t->imports++;
        return aotx_import_take(imports, arg, ring, stop);
    }
    return take_tool(t, ring, stop, (uint32_t)agent, (uint32_t)request, tool,
                     (uint32_t)number, arg);
}

int aotx_fs_tool_open(aotx_fs_tool *t, const char *root, const char *requests)
{
    struct aotx_children *kids = t->kids;
    struct aotx_modules *table = t->table;
    uint32_t timeout = t->timeout;
    memset(t, 0, sizeof(*t));
    t->kids = kids;
    t->table = table;
    t->timeout = (timeout > 0u) ? timeout : AOTX_MODULE_TIMEOUT;
    t->root_fd = -1;
    t->requests_fd = -1;
    if (root == NULL || requests == NULL) {
        return 0;
    }
    t->root_fd = open(root, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (t->root_fd < 0) {
        return -1;
    }
    snprintf(t->requests_path, sizeof(t->requests_path), "%s", requests);
    t->requests_fd = open(t->requests_path, O_RDONLY | O_CLOEXEC);
    if (t->requests_fd >= 0 && lseek(t->requests_fd, 0, SEEK_END) < 0) {
        return -1;
    }
    return 0;
}

int aotx_fs_tool_poll(aotx_fs_tool *t, struct aotx_import *imports,
                      const aotx_inbound_ring *ring, const volatile sig_atomic_t *stop)
{
    unsigned char buffer[4096];
    int taken = 0;
    if (aotx_run_poll(t, ring, stop) != 0) {
        return -1;
    }
    if (t->root_fd < 0) {
        return 0;
    }
    if (t->requests_fd < 0) {
        t->requests_fd = open(t->requests_path, O_RDONLY | O_CLOEXEC);
        if (t->requests_fd < 0) {
            return 0;
        }
    }
    for (;;) {
        ssize_t n = read(t->requests_fd, buffer, sizeof(buffer));
        ssize_t i;
        if (n <= 0) {
            return taken;
        }
        for (i = 0; i < n; i++) {
            if (buffer[i] != '\n') {
                if (t->fill + 1u < AOTX_FS_LINE) {
                    t->line[t->fill++] = (char)buffer[i];
                } else {
                    /* A line longer than the buffer is not a line this reader wrote. */
                    t->over = 1;
                }
                continue;
            }
            t->line[t->fill] = '\0';
            if (!t->over) {
                if (take_line(t, imports, ring, stop) != 0) {
                    return -1;
                }
                taken++;
            } else {
                t->errors++;
            }
            t->fill = 0;
            t->over = 0;
        }
    }
}

void aotx_fs_tool_close(aotx_fs_tool *t)
{
    if (t->kids != NULL) {
        aotx_run_close(t->kids);
    }
    if (t->root_fd >= 0) {
        close(t->root_fd);
    }
    if (t->requests_fd >= 0) {
        close(t->requests_fd);
    }
    t->root_fd = -1;
    t->requests_fd = -1;
}
