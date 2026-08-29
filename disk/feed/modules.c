/* Purpose: Hold the programs of the host tools of a run, and keep them in a file on disk.
 * Owns: The open table file and the rows of the table.
 * Threading: One thread; the feeder is the only caller.
 * Lifetime: From the start of the feeder to its close. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/feed/modules.h"

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* The bytes one line of the table may take. */
#define AOTX_MODULE_LINE 1024u

/* Reports whether a text goes into a line of the table without an escape. The table holds
 * paths, and a path with a quotation mark, a backslash or a control byte would need an
 * escape. Such a path is refused instead, so the writer and the reader hold one rule. */
static int plain(const char *text)
{
    size_t i;
    for (i = 0; text[i] != '\0'; i++) {
        unsigned char b = (unsigned char)text[i];
        if (b < 0x20u || b == 0x7fu || b == '"' || b == '\\') {
            return 0;
        }
    }
    return 1;
}

/* Gives the row of a name, or null. */
static aotx_module_row *find(aotx_modules *m, const char *name)
{
    uint32_t i;
    for (i = 0; i < m->count; i++) {
        if (strcmp(m->row[i].name, name) == 0) {
            return &m->row[i];
        }
    }
    return NULL;
}

/* Takes one row into the table. An import of a name the table holds replaces that row
 * whole, which is the rule the catalog applies to an entry. Returns the row, or null when
 * the table is full. */
static aotx_module_row *take(aotx_modules *m, const char *name)
{
    aotx_module_row *row = find(m, name);
    if (row != NULL) {
        return row;
    }
    if (m->count >= AOTX_MODULE_ROWS) {
        m->refused++;
        return NULL;
    }
    row = &m->row[m->count++];
    memset(row, 0, sizeof(*row));
    return row;
}

/* Fills one row from one line of the table file. Returns 1, or 0 when the line holds no
 * row that the feeder can run. */
static int line_row(aotx_modules *m, const char *line)
{
    aotx_module_row *row;
    char name[AOTX_IMPORT_NAME_BYTES];
    char dir[AOTX_MODULE_DIR_BYTES];
    char program[AOTX_MODULE_PROGRAM];
    char word[16];
    uint64_t import = 0;
    uint64_t timeout = 0;
    if (!aotx_json_text(line, "\"name\":\"", name, sizeof(name)) ||
        !aotx_json_text(line, "\"dir\":\"", dir, sizeof(dir)) ||
        !aotx_json_text(line, "\"program\":\"", program, sizeof(program)) ||
        !aotx_json_number(line, "\"import\":", &import) ||
        !aotx_json_number(line, "\"timeout\":", &timeout)) {
        return 0;
    }
    row = take(m, name);
    if (row == NULL) {
        return 0;
    }
    row->import = (uint32_t)import;
    row->number = AOTX_TOOL_MODULE_BASE + (uint32_t)import;
    row->timeout = (uint32_t)timeout;
    row->authorize = (aotx_json_text(line, "\"authorize\":\"", word, sizeof(word)) &&
                      strcmp(word, "always") == 0) ? 1u : 0u;
    snprintf(row->name, sizeof(row->name), "%s", name);
    snprintf(row->dir, sizeof(row->dir), "%s", dir);
    snprintf(row->program, sizeof(row->program), "%s", program);
    if (row->import > m->high) {
        m->high = row->import;
    }
    m->read++;
    return 1;
}

/* Reads the table file from its first byte. A line the reader cannot read is counted and
 * the file goes on, because one bad line must not keep the other programs out. */
static void read_table(aotx_modules *m)
{
    char line[AOTX_MODULE_LINE];
    unsigned char buffer[4096];
    uint32_t fill = 0;
    int over = 0;
    int fd = open(m->path, O_RDONLY | O_CLOEXEC);
    ssize_t n;
    if (fd < 0) {
        return;
    }
    while ((n = read(fd, buffer, sizeof(buffer))) > 0) {
        ssize_t i;
        for (i = 0; i < n; i++) {
            if (buffer[i] != '\n') {
                if (fill + 1u < sizeof(line)) {
                    line[fill++] = (char)buffer[i];
                } else {
                    over = 1;
                }
                continue;
            }
            line[fill] = '\0';
            if (over || !line_row(m, line)) {
                m->refused++;
            }
            fill = 0;
            over = 0;
        }
    }
    close(fd);
}

int aotx_modules_open(aotx_modules *m, const char *requests)
{
    const char *cut;
    size_t dir_len;
    memset(m, 0, sizeof(*m));
    m->fd = -1;
    if (requests == NULL) {
        return 0;
    }
    /* The table stands beside the requests file, because both are the feeder's view of one
     * journal. A requests path with no slash names a file of the working directory. */
    cut = strrchr(requests, '/');
    dir_len = (cut != NULL) ? (size_t)(cut - requests) + 1u : 0u;
    if (dir_len + sizeof(AOTX_MODULE_TABLE) > sizeof(m->path)) {
        return -1;
    }
    memcpy(m->path, requests, dir_len);
    memcpy(m->path + dir_len, AOTX_MODULE_TABLE, sizeof(AOTX_MODULE_TABLE));
    read_table(m);
    m->fd = open(m->path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    return (m->fd >= 0) ? 0 : -1;
}

int aotx_modules_add(aotx_modules *m, uint32_t import, const char *name, const char *dir,
                     const char *program, uint32_t timeout, uint32_t authorize)
{
    aotx_module_row *row;
    char line[AOTX_MODULE_LINE];
    int used;
    if (m->fd < 0) {
        /* A feeder with no requests file answers no request, so it needs no table. */
        return 0;
    }
    if (!plain(name) || !plain(dir) || !plain(program) ||
        strlen(dir) >= sizeof(row->dir) || strlen(program) >= sizeof(row->program)) {
        m->refused++;
        return -1;
    }
    row = take(m, name);
    if (row == NULL) {
        return -1;
    }
    row->import = import;
    row->number = AOTX_TOOL_MODULE_BASE + import;
    row->timeout = (timeout > 0u) ? timeout : AOTX_MODULE_TIMEOUT;
    row->authorize = authorize;
    snprintf(row->name, sizeof(row->name), "%s", name);
    snprintf(row->dir, sizeof(row->dir), "%s", dir);
    snprintf(row->program, sizeof(row->program), "%s", program);
    if (row->import > m->high) {
        m->high = row->import;
    }
    used = snprintf(line, sizeof(line),
                    "{\"name\":\"%s\",\"kind\":\"tool\",\"side\":\"host\",\"dir\":\"%s\","
                    "\"program\":\"%s\",\"timeout\":%u,\"authorize\":\"%s\","
                    "\"import\":%u,\"number\":%u}\n",
                    row->name, row->dir, row->program, row->timeout,
                    (row->authorize != 0u) ? "always" : "never", row->import, row->number);
    if (used < 0 || (size_t)used >= sizeof(line)) {
        m->refused++;
        return -1;
    }
    if (write(m->fd, line, (size_t)used) != (ssize_t)used) {
        return -1;
    }
    m->written++;
    return 0;
}

const aotx_module_row *aotx_modules_number(const aotx_modules *m, uint32_t number)
{
    uint32_t i;
    for (i = 0; i < m->count; i++) {
        if (m->row[i].number == number) {
            return &m->row[i];
        }
    }
    return NULL;
}

const aotx_module_row *aotx_modules_name(const aotx_modules *m, const char *name)
{
    uint32_t i;
    for (i = 0; i < m->count; i++) {
        if (strcmp(m->row[i].name, name) == 0) {
            return &m->row[i];
        }
    }
    return NULL;
}

uint32_t aotx_modules_high(const aotx_modules *m)
{
    return m->high;
}

void aotx_modules_close(aotx_modules *m)
{
    if (m->fd >= 0) {
        close(m->fd);
    }
    m->fd = -1;
}
