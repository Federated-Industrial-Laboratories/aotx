/* Purpose: Write and check the models manifest that names each model file and its digest.
 * Owns: The entries of one run and the buffer that hashes one file.
 * Threading: One thread; the program does one command and ends.
 * Lifetime: The run of the program. */
#include "disk/modelfile/manifest.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* The buffer that hashes a file at the write command. */
#define AOTX_WRITE_BUFFER (4u * 1024u * 1024u)

static int usage(void)
{
    fprintf(stderr,
            "usage: aotx_manifest write <dir> <name> <file> <source> <revision> <license>\n"
            "       aotx_manifest check <dir>\n"
            "The write command hashes the file and adds one line to the manifest.\n"
            "The check command hashes each file and compares it with its line.\n"
            "Exit codes: 0 every file is ok, 1 a file is different, missing, or already\n"
            "in the manifest, 2 an error.\n");
    return 2;
}

/* Gives the last part of a path, which is the name of the file in its directory. */
static const char *last_part(const char *path)
{
    const char *slash = strrchr(path, '/');
    return (slash != NULL) ? (slash + 1) : path;
}

/* Fills a field and says which field refuses the value. Returns 0 or 2. */
static int put_field(char *out, size_t out_bytes, const char *value, const char *field)
{
    if (aotx_manifest_field(value) != 0) {
        fprintf(stderr, "aotx_manifest: the %s holds a byte that the format refuses\n", field);
        return 2;
    }
    if (strlen(value) + 1u > out_bytes) {
        fprintf(stderr, "aotx_manifest: the %s is too long\n", field);
        return 2;
    }
    memcpy(out, value, strlen(value) + 1u);
    return 0;
}

/* Refuses a name or a path that the manifest already holds. One file has one line, and
 * one name names one file. Returns 0, 1 when the manifest holds the name or the file, or
 * 2 when the manifest does not read. */
static int entry_is_new(const char *dir, const aotx_manifest_entry *entry)
{
    char path[AOTX_MANIFEST_PATH];
    aotx_manifest_entry entries[AOTX_MANIFEST_MAX];
    int count;
    int i;
    if (aotx_manifest_path(path, sizeof(path), dir, AOTX_MANIFEST_NAME) != 0) {
        return 2;
    }
    if (access(path, F_OK) != 0) {
        return 0;
    }
    count = aotx_manifest_read(dir, entries, AOTX_MANIFEST_MAX);
    if (count < 0) {
        return 2;
    }
    for (i = 0; i < count; i++) {
        if (strcmp(entries[i].name, entry->name) == 0) {
            fprintf(stderr, "aotx_manifest: the manifest already holds the name %s\n",
                    entry->name);
            return 1;
        }
        if (strcmp(entries[i].path, entry->path) == 0) {
            fprintf(stderr,
                    "aotx_manifest: the manifest already holds the file %s, with the name %s\n",
                    entry->path, entries[i].name);
            return 1;
        }
    }
    return 0;
}

static int write_entry(const char *dir, const aotx_manifest_entry *entry)
{
    char path[AOTX_MANIFEST_PATH];
    char line[AOTX_MANIFEST_LINE];
    FILE *file;
    if (aotx_manifest_write_line(line, sizeof(line), entry) != 0) {
        fprintf(stderr, "aotx_manifest: the line does not fit\n");
        return 2;
    }
    /* The line is proven against the file before it goes in, so the manifest holds no
     * line that a check cannot pass. */
    if (aotx_manifest_check(dir, entry) != 0) {
        fprintf(stderr, "aotx_manifest: the file is not in %s under the name %s\n", dir,
                entry->path);
        return 2;
    }
    if (aotx_manifest_path(path, sizeof(path), dir, AOTX_MANIFEST_NAME) != 0) {
        return 2;
    }
    file = fopen(path, "a");
    if (file == NULL) {
        fprintf(stderr, "aotx_manifest: %s: the manifest does not open to write\n", path);
        return 2;
    }
    if (fputs(line, file) < 0 || fflush(file) != 0 || fsync(fileno(file)) != 0) {
        fprintf(stderr, "aotx_manifest: %s: the line does not write\n", path);
        fclose(file);
        return 2;
    }
    fclose(file);
    return 0;
}

static int command_write(char **argv)
{
    aotx_manifest_entry entry;
    void *buffer;
    int rc;
    memset(&entry, 0, sizeof(entry));
    if (put_field(entry.name, sizeof(entry.name), argv[3], "name") != 0 ||
        put_field(entry.path, sizeof(entry.path), last_part(argv[4]), "path") != 0 ||
        put_field(entry.source, sizeof(entry.source), argv[5], "source") != 0 ||
        put_field(entry.revision, sizeof(entry.revision), argv[6], "revision") != 0 ||
        put_field(entry.license, sizeof(entry.license), argv[7], "license") != 0) {
        return 2;
    }
    snprintf(entry.role, sizeof(entry.role), "%s", entry.name);
    rc = entry_is_new(argv[2], &entry);
    if (rc != 0) {
        return rc;
    }
    /* A path that does not resolve from the working directory is the common cause of a
     * write that fails. The message therefore names that cause and not the hash. */
    if (access(argv[4], R_OK) != 0) {
        fprintf(stderr, "aotx_manifest: %s: the file does not open to read\n", argv[4]);
        return 2;
    }
    buffer = malloc(AOTX_WRITE_BUFFER);
    if (buffer == NULL) {
        fprintf(stderr, "aotx_manifest: the memory for the hash buffer is not there\n");
        return 2;
    }
    rc = aotx_sha256_file(argv[4], entry.sha256, &entry.bytes, buffer, AOTX_WRITE_BUFFER);
    free(buffer);
    if (rc != 0) {
        fprintf(stderr, "aotx_manifest: %s: the file does not hash\n", argv[4]);
        return 2;
    }
    rc = write_entry(argv[2], &entry);
    if (rc != 0) {
        return rc;
    }
    printf("%s %s bytes %llu sha256 %s\n", entry.name, entry.path,
           (unsigned long long)entry.bytes, entry.sha256);
    return 0;
}

static int command_check(const char *dir)
{
    aotx_manifest_entry entries[AOTX_MANIFEST_MAX];
    int count = aotx_manifest_read(dir, entries, AOTX_MANIFEST_MAX);
    int ok = 0;
    int different = 0;
    int missing = 0;
    int i;
    if (count < 0) {
        return 2;
    }
    if (count == 0) {
        fprintf(stderr, "aotx_manifest: %s: the manifest holds no entry\n", dir);
        return 2;
    }
    for (i = 0; i < count; i++) {
        int rc = aotx_manifest_check(dir, &entries[i]);
        const char *word = (rc == 0) ? "ok" : ((rc == 1) ? "different" : "missing");
        if (rc == 0) {
            ok++;
        } else if (rc == 1) {
            different++;
        } else {
            missing++;
        }
        printf("%s %s\n", entries[i].name, word);
    }
    printf("checked %d, ok %d, different %d, missing %d\n", count, ok, different, missing);
    return (ok == count) ? 0 : 1;
}

int main(int argc, char **argv)
{
    if (argc >= 2 && strcmp(argv[1], "write") == 0) {
        if (argc != 8) {
            return usage();
        }
        return command_write(argv);
    }
    if (argc >= 2 && strcmp(argv[1], "check") == 0) {
        if (argc != 3) {
            return usage();
        }
        return command_check(argv[2]);
    }
    return usage();
}
