/* Purpose: Check the models manifest: the write command, the check command, and the reader.
 * Owns: The temporary directory of one run and the entries that the reader fills.
 * Threading: One thread; the test drives the program and waits for it.
 * Lifetime: The run of the program. */
#include "tests/disk_fake.h"

#include "disk/modelfile/manifest.h"

#include <fcntl.h>

#define AOTX_TEST_SOURCE   "example.test/models"
#define AOTX_TEST_REVISION "main"
#define AOTX_TEST_LICENSE  "Apache-2.0"

static char program[AOTX_PATH_BYTES];

/* Runs the program and gives its exit status. The output goes to the given file. */
static int run(char *const argv[], const char *out_path)
{
    int out_fd = open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    int child;
    int status;
    if (out_fd < 0) {
        return -1;
    }
    child = aotx_spawn(argv, -1, out_fd);
    close(out_fd);
    if (child < 0) {
        return -1;
    }
    status = aotx_wait(child);
    return status;
}

/* Gives one when the file holds the text. */
static int holds(const char *path, const char *want)
{
    char line[AOTX_MANIFEST_LINE];
    FILE *f = fopen(path, "r");
    int found = 0;
    if (f == NULL) {
        return 0;
    }
    while (fgets(line, (int)sizeof(line), f) != NULL) {
        if (strstr(line, want) != NULL) {
            found = 1;
        }
    }
    fclose(f);
    return found;
}

/* Writes one file whose content no other file holds. */
static void make_file(const char *path, int index)
{
    FILE *f = fopen(path, "wb");
    int i;
    int bytes = 64 + index * 37;
    if (f == NULL) {
        return;
    }
    for (i = 0; i < bytes; i++) {
        fputc((index * 131 + i * 17) & 0xff, f);
    }
    fclose(f);
}

/* Writes one line for each file, then checks every file. */
static void batch(int n)
{
    char dir[256];
    char out[512];
    char path[512];
    char name[64];
    aotx_manifest_entry entries[AOTX_MANIFEST_MAX];
    char *argv[9];
    char tally[128];
    int count;
    int i;
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory is not there");
    snprintf(out, sizeof(out), "%s/out.txt", dir);
    for (i = 0; i < n; i++) {
        snprintf(name, sizeof(name), "model-%d.bin", i);
        snprintf(path, sizeof(path), "%s/%s", dir, name);
        make_file(path, i);
        argv[0] = program;
        argv[1] = (char *)"write";
        argv[2] = dir;
        argv[3] = name;
        argv[4] = path;
        argv[5] = (char *)AOTX_TEST_SOURCE;
        argv[6] = (char *)AOTX_TEST_REVISION;
        argv[7] = NULL;
        /* The command takes one more field, so the short form must be refused. */
        if (i == 0) {
            CHECK(run(argv, out) == 2, "a command without the license is accepted");
        }
        argv[7] = (char *)AOTX_TEST_LICENSE;
        argv[8] = NULL;
        CHECK(run(argv, out) == 0, "the write of element %d fails", i);
    }
    count = aotx_manifest_read(dir, entries, AOTX_MANIFEST_MAX);
    CHECK(count == n, "the manifest holds %d lines and not %d", count, n);
    for (i = 0; i < count; i++) {
        snprintf(name, sizeof(name), "model-%d.bin", i);
        CHECK(strcmp(entries[i].name, name) == 0, "line %d holds the name %s", i,
              entries[i].name);
        CHECK(strcmp(entries[i].role, name) == 0, "line %d holds the role %s", i,
              entries[i].role);
        CHECK(strcmp(entries[i].path, name) == 0, "line %d holds the path %s", i,
              entries[i].path);
        CHECK(strcmp(entries[i].source, AOTX_TEST_SOURCE) == 0, "line %d holds another source",
              i);
        CHECK(strcmp(entries[i].revision, AOTX_TEST_REVISION) == 0,
              "line %d holds another revision", i);
        CHECK(strcmp(entries[i].license, AOTX_TEST_LICENSE) == 0,
              "line %d holds another license", i);
        CHECK(entries[i].bytes == (uint64_t)(64 + i * 37), "line %d holds %llu bytes", i,
              (unsigned long long)entries[i].bytes);
        CHECK(strlen(entries[i].sha256) == 64, "line %d holds a short digest", i);
        CHECK(aotx_manifest_check(dir, &entries[i]) == 0, "file %d is not ok", i);
        if (i > 0) {
            CHECK(strcmp(entries[i].sha256, entries[i - 1].sha256) != 0,
                  "file %d has the digest of file %d", i, i - 1);
        }
    }
    argv[0] = program;
    argv[1] = (char *)"check";
    argv[2] = dir;
    argv[3] = NULL;
    CHECK(run(argv, out) == 0, "the check of %d files fails", n);
    snprintf(tally, sizeof(tally), "checked %d, ok %d, different 0, missing 0", n, n);
    CHECK(holds(out, tally) == 1, "the check does not report %s", tally);
    /* One byte of one file changes, so the check must see the difference. */
    snprintf(path, sizeof(path), "%s/model-0.bin", dir);
    {
        FILE *f = fopen(path, "r+b");
        CHECK(f != NULL, "the file does not open to write");
        if (f != NULL) {
            fseek(f, 3, SEEK_SET);
            fputc(0xa5, f);
            fclose(f);
        }
    }
    CHECK(aotx_manifest_check(dir, &entries[0]) == 1, "a changed byte is not seen");
    CHECK(run(argv, out) == 1, "the check of a changed file gives success");
    CHECK(holds(out, "model-0.bin different") == 1, "the check does not report the difference");
    /* The file goes away, so the check must report it as missing. */
    CHECK(remove(path) == 0, "the file does not go away");
    CHECK(aotx_manifest_check(dir, &entries[0]) == 2, "a file that is not there is ok");
    CHECK(run(argv, out) == 1, "the check of a file that is not there gives success");
    CHECK(holds(out, "model-0.bin missing") == 1, "the check does not report the file");
    aotx_remove_tree(dir);
}

/* The cases that a command must refuse. */
static void refusals(void)
{
    char dir[256];
    char out[512];
    char path[512];
    char second[512];
    char line[AOTX_MANIFEST_LINE];
    aotx_manifest_entry entries[AOTX_MANIFEST_MAX];
    aotx_manifest_entry one;
    char *argv[9];
    FILE *f;
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory is not there");
    snprintf(out, sizeof(out), "%s/out.txt", dir);
    snprintf(path, sizeof(path), "%s/model.bin", dir);
    make_file(path, 1);
    argv[0] = program;
    argv[1] = (char *)"write";
    argv[2] = dir;
    argv[3] = (char *)"model.bin";
    argv[4] = path;
    argv[5] = (char *)AOTX_TEST_SOURCE;
    argv[6] = (char *)AOTX_TEST_REVISION;
    argv[7] = (char *)AOTX_TEST_LICENSE;
    argv[8] = NULL;
    CHECK(run(argv, out) == 0, "the first write fails");
    /* One name names one file, so a name that the manifest holds is refused. */
    snprintf(second, sizeof(second), "%s/second.bin", dir);
    make_file(second, 2);
    argv[4] = second;
    CHECK(run(argv, out) == 1, "a second file with the same name is accepted");
    /* One file has one line, so a second name for the same file is refused. */
    argv[3] = (char *)"another-name";
    argv[4] = path;
    CHECK(run(argv, out) == 1, "a second name for the same file is accepted");
    /* A field that holds a quotation mark would break the line, so it is refused. */
    argv[5] = (char *)"a \"source\"";
    CHECK(run(argv, out) == 2, "a field with a quotation mark is accepted");
    argv[5] = (char *)AOTX_TEST_SOURCE;
    /* A file that is not in the directory cannot be checked, so the write refuses it. */
    argv[3] = (char *)"outside.bin";
    argv[4] = (char *)"/etc/hostname";
    CHECK(run(argv, out) == 2, "a file outside the directory is accepted");
    argv[1] = (char *)"other";
    CHECK(run(argv, out) == 2, "a command that is not known is accepted");
    /* A line that the writer did not write must not read. */
    CHECK(aotx_manifest_path(line, sizeof(line), dir, AOTX_MANIFEST_NAME) == 0,
          "the manifest path does not join");
    f = fopen(line, "a");
    CHECK(f != NULL, "the manifest does not open");
    if (f != NULL) {
        fputs("{\"name\":\"bad\",\"path\":\"bad.bin\"}\n", f);
        fclose(f);
    }
    CHECK(aotx_manifest_read(dir, entries, AOTX_MANIFEST_MAX) == -1, "a short line reads");
    argv[0] = program;
    argv[1] = (char *)"check";
    argv[2] = dir;
    argv[3] = NULL;
    CHECK(run(argv, out) == 2, "a check of a manifest with a bad line gives success");
    /* The reader refuses a digest that is not 64 characters of the low case. */
    memset(&one, 0, sizeof(one));
    CHECK(aotx_manifest_line("{\"name\":\"a\",\"path\":\"b\",\"source\":\"c\",\"revision\":\"d\","
                             "\"license\":\"e\",\"bytes\":1,\"sha256\":\"AB\"}", &one) == -1,
          "a short digest reads");
    CHECK(aotx_manifest_field("plain text") == 0, "a plain field is refused");
    CHECK(aotx_manifest_field("a\nline") == -1, "a field with a line end is accepted");
    aotx_remove_tree(dir);
}

/* Checks the manifest of the model files when one is there. */
static int models(const char *dir)
{
    char path[512];
    char out[512];
    char *argv[4];
    int status;
    snprintf(path, sizeof(path), "%s/%s", dir, AOTX_MANIFEST_NAME);
    if (access(path, R_OK) != 0) {
        printf("manifest_test: skip, %s is not there\n", path);
        return 1;
    }
    snprintf(out, sizeof(out), "%s", "/dev/null");
    argv[0] = program;
    argv[1] = (char *)"check";
    argv[2] = (char *)dir;
    argv[3] = NULL;
    status = run(argv, out);
    CHECK(status == 0, "the check of the model files gives %d", status);
    printf("manifest_test: the model files check with the status %d\n", status);
    return 0;
}

int main(int argc, char **argv)
{
    int skipped;
    if (argc < 2) {
        printf("usage: manifest_test <aotx_manifest> [models directory]\n");
        return 1;
    }
    snprintf(program, sizeof(program), "%s", argv[1]);
    batch(1);
    batch(64);
    refusals();
    skipped = models((argc > 2) ? argv[2] : "models");
    printf("manifest_test: skipped %d\n", skipped);
    return aotx_report("manifest_test", 400);
}
