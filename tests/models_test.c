/* Purpose: Check the five model-store commands over one-entry and 64-entry stores.
 * Owns: One temporary catalog, store, and output file for each batch.
 * Threading: One child command at a time.
 * Lifetime: The test. */
#include "disk/models/models.h"
#include "disk/modelfile/manifest.h"
#include "disk/wire/diskwire.h"
#include "tests/disk_fake.h"

#include <fcntl.h>

static const char *program;

static void digest_of(const void *data, size_t bytes, char out[AOTX_SHA256_HEX])
{
    aotx_sha256 state;
    unsigned char raw[AOTX_SHA256_DIGEST];
    aotx_sha256_init(&state);
    aotx_sha256_update(&state, data, bytes);
    aotx_sha256_final(&state, raw);
    aotx_sha256_text(raw, out);
}

static void put(const char *path, const void *data, size_t bytes)
{
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    CHECK(fd >= 0, "the file %s does not open", path);
    if (fd >= 0) {
        CHECK(write(fd, data, bytes) == (ssize_t)bytes, "the file %s does not write", path);
        close(fd);
    }
}

static void copy_field(char *out, size_t out_bytes, const char *in)
{
    size_t bytes = strlen(in) + 1u;
    CHECK(bytes <= out_bytes, "the fixture field does not fit");
    if (bytes <= out_bytes) {
        memcpy(out, in, bytes);
    }
}

static int run_command(char *const args[], const char *output)
{
    int fd = open(output, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    int child;
    int state;
    CHECK(fd >= 0, "the command output does not open");
    child = aotx_spawn(args, -1, fd);
    if (fd >= 0) {
        close(fd);
    }
    CHECK(child > 0, "the model command does not start");
    state = (child > 0) ? aotx_wait(child) : -1;
    return state;
}

static void make_fixture(const char *dir, const char *catalog_path, int n)
{
    FILE *catalog = fopen(catalog_path, "w");
    unsigned char content[128];
    char source[192];
    int i;
    CHECK(catalog != NULL, "the command catalog does not open");
    snprintf(source, sizeof(source), "%s/source", dir);
    CHECK(mkdir(source, 0700) == 0, "the command source does not open");
    for (i = 0; i < n; i++) {
        char digest[AOTX_SHA256_HEX];
        char path[256];
        size_t bytes = (size_t)i + 1u;
        memset(content, 'A' + (i % 26), bytes);
        digest_of(content, bytes, digest);
        snprintf(path, sizeof(path), "%s/model-%02d.gguf", dir, i);
        put(path, content, bytes);
        snprintf(path, sizeof(path), "%s/source/%040d", dir, i + 1);
        CHECK(mkdir(path, 0700) == 0, "source revision %d does not open", i);
        snprintf(path, sizeof(path), "%s/source/%040d/model-%02d.gguf", dir, i + 1, i);
        put(path, content, bytes);
        if (catalog != NULL) {
            fprintf(catalog, "{\"name\":\"model-%02d\",\"role\":\"language\","
                    "\"repository\":\"file://localhost%s/source\","
                    "\"file\":\"model-%02d.gguf\","
                    "\"revision\":\"%040d\",\"bytes\":%zu,\"sha256\":\"%s\","
                    "\"license\":\"Apache-2.0\",\"quant\":\"Q8_0\","
                    "\"profiles\":\"8g\",\"verified\":false,"
                    "\"source\":\"source/model-%02d\",\"note\":\"batch %d entry %d\"}\n",
                    i, dir, i, i + 1, bytes, digest, i, n, i);
        }
    }
    if (catalog != NULL) {
        fclose(catalog);
    }
}

static void seed_manifest(const char *dir, const char *catalog_path, int n)
{
    aotx_model_catalog catalog;
    aotx_manifest_entry entry;
    char line[AOTX_MANIFEST_LINE];
    char path[192];
    char reason[192];
    int fd;
    CHECK(aotx_model_catalog_read(catalog_path, &catalog, reason, sizeof(reason)) == n,
          "the seed catalog does not read: %s", reason);
    memset(&entry, 0, sizeof(entry));
    copy_field(entry.name, sizeof(entry.name), catalog.entry[0].name);
    snprintf(entry.role, sizeof(entry.role), "reranker");
    copy_field(entry.path, sizeof(entry.path), catalog.entry[0].file);
    copy_field(entry.source, sizeof(entry.source), catalog.entry[0].source);
    copy_field(entry.revision, sizeof(entry.revision), catalog.entry[0].revision);
    copy_field(entry.license, sizeof(entry.license), catalog.entry[0].license);
    entry.bytes = catalog.entry[0].bytes;
    copy_field(entry.sha256, sizeof(entry.sha256), catalog.entry[0].sha256);
    CHECK(aotx_manifest_write_line(line, sizeof(line), &entry) == 0,
          "the seed manifest line does not write");
    snprintf(path, sizeof(path), "%s/manifest.jsonl", dir);
    fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    CHECK(fd >= 0 && write(fd, line, strlen(line)) == (ssize_t)strlen(line),
          "the seed manifest does not write");
    if (n > 1 && fd >= 0) {
        copy_field(entry.name, sizeof(entry.name), catalog.entry[1].name);
        copy_field(entry.role, sizeof(entry.role), catalog.entry[1].role);
        copy_field(entry.path, sizeof(entry.path), catalog.entry[1].file);
        copy_field(entry.source, sizeof(entry.source), catalog.entry[1].source);
        copy_field(entry.revision, sizeof(entry.revision), catalog.entry[1].revision);
        copy_field(entry.license, sizeof(entry.license), catalog.entry[1].license);
        entry.bytes = catalog.entry[1].bytes;
        copy_field(entry.sha256, sizeof(entry.sha256), catalog.entry[1].sha256);
        CHECK(aotx_manifest_write_line(line, sizeof(line), &entry) == 0
              && write(fd, line, strlen(line)) == (ssize_t)strlen(line),
              "the second seed manifest line does not write");
    }
    if (fd >= 0) {
        close(fd);
    }
}

static int occurrences(const char *text, const char *part)
{
    int count = 0;
    size_t bytes = strlen(part);
    while ((text = strstr(text, part)) != NULL) {
        count++;
        text += bytes;
    }
    return count;
}

static void batch(int n)
{
    char dir[128];
    char catalog[192];
    char output[192];
    char name[32];
    char *list_args[7];
    char *fetch_args[8];
    char *activate_args[9];
    char *check_args[6];
    char *remove_args[8];
    char *usage_args[8];
    char text[4096];
    int fd;
    ssize_t got;
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the command store does not open");
    snprintf(catalog, sizeof(catalog), "%s/catalog.jsonl", dir);
    snprintf(output, sizeof(output), "%s/output.txt", dir);
    make_fixture(dir, catalog, n);
    seed_manifest(dir, catalog, n);

    list_args[0] = (char *)program; list_args[1] = (char *)"--dir"; list_args[2] = dir;
    list_args[3] = (char *)"--catalog"; list_args[4] = catalog;
    list_args[5] = (char *)"list"; list_args[6] = NULL;
    CHECK(run_command(list_args, output) == 0, "list failed for batch %d", n);

    usage_args[0] = (char *)program; usage_args[1] = (char *)"list";
    usage_args[2] = (char *)"extra"; usage_args[3] = NULL;
    CHECK(run_command(usage_args, output) == 2,
          "a command with an extra argument did not give status 2");

    fetch_args[0] = (char *)program; fetch_args[1] = (char *)"--dir"; fetch_args[2] = dir;
    fetch_args[3] = (char *)"--catalog"; fetch_args[4] = catalog;
    fetch_args[5] = (char *)"fetch"; fetch_args[6] = (char *)"model-00";
    fetch_args[7] = NULL;
#ifdef AOTX_FETCH_TEST
    CHECK(run_command(fetch_args, output) == 0, "fetch failed for batch %d", n);
    {
        aotx_model_catalog read_catalog;
        aotx_model_store_record record[2];
        char reason[192];
        int records = aotx_model_store_read(dir, record, 2u);
        CHECK(aotx_model_catalog_read(catalog, &read_catalog, reason, sizeof(reason)) == n,
              "the catalog after fetch does not read: %s", reason);
        CHECK(read_catalog.entry[0].verified == 0,
              "fetch changed the catalog verification fact");
        CHECK(records == 1 && record[0].verified == 1,
              "fetch did not put verification in the local store");
    }
#else
    CHECK(run_command(fetch_args, output) == 1,
          "fetch without libcurl did not give status 1");
#endif
    fetch_args[6] = (char *)"not-known";
    CHECK(run_command(fetch_args, output) == 1,
          "fetch of an unknown name did not give status 1");

    activate_args[0] = (char *)program; activate_args[1] = (char *)"--dir";
    activate_args[2] = dir; activate_args[3] = (char *)"--catalog";
    activate_args[4] = catalog; activate_args[5] = (char *)"activate";
    activate_args[6] = (char *)"language"; activate_args[7] = (char *)"model-00";
    activate_args[8] = NULL;
    CHECK(run_command(activate_args, output) == 0, "activate failed for batch %d", n);

    check_args[0] = (char *)program; check_args[1] = (char *)"--dir";
    check_args[2] = dir; check_args[3] = (char *)"check"; check_args[4] = NULL;
    check_args[5] = NULL;
    CHECK(run_command(check_args, output) == 0, "check failed for batch %d", n);

    remove_args[0] = (char *)program; remove_args[1] = (char *)"--dir";
    remove_args[2] = dir; remove_args[3] = (char *)"--catalog";
    remove_args[4] = catalog; remove_args[5] = (char *)"remove";
    remove_args[6] = (char *)"model-00"; remove_args[7] = NULL;
    CHECK(run_command(remove_args, output) == 1,
          "remove did not refuse the active file for batch %d", n);
    if (n > 1) {
        snprintf(name, sizeof(name), "model-%02d", n - 1);
        remove_args[6] = name;
        CHECK(run_command(remove_args, output) == 0,
              "remove failed for the inactive file %s", name);
        snprintf(text, sizeof(text), "%s/model-%02d.gguf", dir, n - 1);
        CHECK(access(text, F_OK) != 0, "remove left %s", text);
    }
    snprintf(text, sizeof(text), "%s/manifest.jsonl", dir);
    fd = open(text, O_RDONLY);
    got = (fd >= 0) ? read(fd, text, sizeof(text) - 1u) : -1;
    if (fd >= 0) {
        close(fd);
    }
    if (got > 0) {
        text[got] = '\0';
    }
    CHECK(got > 0 && strstr(text, "\"name\":\"model-00\"") != NULL &&
          strstr(text, "\"role\":\"language\"") != NULL &&
          strstr(text, "\"path\":\"model-00.gguf\"") != NULL,
          "activate did not write the named language manifest line");
    CHECK(got > 0 && occurrences(text, "\"name\":\"model-00\"") == 1,
          "activate appended the manifest name instead of updating it");
    /* The manifest keeps one line for each role. An activation under the language role
     * takes the place of the line that held the role before it. */
    CHECK(got > 0 && occurrences(text, "\"role\":\"language\"") == 1,
          "activate left two names under the language role");
    aotx_remove_tree(dir);
    printf("models batch %d: five commands\n", n);
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        return 2;
    }
    program = argv[1];
    batch(1);
    batch(64);
    return aotx_report("models_test", 160);
}
