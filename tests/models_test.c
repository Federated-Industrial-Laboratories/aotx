/* Purpose: Check the five model-store commands over one-entry and 64-entry stores.
 * Owns: One temporary catalog, store, and output file for each batch.
 * Threading: One child command at a time.
 * Lifetime: The test. */
#include "disk/models/models.h"
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
    CHECK(got > 0 && strstr(text, "\"name\":\"language\"") != NULL &&
          strstr(text, "\"path\":\"model-00.gguf\"") != NULL,
          "activate did not write the language manifest line");
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
