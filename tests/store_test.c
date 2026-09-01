/* Purpose: Check store states, part files, digest differences, and local record lines.
 * Owns: One temporary store for each batch.
 * Threading: One thread.
 * Lifetime: The test. */
#include "disk/models/models.h"
#include "disk/modelfile/manifest.h"
#include "disk/wire/diskwire.h"
#include "tests/disk_fake.h"

#include <fcntl.h>

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

typedef struct parameter_file { unsigned char data[2048]; size_t used; } parameter_file;

static void parameter_number(parameter_file *file, uint64_t value, unsigned int bytes)
{
    for (unsigned int i = 0u; i < bytes; ++i)
        file->data[file->used++] = (unsigned char)(value >> (8u * i));
}

static void parameter_text(parameter_file *file, const char *text)
{
    size_t bytes = strlen(text);
    parameter_number(file, bytes, 8u);
    memcpy(file->data + file->used, text, bytes);
    file->used += bytes;
}

static void parameter_real(parameter_file *file, const char *key, float value)
{
    uint32_t raw;
    memcpy(&raw, &value, sizeof raw);
    parameter_text(file, key);
    parameter_number(file, AOTX_GGUF_F32, 4u);
    parameter_number(file, raw, 4u);
}

static void parameter_whole(parameter_file *file, const char *key, uint32_t value)
{
    parameter_text(file, key);
    parameter_number(file, AOTX_GGUF_U32, 4u);
    parameter_number(file, value, 4u);
}

static void parameter_discovery(void)
{
    parameter_file file = { { 0 }, 0u };
    aotx_model_catalog catalog;
    char dir[128], path[AOTX_MODEL_PATH], line[4096], reason[192];
    memset(&catalog, 0, sizeof catalog);
    memcpy(file.data + file.used, "GGUF", 4u); file.used += 4u;
    parameter_number(&file, AOTX_GGUF_VERSION, 4u);
    parameter_number(&file, 0u, 8u);
    parameter_number(&file, 7u, 8u);
    parameter_real(&file, "sampling.temperature.default", 0.7f);
    parameter_real(&file, "sampling.temperature.min", 0.1f);
    parameter_real(&file, "sampling.temperature.max", 2.0f);
    parameter_whole(&file, "sampling.top_k.default", 40u);
    parameter_whole(&file, "sampling.top_k.min", 1u);
    parameter_whole(&file, "sampling.top_k.max", 100u);
    parameter_real(&file, "sampling.top_p.default", 0.9f);
    while ((file.used & 31u) != 0u) file.data[file.used++] = 0u;
    CHECK(aotx_temp_dir(dir, sizeof dir) == 0, "the parameter store does not open");
    snprintf(path, sizeof path, "%s/declared.gguf", dir);
    put(path, file.data, file.used);
    snprintf(catalog.entry[0].name, sizeof catalog.entry[0].name, "declared");
    snprintf(catalog.entry[0].file, sizeof catalog.entry[0].file, "declared.gguf");
    catalog.count = 1u;
    CHECK(aotx_model_parameters_scan(dir, &catalog, reason, sizeof reason) == 0,
          "the parameter catalog does not scan: %s", reason);
    snprintf(path, sizeof path, "%s/parameters.jsonl", dir);
    int fd = open(path, O_RDONLY);
    ssize_t got = fd >= 0 ? read(fd, line, sizeof line - 1u) : -1;
    if (fd >= 0) close(fd);
    if (got > 0) line[got] = '\0';
    CHECK(got > 0 && strstr(line, "\"temperature\":{\"default\":0.699999988,"
          "\"min\":0.100000001,\"max\":2}") != NULL,
          "the declared real parameter is absent");
    CHECK(got > 0 && strstr(line, "\"top_k\":{\"default\":40,\"min\":1,\"max\":100}")
          != NULL, "the declared whole parameter is absent");
    CHECK(got > 0 && strstr(line, "top_p") == NULL,
          "an incomplete parameter triplet entered the catalog");
    aotx_remove_tree(dir);
    printf("parameter discovery: two complete fields, one mutation absent\n");
}

static void copy_text(char *out, size_t out_bytes, const char *text)
{
    size_t bytes = strlen(text);
    CHECK(bytes < out_bytes, "the fixture text does not fit");
    if (bytes >= out_bytes) {
        bytes = out_bytes - 1u;
    }
    memcpy(out, text, bytes);
    out[bytes] = '\0';
}

static void fill_entry(aotx_model_catalog_entry *entry, int index,
                       const void *data, size_t bytes)
{
    memset(entry, 0, sizeof(*entry));
    snprintf(entry->name, sizeof(entry->name), "model-%02d", index);
    snprintf(entry->role, sizeof(entry->role), "language");
    snprintf(entry->repository, sizeof(entry->repository), "source/model-%02d", index);
    snprintf(entry->file, sizeof(entry->file), "model-%02d.gguf", index);
    snprintf(entry->revision, sizeof(entry->revision), "%040d", index + 1);
    entry->bytes = bytes;
    digest_of(data, bytes, entry->sha256);
    snprintf(entry->license, sizeof(entry->license), "Apache-2.0");
    snprintf(entry->quant, sizeof(entry->quant), "Q8_0");
    snprintf(entry->profiles, sizeof(entry->profiles), "8g");
    snprintf(entry->source, sizeof(entry->source), "source/model-%02d", index);
}

static void write_manifest(const char *dir, const aotx_model_catalog_entry *entry)
{
    aotx_manifest_entry manifest;
    char path[AOTX_MODEL_PATH];
    char line[AOTX_MODEL_LINE];
    memset(&manifest, 0, sizeof(manifest));
    snprintf(manifest.name, sizeof(manifest.name), "%s", entry->name);
    snprintf(manifest.role, sizeof(manifest.role), "%s", entry->role);
    snprintf(manifest.path, sizeof(manifest.path), "%s", entry->file);
    copy_text(manifest.source, sizeof(manifest.source), entry->source);
    snprintf(manifest.revision, sizeof(manifest.revision), "%s", entry->revision);
    copy_text(manifest.license, sizeof(manifest.license), entry->license);
    manifest.bytes = entry->bytes;
    snprintf(manifest.sha256, sizeof(manifest.sha256), "%s", entry->sha256);
    CHECK(aotx_manifest_write_line(line, sizeof(line), &manifest) == 0,
          "the manifest line does not write");
    snprintf(path, sizeof(path), "%s/manifest.jsonl", dir);
    put(path, line, strlen(line));
}

static void append_local(const char *dir, const aotx_model_catalog_entry *entry, int wrong)
{
    aotx_model_store_record record;
    char reason[128];
    memset(&record, 0, sizeof(record));
    snprintf(record.name, sizeof(record.name), "%s", entry->name);
    snprintf(record.file, sizeof(record.file), "%s", entry->file);
    record.bytes = entry->bytes;
    snprintf(record.sha256, sizeof(record.sha256), "%s", entry->sha256);
    if (wrong) {
        record.sha256[0] = (record.sha256[0] == '0') ? '1' : '0';
    }
    snprintf(record.source, sizeof(record.source), "https://source/%s", entry->file);
    snprintf(record.date, sizeof(record.date), "2000-01-01T00:%02d:00Z", wrong);
    snprintf(record.revision, sizeof(record.revision), "%s", entry->revision);
    record.verified = 1;
    CHECK(aotx_model_store_append(dir, &record, reason, sizeof(reason)) == 0,
          "the store record does not append: %s", reason);
}

static void round_trip(void)
{
    aotx_model_store_record one;
    aotx_model_store_record two;
    char line[AOTX_MODEL_LINE];
    memset(&one, 0, sizeof(one));
    snprintf(one.name, sizeof(one.name), "round-trip");
    snprintf(one.file, sizeof(one.file), "round-trip.gguf");
    one.bytes = 987654321ull;
    snprintf(one.sha256, sizeof(one.sha256), "%064d", 7);
    snprintf(one.source, sizeof(one.source), "https://source/revision/round-trip.gguf");
    snprintf(one.date, sizeof(one.date), "2000-01-01T00:30:00Z");
    snprintf(one.revision, sizeof(one.revision), "%040d", 8);
    one.verified = 1;
    CHECK(aotx_model_store_write_line(line, sizeof(line), &one) == 0,
          "the local store line does not write");
    CHECK(aotx_model_store_line(line, &two) == 0, "the local store line does not read");
    CHECK(strcmp(one.name, two.name) == 0 && strcmp(one.file, two.file) == 0 &&
          one.bytes == two.bytes && strcmp(one.sha256, two.sha256) == 0 &&
          strcmp(one.source, two.source) == 0 && strcmp(one.date, two.date) == 0 &&
          strcmp(one.revision, two.revision) == 0 && two.verified == 1,
          "the local store line changed on its round trip");
}

static void batch(int n)
{
    aotx_model_catalog catalog;
    aotx_model_view view[AOTX_MODEL_CATALOG_MAX + 2u];
    char dir[128];
    char path[AOTX_MODEL_PATH];
    char reason[192];
    unsigned char content[128];
    int final_count = (n == 1) ? 1 : n / 2;
    int i;
    int count;
    memset(&catalog, 0, sizeof(catalog));
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary store does not open");
    for (i = 0; i < n; i++) {
        size_t bytes = (size_t)i + 1u;
        memset(content, 'a' + (i % 26), bytes);
        fill_entry(&catalog.entry[i], i, content, bytes);
        if (i < final_count) {
            snprintf(path, sizeof(path), "%s/%s", dir, catalog.entry[i].file);
            put(path, content, bytes);
            if (i > 0) {
                append_local(dir, &catalog.entry[i], i == 1);
            }
        }
    }
    catalog.count = (unsigned int)n;
    write_manifest(dir, &catalog.entry[0]);
    if (n > 1) {
        snprintf(path, sizeof(path), "%s/%s.part", dir, catalog.entry[final_count].file);
        put(path, "part-content", 12u);
        snprintf(path, sizeof(path), "%s/other.gguf", dir);
        put(path, "other", 5u);
    }
    count = aotx_model_store_scan(dir, &catalog, view, AOTX_MODEL_CATALOG_MAX + 2u,
                                  reason, sizeof(reason));
    CHECK(count == n + ((n > 1) ? 1 : 0), "the batch %d gives %d rows: %s", n, count,
          reason);
    CHECK(view[0].state == AOTX_MODEL_ON_DISK, "the active file has state %d", view[0].state);
    if (n > 1) {
        CHECK(view[1].state == AOTX_MODEL_DIGEST_DIFFERS,
              "the wrong local digest has state %d", view[1].state);
        for (i = 2; i < final_count; i++) {
            CHECK(view[i].state == AOTX_MODEL_NOT_ACTIVE && view[i].verified == 1,
                  "the fetched file %d has state %d and verified %d", i,
                  view[i].state, view[i].verified);
        }
        CHECK(view[final_count].state == AOTX_MODEL_FETCHING &&
              view[final_count].bytes_on_disk == 12u,
              "the part has state %d and %llu bytes", view[final_count].state,
              (unsigned long long)view[final_count].bytes_on_disk);
        for (i = final_count + 1; i < n; i++) {
            CHECK(view[i].state == AOTX_MODEL_NOT_FETCHED,
                  "the absent file %d has state %d", i, view[i].state);
        }
        CHECK(view[n].state == AOTX_MODEL_NOT_ACTIVE,
              "the file outside the catalog has state %d", view[n].state);
    }
    aotx_remove_tree(dir);
    printf("store batch %d: rows %d\n", n, count);
}

static void one_line_for_each_role(void)
{
    aotx_model_catalog catalog;
    char dir[128];
    char path[AOTX_MODEL_PATH];
    char reason[192];
    char text[4096];
    char *at;
    FILE *in;
    size_t got;
    int lines = 0;
    int i;
    memset(&catalog, 0, sizeof(catalog));
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary store does not open");
    for (i = 0; i < 2; i++) {
        unsigned char content[8];
        size_t bytes = (size_t)i + 4u;
        memset(content, 'r' + i, bytes);
        fill_entry(&catalog.entry[i], i, content, bytes);
        snprintf(path, sizeof(path), "%s/%s", dir, catalog.entry[i].file);
        put(path, content, bytes);
    }
    catalog.count = 2u;
    CHECK(aotx_model_store_activate(dir, &catalog.entry[0], "language",
                                    reason, sizeof(reason)) == 0,
          "the first activation refuses: %s", reason);
    CHECK(aotx_model_store_activate(dir, &catalog.entry[1], "language",
                                    reason, sizeof(reason)) == 0,
          "the second activation refuses: %s", reason);
    snprintf(path, sizeof(path), "%s/manifest.jsonl", dir);
    in = fopen(path, "r");
    CHECK(in != NULL, "the manifest does not open");
    got = in != NULL ? fread(text, 1u, sizeof(text) - 1u, in) : 0u;
    if (in != NULL) fclose(in);
    text[got] = '\0';
    for (at = text; (at = strstr(at, "\"role\":\"language\"")) != NULL; at++) lines++;
    CHECK(lines == 1, "the manifest holds %d language lines and 1 is the bound", lines);
    CHECK(strstr(text, "model-01") != NULL, "the manifest does not name the new model");
    CHECK(strstr(text, "model-00") == NULL, "the old model line did not leave");
    aotx_remove_tree(dir);
    printf("manifest role lines: 1\n");
}

int main(void)
{
    round_trip();
    parameter_discovery();
    batch(1);
    batch(64);
    one_line_for_each_role();
    return aotx_report("store_test", 157);
}
