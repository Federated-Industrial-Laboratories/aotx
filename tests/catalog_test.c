/* Purpose: Check every repository catalog field and reject malformed catalog lines.
 * Owns: One temporary catalog file and one catalog table.
 * Threading: One thread.
 * Lifetime: The test. */
#include "disk/models/models.h"
#include "tests/disk_fake.h"

#include <fcntl.h>

typedef struct known_entry {
    const char *name;
    const char *role;
    const char *file;
    const char *revision;
    uint64_t bytes;
    const char *sha256;
    const char *source;
    int verified;
} known_entry;

static const known_entry known[] = {
    { "embedding", "embedding", "Qwen3-Embedding-0.6B-Q8_0.gguf",
      "370f27d7550e0def9b39c1f16d3fbaa13aa67728", 639150592ull,
      "06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439",
      "Qwen/Qwen3-Embedding-0.6B-GGUF", 1 },
    { "reranker", "reranker", "qwen3-reranker-0.6b-q8_0.gguf",
      "a02f48bb4f057028298c21fa033da2b30d7742d5", 639153184ull,
      "22c9979ce4fbcdc5acdc310c6641c32797eff1aa980b8f7a2db8a8ea23429a48",
      "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF", 1 },
    { "language", "language", "Qwen3-4B-Q8_0.gguf",
      "bc640142c66e1fdd12af0bd68f40445458f3869b", 4280404704ull,
      "8c2f07f26af9747e41988551106f149b03eb9b5cb6df636027b6bf6278473300",
      "Qwen/Qwen3-4B-GGUF", 1 },
    { "language-q4", "language-q4", "Qwen3-4B-Q4_0.gguf",
      "bc640142c66e1fdd12af0bd68f40445458f3869b", 2463746784ull,
      "ae782b4a90b57dc4faf880855ea4b285a96b4a335544144526dab73df91d4d61",
      "Qwen/Qwen3-4B-GGUF, requantized with llama-quantize at commit 6c84c7d, token embedding q8_0", 1 },
    { "qwen3-0.6b-q8-0", "language", "Qwen3-0.6B-Q8_0.gguf",
      "23749fefcc72300e3a2ad315e1317431b06b590a", 639446688ull,
      "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031",
      "Qwen/Qwen3-0.6B-GGUF", 0 },
    { "qwen3-1.7b-q8-0", "language", "Qwen3-1.7B-Q8_0.gguf",
      "90862c4b9d2787eaed51d12237eafdfe7c5f6077", 1834426016ull,
      "061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a",
      "Qwen/Qwen3-1.7B-GGUF", 0 },
    { "qwen3-8b-q8-0", "language", "Qwen3-8B-Q8_0.gguf",
      "7c41481f57cb95916b40956ab2f0b139b296d974", 8709518112ull,
      "408b955510e196121c1c375201744783b5c9a43c7956d73fc78df54c66e883d6",
      "Qwen/Qwen3-8B-GGUF", 0 },
    { "qwen3-14b-q8-0", "language", "Qwen3-14B-Q8_0.gguf",
      "530227a7d994db8eca5ab5ced2fb692b614357fd", 15698533728ull,
      "a0dfe649137410b7d82f06a209240508e218f32f5b6fd81b69d6932160cfcd9d",
      "Qwen/Qwen3-14B-GGUF", 0 },
    { "qwen3-0.6b-q4-0", "language-q4", "Qwen3-0.6B-Q4_0.gguf",
      "50968a4468ef4233ed78cd7c3de230dd1d61a56b", 382156480ull,
      "33bcc57074ec7b6eada5a90651ee546ec0c2b271002c22baf9f1b2dd1e8f75cb",
      "unsloth/Qwen3-0.6B-GGUF", 0 },
    { "qwen3-1.7b-q4-0", "language-q4", "Qwen3-1.7B-Q4_0.gguf",
      "d7f544eead698dbd1f15126ef60b45a1e1933222", 1056782912ull,
      "c876f159707a4e4f70e045106c69db15bfc935a4981706fd4f65c6e7ea1e81c5",
      "unsloth/Qwen3-1.7B-GGUF", 0 },
    { "qwen3-8b-q4-0", "language-q4", "Qwen_Qwen3-8B-Q4_0.gguf",
      "0b69f75b7472688e6808490aa2b85efdb81b5ce7", 4787332640ull,
      "c9bd6e5597ad3a70ef5b9a0acb995f739e545c5f16cbdf52b1c78835172b893c",
      "bartowski/Qwen_Qwen3-8B-GGUF", 0 },
    { "qwen3-14b-q4-0", "language-q4", "Qwen3-14B-Q4_0.gguf",
      "a04a82c4739b3ef5fa6da7d10261db2c67dd1985", 8543001984ull,
      "009f54ffc8d8082e7921139924229d4deea61c9174a0a357d91384bd299ff78e",
      "unsloth/Qwen3-14B-GGUF", 0 }
};

static void real_catalog(const char *path)
{
    aotx_model_catalog catalog;
    char reason[192];
    unsigned int i;
    CHECK(aotx_model_catalog_read(path, &catalog, reason, sizeof(reason)) ==
          (int)(sizeof(known) / sizeof(known[0])), "the catalog does not read: %s", reason);
    for (i = 0u; i < sizeof(known) / sizeof(known[0]) && i < catalog.count; i++) {
        const aotx_model_catalog_entry *got = &catalog.entry[i];
        const char *repository = (i == 3u) ? "Qwen/Qwen3-4B-GGUF" : known[i].source;
        const char *license = (i == 2u || i == 3u)
                            ? "Apache-2.0 (repository card)" : "Apache-2.0";
        const char *quant = (strcmp(known[i].role, "language-q4") == 0)
                          ? "Q4_0" : "Q8_0";
        const char *profiles = (i < 4u) ? "reference"
                             : ((i == 4u || i == 5u || i == 8u || i == 9u)
                                ? "8g" : "24g");
        const char *note = (i == 3u)
                         ? "made offline from the official Q8_0"
                         : ((i >= 8u) ? "community conversion" : "");
        CHECK(strcmp(got->name, known[i].name) == 0, "entry %u name is %s", i, got->name);
        CHECK(strcmp(got->role, known[i].role) == 0, "entry %u role is %s", i, got->role);
        CHECK(strcmp(got->repository, repository) == 0,
              "entry %u repository is %s", i, got->repository);
        CHECK(strcmp(got->file, known[i].file) == 0, "entry %u file is %s", i, got->file);
        CHECK(strcmp(got->revision, known[i].revision) == 0,
              "entry %u revision is %s", i, got->revision);
        CHECK(got->bytes == known[i].bytes, "entry %u bytes are %llu", i,
              (unsigned long long)got->bytes);
        CHECK(strcmp(got->sha256, known[i].sha256) == 0,
              "entry %u digest is %s", i, got->sha256);
        CHECK(strcmp(got->source, known[i].source) == 0,
              "entry %u source is %s", i, got->source);
        CHECK(strcmp(got->license, license) == 0, "entry %u license is %s", i,
              got->license);
        CHECK(strcmp(got->quant, quant) == 0, "entry %u quant is %s", i, got->quant);
        CHECK(strcmp(got->profiles, profiles) == 0, "entry %u profiles are %s", i,
              got->profiles);
        CHECK(got->verified == known[i].verified, "entry %u verified is %d", i,
              got->verified);
        CHECK(strcmp(got->note, note) == 0, "entry %u note is %s", i, got->note);
    }
}

static void batch(int n)
{
    aotx_model_catalog catalog;
    char dir[128];
    char path[192];
    char reason[192];
    FILE *file;
    int i;
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    snprintf(path, sizeof(path), "%s/catalog.jsonl", dir);
    file = fopen(path, "w");
    CHECK(file != NULL, "the fixture catalog does not open");
    for (i = 0; file != NULL && i < n; i++) {
        fprintf(file, "{\"name\":\"model-%02d\",\"role\":\"language\","
                "\"repository\":\"source/model-%02d\",\"file\":\"model-%02d.gguf\","
                "\"revision\":\"%040d\",\"bytes\":%d,"
                "\"sha256\":\"%064d\",\"license\":\"Apache-2.0\","
                "\"quant\":\"Q8_0\",\"profiles\":\"8g\",\"verified\":false,"
                "\"source\":\"source/model-%02d\",\"note\":\"entry %02d\"}\n",
                i, i, i, i + 1, i + 1, i + 1, i, i);
    }
    if (file != NULL) {
        fclose(file);
    }
    CHECK(aotx_model_catalog_read(path, &catalog, reason, sizeof(reason)) == n,
          "the catalog batch %d does not read: %s", n, reason);
    for (i = 0; i < n && i < (int)catalog.count; i++) {
        CHECK(catalog.entry[i].bytes == (uint64_t)(i + 1),
              "batch entry %d has %llu bytes", i,
              (unsigned long long)catalog.entry[i].bytes);
        CHECK(strstr(catalog.entry[i].note, "entry") != NULL,
              "batch entry %d has no distinct note", i);
    }
    file = fopen(path, "a");
    if (file != NULL) {
        fputs("{\"name\":\"broken\"}\n", file);
        fclose(file);
    }
    CHECK(aotx_model_catalog_read(path, &catalog, reason, sizeof(reason)) < 0,
          "a malformed catalog line was accepted");
    aotx_remove_tree(dir);
    printf("catalog batch %d: fields %d\n", n, n * 13);
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        return 2;
    }
    real_catalog(argv[1]);
    batch(1);
    batch(64);
    return aotx_report("catalog_test", 240);
}
