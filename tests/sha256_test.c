/* Purpose: Check the digest against the published values and against the system tool.
 * Owns: The buffers of the cases and the counters of the run.
 * Threading: One thread.
 * Lifetime: The run of the program. */
#include "tests/disk_fake.h"

#include <dirent.h>

#define AOTX_HEX_BYTES 65

/* The published values of FIPS 180-4 and of the byte string of one million letters. */
static const char *empty_value =
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
static const char *abc_value =
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
static const char *block_448 =
    "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
static const char *block_448_value =
    "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1";
static const char *block_896 =
    "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno"
    "ijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu";
static const char *block_896_value =
    "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1";
static const char *million_value =
    "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0";

static void digest_of(const void *data, size_t bytes, char *text)
{
    aotx_sha256 state;
    unsigned char digest[AOTX_SHA256_DIGEST];
    aotx_sha256_init(&state);
    aotx_sha256_update(&state, data, bytes);
    aotx_sha256_final(&state, digest);
    aotx_sha256_text(digest, text);
}

static void fill(unsigned char *buffer, size_t bytes, unsigned seed)
{
    size_t i;
    /* Every element carries different content, so a wrong index cannot hide. */
    for (i = 0; i < bytes; i++) {
        seed = seed * 1103515245u + 12345u;
        buffer[i] = (unsigned char)(seed >> 16);
    }
}

/* The published values, which the standard states. */
static void published(void)
{
    char text[AOTX_HEX_BYTES];
    aotx_sha256 state;
    unsigned char digest[AOTX_SHA256_DIGEST];
    int i;
    digest_of("", 0, text);
    CHECK(strcmp(text, empty_value) == 0, "the empty string gives %s", text);
    digest_of("abc", 3, text);
    CHECK(strcmp(text, abc_value) == 0, "abc gives %s", text);
    digest_of(block_448, strlen(block_448), text);
    CHECK(strcmp(text, block_448_value) == 0, "the 448-bit string gives %s", text);
    digest_of(block_896, strlen(block_896), text);
    CHECK(strcmp(text, block_896_value) == 0, "the 896-bit string gives %s", text);
    aotx_sha256_init(&state);
    for (i = 0; i < 1000000; i++) {
        aotx_sha256_update(&state, "a", 1);
    }
    aotx_sha256_final(&state, digest);
    aotx_sha256_text(digest, text);
    CHECK(strcmp(text, million_value) == 0, "one million letters give %s", text);
}

/* A run of small calls must give the digest that one large call gives. */
static void batch(int n)
{
    unsigned char buffer[8192];
    char whole[AOTX_HEX_BYTES];
    char parts[AOTX_HEX_BYTES];
    char first[AOTX_HEX_BYTES];
    int i;
    for (i = 0; i < n; i++) {
        size_t bytes = (size_t)(37 + i * 113) % 8192u;
        size_t step = (size_t)(1 + i * 7) % 300u + 1u;
        size_t at = 0;
        aotx_sha256 state;
        unsigned char digest[AOTX_SHA256_DIGEST];
        fill(buffer, bytes, (unsigned)(i + 1));
        digest_of(buffer, bytes, whole);
        aotx_sha256_init(&state);
        while (at < bytes) {
            size_t take = (bytes - at < step) ? (bytes - at) : step;
            aotx_sha256_update(&state, buffer + at, take);
            at += take;
        }
        aotx_sha256_final(&state, digest);
        aotx_sha256_text(digest, parts);
        CHECK(strcmp(whole, parts) == 0, "the pieces of element %d give another digest", i);
        if (i == 0) {
            memcpy(first, whole, sizeof(first));
        } else {
            CHECK(strcmp(first, whole) != 0, "element %d gives the digest of element 0", i);
        }
    }
}

/* Gives the digest that the system tool computes, or an empty text when the tool is not
 * there. The tool is the outside reference for the value of a large file. */
static void tool_digest(const char *path, char *text)
{
    char command[2048];
    FILE *pipe;
    text[0] = '\0';
    snprintf(command, sizeof(command), "sha256sum '%s' 2>/dev/null", path);
    pipe = popen(command, "r");
    if (pipe == NULL) {
        return;
    }
    if (fscanf(pipe, "%64s", text) != 1) {
        text[0] = '\0';
    }
    pclose(pipe);
}

/* Hashes every model file in the directory and compares each value with the system tool. */
static int real_files(const char *dir)
{
    DIR *d = opendir(dir);
    struct dirent *entry;
    void *buffer = malloc(4u * 1024u * 1024u);
    int skipped = 0;
    if (buffer == NULL) {
        return 1;
    }
    if (d == NULL) {
        printf("sha256_test: skip, the model directory %s is not there\n", dir);
        free(buffer);
        return 1;
    }
    while ((entry = readdir(d)) != NULL) {
        char path[512];
        char mine[AOTX_HEX_BYTES];
        char theirs[AOTX_HEX_BYTES];
        uint64_t bytes = 0;
        const char *dot = strrchr(entry->d_name, '.');
        if (dot == NULL || strcmp(dot, ".gguf") != 0) {
            continue;
        }
        snprintf(path, sizeof(path), "%s/%s", dir, entry->d_name);
        tool_digest(path, theirs);
        if (theirs[0] == '\0') {
            printf("sha256_test: skip, the system tool gives no value for %s\n", entry->d_name);
            skipped++;
            continue;
        }
        CHECK(aotx_sha256_file(path, mine, &bytes, buffer, 4u * 1024u * 1024u) == 0,
              "%s does not hash", entry->d_name);
        CHECK(strcmp(mine, theirs) == 0, "%s gives %s and the tool gives %s", entry->d_name,
              mine, theirs);
        printf("sha256_test: %s bytes %llu sha256 %s\n", entry->d_name,
               (unsigned long long)bytes, mine);
    }
    closedir(d);
    free(buffer);
    return skipped;
}

int main(int argc, char **argv)
{
    const char *dir = (argc > 1) ? argv[1] : "models";
    int skipped;
    published();
    batch(1);
    batch(64);
    skipped = real_files(dir);
    printf("sha256_test: skipped %d\n", skipped);
    return aotx_report("sha256_test", 130);
}
