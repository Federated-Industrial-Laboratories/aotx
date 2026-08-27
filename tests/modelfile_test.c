/* Purpose: Read the model files that are present and stream one of them end to end.
 * Owns: The buffers of the reads and the counters of the run.
 * Threading: One thread.
 * Lifetime: The run of the program. */
#include "tests/disk_fake.h"

#include "disk/modelfile/manifest.h"

#include <dirent.h>
#include <fcntl.h>

#define AOTX_STREAM_BUFFER (64u * 1024u * 1024u)
#define AOTX_PIECE_BYTES   (1024u * 1024u)
#define AOTX_FILES_MAX     8
#define AOTX_HEX_BYTES     65

/* The pre-tokenizer name that every model of this set must carry. */
#define AOTX_PRE_NAME      "qwen2"

static char names[AOTX_FILES_MAX][256];
static uint64_t sizes[AOTX_FILES_MAX];
static int file_count;

/* Gives the name of a tensor type, or "other" for a type that this reader does not know. */
static const char *type_name(uint32_t type)
{
    switch (type) {
    case AOTX_TENSOR_F32:
        return "F32";
    case AOTX_TENSOR_F16:
        return "F16";
    case AOTX_TENSOR_Q4_0:
        return "Q4_0";
    case AOTX_TENSOR_Q8_0:
        return "Q8_0";
    default:
        return "other";
    }
}

/* Lists the model files of the directory, largest last. */
static void list_files(const char *dir)
{
    DIR *d = opendir(dir);
    struct dirent *entry;
    if (d == NULL) {
        return;
    }
    while ((entry = readdir(d)) != NULL && file_count < AOTX_FILES_MAX) {
        char path[512];
        struct stat st;
        const char *dot = strrchr(entry->d_name, '.');
        if (dot == NULL || strcmp(dot, ".gguf") != 0) {
            continue;
        }
        snprintf(path, sizeof(path), "%s/%s", dir, entry->d_name);
        if (stat(path, &st) != 0 || !S_ISREG(st.st_mode)) {
            continue;
        }
        snprintf(names[file_count], sizeof(names[0]), "%s", entry->d_name);
        sizes[file_count] = (uint64_t)st.st_size;
        file_count++;
    }
    closedir(d);
}

/* Reads a batch of tensors and compares each one with the bytes that the file holds at the
 * same place. Every element is a different tensor, so a wrong index cannot hide. */
static void batch(const aotx_modelfile *file, const char *path, int n)
{
    unsigned char *mine = (unsigned char *)malloc(AOTX_PIECE_BYTES);
    unsigned char *theirs = (unsigned char *)malloc(AOTX_PIECE_BYTES);
    uint64_t count = aotx_modelfile_tensor_count(file);
    uint64_t data_bytes = aotx_modelfile_data_bytes(file);
    struct stat st;
    uint64_t data_offset;
    int fd = open(path, O_RDONLY);
    int i;
    if (mine == NULL || theirs == NULL || fd < 0 || fstat(fd, &st) != 0) {
        CHECK(0, "the batch of %d cannot start", n);
        free(mine);
        free(theirs);
        if (fd >= 0) {
            close(fd);
        }
        return;
    }
    data_offset = (uint64_t)st.st_size - data_bytes;
    for (i = 0; i < n; i++) {
        aotx_tensor_info info;
        uint64_t want;
        ssize_t got;
        uint64_t index = (count > 0) ? ((uint64_t)i * 7u) % count : 0;
        if (count == 0) {
            break;
        }
        CHECK(aotx_modelfile_tensor(file, index, &info) == 0, "tensor %llu is absent",
              (unsigned long long)index);
        want = (info.bytes < AOTX_PIECE_BYTES) ? info.bytes : AOTX_PIECE_BYTES;
        if (want == 0) {
            continue;
        }
        CHECK(aotx_modelfile_read(file, info.offset, want, mine) == 0,
              "tensor %llu does not read", (unsigned long long)index);
        got = pread(fd, theirs, (size_t)want, (off_t)(data_offset + info.offset));
        CHECK(got == (ssize_t)want, "the file does not give the bytes of tensor %llu",
              (unsigned long long)index);
        CHECK(memcmp(mine, theirs, (size_t)want) == 0,
              "tensor %llu gives other bytes", (unsigned long long)index);
    }
    close(fd);
    free(mine);
    free(theirs);
}

/* Prints what one model file holds and checks the pre-tokenizer name. */
static void examine(const char *dir, const char *name)
{
    char path[512];
    aotx_modelfile *file = NULL;
    aotx_string_array tokens;
    aotx_string_array merges;
    aotx_tensor_info info;
    const char *architecture = "none";
    const char *model = "none";
    const char *pre = "none";
    uint64_t token_count = 0;
    uint64_t merge_count = 0;
    int rc;
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    rc = aotx_modelfile_open(path, &file);
    CHECK(rc == 0, "%s does not open, the code is %d", name, rc);
    if (rc != 0) {
        return;
    }
    (void)aotx_modelfile_string(file, "general.architecture", &architecture, NULL);
    (void)aotx_modelfile_string(file, "tokenizer.ggml.model", &model, NULL);
    (void)aotx_modelfile_string(file, "tokenizer.ggml.pre", &pre, NULL);
    if (aotx_modelfile_strings(file, "tokenizer.ggml.tokens", &tokens) == 0) {
        token_count = tokens.count;
    }
    if (aotx_modelfile_strings(file, "tokenizer.ggml.merges", &merges) == 0) {
        merge_count = merges.count;
    }
    CHECK(aotx_modelfile_tensor(file, 0, &info) == 0, "%s holds no tensor", name);
    printf("modelfile_test: %s\n", name);
    printf("  architecture %s, tokenizer %s, pre %s\n", architecture, model, pre);
    printf("  tokens %llu, merges %llu, tensors %llu, tensor bytes %llu\n",
           (unsigned long long)token_count, (unsigned long long)merge_count,
           (unsigned long long)aotx_modelfile_tensor_count(file),
           (unsigned long long)aotx_modelfile_data_bytes(file));
    printf("  first tensor %s, type %s, dims %llu %llu %llu %llu, bytes %llu\n", info.name,
           type_name(info.type), (unsigned long long)info.dims[0],
           (unsigned long long)info.dims[1], (unsigned long long)info.dims[2],
           (unsigned long long)info.dims[3], (unsigned long long)info.bytes);
    /* The design states that every model of this set carries the same pre-tokenizer. */
    CHECK(strcmp(pre, AOTX_PRE_NAME) == 0, "%s has the pre-tokenizer %s and not %s", name, pre,
          AOTX_PRE_NAME);
    CHECK(token_count > 0, "%s holds no token", name);
    CHECK(aotx_modelfile_tensor_count(file) > 0, "%s holds no tensor", name);
    batch(file, path, 1);
    batch(file, path, 64);
    aotx_modelfile_close(file);
}

/* Reads the largest file end to end through a 64 MB buffer. The digest of those bytes
 * must equal the digest of the bytes that the reader gives for the same file. */
static void stream(const char *dir, const char *name, uint64_t size)
{
    char path[512];
    char whole_text[AOTX_HEX_BYTES];
    char parts_text[AOTX_HEX_BYTES];
    unsigned char digest[AOTX_SHA256_DIGEST];
    aotx_sha256 state;
    aotx_modelfile *file = NULL;
    aotx_manifest_entry entries[AOTX_MANIFEST_MAX];
    void *buffer = malloc(AOTX_STREAM_BUFFER);
    uint64_t start;
    uint64_t end;
    uint64_t data_bytes;
    uint64_t data_offset;
    uint64_t at = 0;
    double seconds;
    int entry_count;
    int fd;
    int i;
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    if (buffer == NULL) {
        CHECK(0, "the stream buffer is not there");
        return;
    }
    fd = open(path, O_RDONLY);
    CHECK(fd >= 0, "%s does not open", name);
    if (fd < 0) {
        free(buffer);
        return;
    }
    /* The read alone gives the rate that a load of the weights can reach. The digest adds
     * its own cost, so the two rates are measured apart. */
    start = aotx_wall_ns();
    while (at < size) {
        uint64_t take = size - at;
        ssize_t got;
        if (take > AOTX_STREAM_BUFFER) {
            take = AOTX_STREAM_BUFFER;
        }
        got = pread(fd, buffer, (size_t)take, (off_t)at);
        if (got <= 0) {
            CHECK(0, "%s does not read at %llu", name, (unsigned long long)at);
            break;
        }
        at += (uint64_t)got;
    }
    end = aotx_wall_ns();
    seconds = (double)(end - start) / 1e9;
    printf("modelfile_test: read %s, %llu bytes, %.2f s, %.0f MB per second\n", name,
           (unsigned long long)size, seconds,
           (seconds > 0.0) ? ((double)size / seconds / 1e6) : 0.0);
    at = 0;
    aotx_sha256_init(&state);
    start = aotx_wall_ns();
    CHECK(aotx_sha256_read(fd, 0, size, buffer, AOTX_STREAM_BUFFER, &state) == 0,
          "%s does not stream", name);
    end = aotx_wall_ns();
    aotx_sha256_final(&state, digest);
    aotx_sha256_text(digest, whole_text);
    seconds = (double)(end - start) / 1e9;
    printf("modelfile_test: streamed %s, %llu bytes, %.2f s, %.0f MB per second\n", name,
           (unsigned long long)size, seconds,
           (seconds > 0.0) ? ((double)size / seconds / 1e6) : 0.0);
    /* The head of the file and the tensor bytes that the reader gives must make the same
     * digest. The offset of the tensor bytes is proven this way. */
    CHECK(aotx_modelfile_open(path, &file) == 0, "%s does not open for the read", name);
    if (file != NULL) {
        data_bytes = aotx_modelfile_data_bytes(file);
        data_offset = size - data_bytes;
        aotx_sha256_init(&state);
        CHECK(aotx_sha256_read(fd, 0, data_offset, buffer, AOTX_STREAM_BUFFER, &state) == 0,
              "the head of %s does not read", name);
        while (at < data_bytes) {
            uint64_t take = data_bytes - at;
            if (take > AOTX_STREAM_BUFFER) {
                take = AOTX_STREAM_BUFFER;
            }
            if (aotx_modelfile_read(file, at, take, buffer) != 0) {
                CHECK(0, "the tensor bytes of %s do not read at %llu", name,
                      (unsigned long long)at);
                break;
            }
            aotx_sha256_update(&state, buffer, (size_t)take);
            at += take;
        }
        aotx_sha256_final(&state, digest);
        aotx_sha256_text(digest, parts_text);
        CHECK(strcmp(whole_text, parts_text) == 0,
              "the pieces of %s give another digest", name);
        aotx_modelfile_close(file);
    }
    close(fd);
    free(buffer);
    /* The manifest holds the digest of every model file, when a manifest is there. */
    entry_count = aotx_manifest_read(dir, entries, AOTX_MANIFEST_MAX);
    if (entry_count <= 0) {
        printf("modelfile_test: skip, the manifest of %s is not there\n", dir);
        return;
    }
    for (i = 0; i < entry_count; i++) {
        if (strcmp(entries[i].path, name) == 0) {
            CHECK(strcmp(entries[i].sha256, whole_text) == 0,
                  "the manifest gives another digest for %s", name);
            CHECK(entries[i].bytes == size, "the manifest gives another size for %s", name);
            return;
        }
    }
    printf("modelfile_test: skip, the manifest holds no line for %s\n", name);
}

int main(int argc, char **argv)
{
    const char *dir = (argc > 1) ? argv[1] : "models";
    int largest = 0;
    int i;
    list_files(dir);
    /* A run with no case is not a clean run, so a directory with no model file fails
     * instead of reporting success over nothing. */
    if (file_count == 0) {
        printf("modelfile_test: no model file is in %s, so 0 cases applied\n", dir);
        return aotx_report("modelfile_test", 100);
    }
    printf("modelfile_test: model files %d\n", file_count);
    for (i = 0; i < file_count; i++) {
        examine(dir, names[i]);
        if (sizes[i] > sizes[largest]) {
            largest = i;
        }
    }
    stream(dir, names[largest], sizes[largest]);
    return aotx_report("modelfile_test", 100);
}
