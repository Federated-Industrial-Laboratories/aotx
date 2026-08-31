/* Purpose: Fetch one catalog file, resume its part, and verify all received bytes.
 * Owns: The part descriptor and one libcurl easy handle during the call.
 * Threading: One process for one fetch.
 * Lifetime: One fetch call. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/models/models.h"

#include "disk/wire/diskwire.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#ifdef AOTX_FETCH
#include <curl/curl.h>

#define AOTX_FETCH_BUFFER (1024u * 1024u)

typedef struct fetch_state {
    int fd;
    aotx_sha256 digest;
    uint64_t received;
    uint64_t resumed;
    uint64_t total;
    uint64_t started_ns;
    uint64_t report_ns;
    const char *name;
    int write_failed;
} fetch_state;

typedef struct upstream_state {
    char digest[AOTX_SHA256_HEX];
    uint64_t bytes;
    int has_digest;
    int has_bytes;
} upstream_state;

static uint64_t now_ns(void)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
}

static size_t take_bytes(char *data, size_t size, size_t count, void *opaque)
{
    fetch_state *state = (fetch_state *)opaque;
    size_t bytes = size * count;
    size_t at = 0u;
    while (at < bytes) {
        ssize_t wrote = write(state->fd, data + at, bytes - at);
        if (wrote < 0 && errno == EINTR) {
            continue;
        }
        if (wrote <= 0) {
            state->write_failed = 1;
            return 0u;
        }
        aotx_sha256_update(&state->digest, data + at, (size_t)wrote);
        state->received += (uint64_t)wrote;
        at += (size_t)wrote;
    }
    return bytes;
}

static int progress(void *opaque, curl_off_t total, curl_off_t now,
                    curl_off_t upload_total, curl_off_t upload_now)
{
    fetch_state *state = (fetch_state *)opaque;
    uint64_t tick = now_ns();
    uint64_t elapsed;
    uint64_t rate;
    (void)upload_total;
    (void)upload_now;
    (void)now;
    if (total > 0 && state->total == 0u) {
        state->total = state->received + (uint64_t)total;
    }
    if (tick < state->report_ns + 1000000000ull) {
        return 0;
    }
    elapsed = tick - state->started_ns;
    rate = (elapsed != 0u)
         ? ((state->received - state->resumed) * 1000000000ull) / elapsed : 0u;
    printf("bytes %llu total %llu rate %llu\n",
           (unsigned long long)state->received,
           (unsigned long long)state->total,
           (unsigned long long)rate);
    fflush(stdout);
    state->report_ns = tick;
    return 0;
}

static int join(char *out, size_t bytes, const char *dir, const char *name)
{
    int wrote = snprintf(out, bytes, "%s/%s", dir, name);
    return (wrote < 0 || (size_t)wrote >= bytes) ? -1 : 0;
}

static int owned_regular_fd(int fd, char *reason, size_t reason_bytes)
{
    struct stat info;
    if (fstat(fd, &info) != 0 || !S_ISREG(info.st_mode) || info.st_uid != geteuid()) {
        snprintf(reason, reason_bytes,
                 "the fetch file is not a regular file owned by this user");
        return -1;
    }
    return 0;
}

static int owned_regular_path(const char *path, int absent_ok,
                              char *reason, size_t reason_bytes)
{
    struct stat info;
    if (lstat(path, &info) != 0) {
        if (absent_ok && errno == ENOENT) {
            return 0;
        }
        snprintf(reason, reason_bytes, "the fetch file state does not read");
        return -1;
    }
    if (!S_ISREG(info.st_mode) || info.st_uid != geteuid()) {
        snprintf(reason, reason_bytes,
                 "the fetch file is not a regular file owned by this user");
        return -1;
    }
    return 0;
}

static int source_url(char *out, size_t bytes, const aotx_model_catalog_entry *entry)
{
    int wrote;
    if (strstr(entry->repository, "://") != NULL) {
        wrote = snprintf(out, bytes, "%s/%s/%s",
                         entry->repository, entry->revision, entry->file);
    } else {
        wrote = snprintf(out, bytes, "https://huggingface.co/%s/resolve/%s/%s",
                         entry->repository, entry->revision, entry->file);
    }
    return (wrote < 0 || (size_t)wrote >= bytes) ? -1 : 0;
}

static int host_of(const char *url, char *out, size_t bytes)
{
    static const char local[] = "file://";
    if (strncmp(url, local, sizeof(local) - 1u) == 0) {
        int wrote = snprintf(out, bytes, "local file");
        return (wrote < 0 || (size_t)wrote >= bytes) ? -1 : 0;
    }
    const char *at = strstr(url, "://");
    const char *end;
    size_t length;
    at = (at != NULL) ? at + 3 : url;
    end = strchr(at, '/');
    length = (end != NULL) ? (size_t)(end - at) : strlen(at);
    if (length == 0u || length + 1u > bytes) {
        return -1;
    }
    memcpy(out, at, length);
    out[length] = '\0';
    return 0;
}

static int existing_part(int fd, uint64_t bytes, aotx_sha256 *digest)
{
    void *buffer = malloc(AOTX_FETCH_BUFFER);
    int state;
    if (buffer == NULL) {
        return -1;
    }
    aotx_sha256_init(digest);
    state = aotx_sha256_read(fd, 0u, bytes, buffer, AOTX_FETCH_BUFFER, digest);
    free(buffer);
    return state == 0 ? 0 : -1;
}

static size_t take_header(char *data, size_t size, size_t count, void *opaque)
{
    upstream_state *state = (upstream_state *)opaque;
    size_t bytes = size * count;
    static const char digest_head[] = "x-linked-etag:";
    static const char size_head[] = "x-linked-size:";
    if (bytes > sizeof(digest_head) - 1u &&
        strncasecmp(data, digest_head, sizeof(digest_head) - 1u) == 0) {
        const char *at = data + sizeof(digest_head) - 1u;
        unsigned int i;
        while (at < data + bytes && (*at == ' ' || *at == '\t' || *at == '"')) {
            at++;
        }
        for (i = 0u; i < 64u && at + i < data + bytes; i++) {
            char c = at[i];
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
                break;
            }
            state->digest[i] = c;
        }
        if (i == 64u) {
            state->digest[64] = '\0';
            state->has_digest = 1;
        }
    } else if (bytes > sizeof(size_head) - 1u &&
               strncasecmp(data, size_head, sizeof(size_head) - 1u) == 0) {
        const char *at = data + sizeof(size_head) - 1u;
        uint64_t value = 0u;
        int digits = 0;
        while (at < data + bytes && (*at == ' ' || *at == '\t')) {
            at++;
        }
        while (at < data + bytes && *at >= '0' && *at <= '9') {
            value = value * 10u + (uint64_t)(*at - '0');
            digits = 1;
            at++;
        }
        if (digits) {
            state->bytes = value;
            state->has_bytes = 1;
        }
    }
    return bytes;
}

/* Compares a Hub file with the catalog before a part resumes. */
static int upstream_matches(const char *url, const char *token,
                            const aotx_model_catalog_entry *entry,
                            unsigned int connect_timeout,
                            char *reason, size_t reason_bytes)
{
    CURL *curl = curl_easy_init();
    struct curl_slist *headers = NULL;
    upstream_state state;
    char auth[1024];
    char error[CURL_ERROR_SIZE] = "";
    CURLcode code;
    if (curl == NULL) {
        snprintf(reason, reason_bytes, "libcurl did not make a metadata handle");
        return -1;
    }
    memset(&state, 0, sizeof(state));
    if (token != NULL && token[0] != '\0') {
        int wrote = snprintf(auth, sizeof(auth), "Authorization: Bearer %s", token);
        if (wrote < 0 || (size_t)wrote >= sizeof(auth) ||
            (headers = curl_slist_append(headers, auth)) == NULL) {
            curl_easy_cleanup(curl);
            snprintf(reason, reason_bytes, "the metadata request header does not fit");
            return -1;
        }
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    }
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 10L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, (long)connect_timeout);
    curl_easy_setopt(curl, CURLOPT_NOPROXY, "*");
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "aotx-models/" AOTX_VERSION);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, error);
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, take_header);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, &state);
    code = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    if (code != CURLE_OK) {
        snprintf(reason, reason_bytes, "the upstream file state does not read: %s",
                 error[0] != '\0' ? error : curl_easy_strerror(code));
        return -1;
    }
    if (!state.has_digest || !state.has_bytes) {
        snprintf(reason, reason_bytes, "the Hub did not give the linked digest and size");
        return -1;
    }
    return (state.bytes == entry->bytes && strcmp(state.digest, entry->sha256) == 0) ? 0 : 1;
}

static CURLcode transfer(const char *url, const char *part, const char *token,
                         uint64_t resume, unsigned int connect_timeout,
                         fetch_state *state, char *curl_error,
                         size_t curl_error_bytes)
{
    CURL *curl = curl_easy_init();
    struct curl_slist *headers = NULL;
    char auth[1024];
    CURLcode code;
    if (curl == NULL) {
        snprintf(curl_error, curl_error_bytes, "libcurl did not make a transfer handle");
        return CURLE_FAILED_INIT;
    }
    state->fd = open(part, O_WRONLY | O_CREAT | O_NOFOLLOW | O_CLOEXEC
                          | (resume == 0u ? O_TRUNC : O_APPEND), 0600);
    if (state->fd < 0) {
        curl_easy_cleanup(curl);
        snprintf(curl_error, curl_error_bytes, "the part file does not open");
        return CURLE_WRITE_ERROR;
    }
    if (owned_regular_fd(state->fd, curl_error, curl_error_bytes) != 0) {
        close(state->fd);
        state->fd = -1;
        curl_easy_cleanup(curl);
        return CURLE_WRITE_ERROR;
    }
    if (token != NULL && token[0] != '\0') {
        int wrote = snprintf(auth, sizeof(auth), "Authorization: Bearer %s", token);
        if (wrote < 0 || (size_t)wrote >= sizeof(auth)) {
            close(state->fd);
            curl_easy_cleanup(curl);
            snprintf(curl_error, curl_error_bytes, "HF_TOKEN is too long");
            return CURLE_BAD_FUNCTION_ARGUMENT;
        }
        headers = curl_slist_append(headers, auth);
        if (headers == NULL) {
            close(state->fd);
            curl_easy_cleanup(curl);
            snprintf(curl_error, curl_error_bytes, "the request header does not allocate");
            return CURLE_OUT_OF_MEMORY;
        }
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    }
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 10L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, (long)connect_timeout);
    curl_easy_setopt(curl, CURLOPT_NOPROXY, "*");
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "aotx-models/" AOTX_VERSION);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, take_bytes);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, state);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, state);
    if (resume != 0u) {
        curl_easy_setopt(curl, CURLOPT_RESUME_FROM_LARGE, (curl_off_t)resume);
    }
    code = curl_easy_perform(curl);
    if (code != CURLE_OK && curl_error[0] == '\0') {
        snprintf(curl_error, curl_error_bytes, "%s", curl_easy_strerror(code));
    }
    if (fsync(state->fd) != 0 && code == CURLE_OK) {
        code = CURLE_WRITE_ERROR;
        snprintf(curl_error, curl_error_bytes, "the part file does not synchronize");
    }
    close(state->fd);
    state->fd = -1;
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return code;
}

int aotx_model_fetch(const char *dir, const aotx_model_catalog_entry *entry,
                     unsigned int connect_timeout,
                     char *reason, size_t reason_bytes)
{
    char path[AOTX_MODEL_PATH];
    char part[AOTX_MODEL_PATH];
    char url[AOTX_MODEL_TEXT * 2u];
    char host[256];
    char curl_error[256] = "";
    char got[AOTX_SHA256_HEX];
    unsigned char digest[AOTX_SHA256_DIGEST];
    struct stat info;
    fetch_state state;
    aotx_model_store_record record;
    const char *token = getenv("HF_TOKEN");
    uint64_t resume = 0u;
    CURLcode code;
    int fd;
    if (entry == NULL || connect_timeout == 0u
        || join(path, sizeof(path), dir, entry->file) != 0 ||
        snprintf(part, sizeof(part), "%s.part", path) >= (int)sizeof(part) ||
        source_url(url, sizeof(url), entry) != 0 || host_of(url, host, sizeof(host)) != 0) {
        snprintf(reason, reason_bytes, "the fetch path or source URL does not fit");
        return -1;
    }
    if (mkdir(dir, 0700) != 0 && errno != EEXIST) {
        snprintf(reason, reason_bytes, "the models directory does not open");
        return -1;
    }
    memset(&state, 0, sizeof(state));
    state.fd = -1;
    state.name = entry->name;
    state.total = entry->bytes;
    if (owned_regular_path(part, 1, reason, reason_bytes) != 0
        || owned_regular_path(path, 1, reason, reason_bytes) != 0) {
        return -1;
    }
    if (lstat(part, &info) == 0 && info.st_size > 0) {
        resume = (uint64_t)info.st_size;
        if (resume >= entry->bytes) {
            unlink(part);
            resume = 0u;
            printf("restart the part because its size is not below the catalog size\n");
        }
    }
    printf("host %s\n", host);
    fflush(stdout);
    if (resume != 0u && strncmp(url, "https://huggingface.co/",
                               sizeof("https://huggingface.co/") - 1u) == 0) {
        int upstream = upstream_matches(url, token, entry, connect_timeout,
                                        reason, reason_bytes);
        if (upstream < 0) {
            return -1;
        }
        if (upstream > 0) {
            unlink(part);
            resume = 0u;
            printf("restart the part because the upstream digest or size changed\n");
            fflush(stdout);
        }
    }
    fd = open(part, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd >= 0 && owned_regular_fd(fd, reason, reason_bytes) != 0) {
        close(fd);
        return -1;
    }
    if (resume != 0u && (fd < 0 || existing_part(fd, resume, &state.digest) != 0)) {
        if (fd >= 0) {
            close(fd);
        }
        snprintf(reason, reason_bytes, "the existing part does not read for resume");
        return -1;
    }
    if (fd >= 0) {
        close(fd);
    }
    if (resume == 0u) {
        aotx_sha256_init(&state.digest);
    }
    state.received = resume;
    state.resumed = resume;
    state.started_ns = now_ns();
    state.report_ns = state.started_ns - 1000000000ull;
    code = transfer(url, part, token, resume, connect_timeout,
                    &state, curl_error, sizeof(curl_error));
    if (code == CURLE_RANGE_ERROR && resume != 0u) {
        printf("restart the part because the host did not take the byte range\n");
        fflush(stdout);
        unlink(part);
        memset(&state, 0, sizeof(state));
        state.fd = -1;
        state.name = entry->name;
        state.total = entry->bytes;
        state.started_ns = now_ns();
        state.report_ns = state.started_ns - 1000000000ull;
        aotx_sha256_init(&state.digest);
        curl_error[0] = '\0';
        code = transfer(url, part, token, 0u, connect_timeout,
                        &state, curl_error, sizeof(curl_error));
    }
    if (code != CURLE_OK) {
        snprintf(reason, reason_bytes, "fetch failed: %s", curl_error);
        return -1;
    }
    aotx_sha256_final(&state.digest, digest);
    aotx_sha256_text(digest, got);
    printf("bytes %llu total %llu rate 0\n", (unsigned long long)state.received,
           (unsigned long long)entry->bytes);
    fflush(stdout);
    if (state.received != entry->bytes || strcmp(got, entry->sha256) != 0) {
        unlink(part);
        snprintf(reason, reason_bytes, "digest mismatch: expected %s, received %s",
                 entry->sha256, got);
        return -1;
    }
    if (owned_regular_path(part, 0, reason, reason_bytes) != 0
        || owned_regular_path(path, 1, reason, reason_bytes) != 0
        || rename(part, path) != 0) {
        snprintf(reason, reason_bytes, "the verified part does not take its final name");
        return -1;
    }
    memset(&record, 0, sizeof(record));
    snprintf(record.name, sizeof(record.name), "%s", entry->name);
    snprintf(record.file, sizeof(record.file), "%s", entry->file);
    record.bytes = entry->bytes;
    snprintf(record.sha256, sizeof(record.sha256), "%s", got);
    snprintf(record.source, sizeof(record.source), "%s", url);
    {
        time_t stamp = time(NULL);
        struct tm clock;
        gmtime_r(&stamp, &clock);
        strftime(record.date, sizeof(record.date), "%Y-%m-%dT%H:%M:%SZ", &clock);
    }
    snprintf(record.revision, sizeof(record.revision), "%s", entry->revision);
    record.verified = 1;
    if (aotx_model_store_append(dir, &record, reason, reason_bytes) != 0) {
        return -1;
    }
    return 0;
}

#else

int aotx_model_fetch(const char *dir, const aotx_model_catalog_entry *entry,
                     unsigned int connect_timeout,
                     char *reason, size_t reason_bytes)
{
    (void)dir;
    (void)entry;
    (void)connect_timeout;
    snprintf(reason, reason_bytes,
             "fetch is not in this build because libcurl was not selected");
    return -1;
}

#endif
