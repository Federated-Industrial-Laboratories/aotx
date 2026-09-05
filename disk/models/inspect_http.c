/* Purpose: Read remote model headers with bounded HTTP ranges.
 * Owns: One connection handle and one bounded byte buffer.
 * Threading: One caller; requests run in order.
 * Lifetime: One header read. */
#include "disk/models/inspect.h"
#include <stdio.h>

#ifdef AOTX_FETCH
#include <curl/curl.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#define AOTX_INSPECT_CHUNK (1024u * 1024u)
#define AOTX_INSPECT_HEADERS (128u * 1024u)
#define AOTX_INSPECT_URL 8192u
#define AOTX_INSPECT_VALIDATOR 1024u

typedef struct inspect_http {
    CURL *curl;
    const char *source;
    const char *reason;
    unsigned char *buffer;
    uint64_t total, received, offset, first, last, length;
    size_t cached, used, wanted, headers;
    unsigned int status, range_seen, length_seen, ready;
    char location[AOTX_INSPECT_URL];
    char resource[AOTX_INSPECT_URL];
    char etag[AOTX_INSPECT_VALIDATOR], modified[AOTX_INSPECT_VALIDATOR];
    char validator[AOTX_INSPECT_VALIDATOR];
    int validator_etag;
    char auth[AOTX_INSPECT_VALIDATOR];
} inspect_http;

static size_t refuse(inspect_http *state, const char *reason)
{
    state->reason = reason;
    return 0u;
}

/* Read one decimal number without signed conversion or wraparound. */
static int number(const char **at, const char *end, uint64_t *value)
{
    const char *start = *at;
    *value = 0u;
    while (*at < end && **at >= '0' && **at <= '9') {
        unsigned int digit = (unsigned int)(**at - '0');
        if (*value > (UINT64_MAX - digit) / 10u) return 1;
        *value = *value * 10u + digit;
        ++*at;
    }
    return *at == start;
}

static int header_text(char *out, size_t capacity, const char *at, const char *end)
{
    size_t bytes = (size_t)(end - at);
    if (out[0] != '\0' || bytes == 0u || bytes >= capacity) return 1;
    for (const char *p = at; p < end; ++p)
        if ((unsigned char)*p < 32u || (unsigned char)*p == 127u) return 1;
    memcpy(out, at, bytes);
    out[bytes] = '\0';
    return 0;
}

static int strong_etag(const char *text)
{
    size_t bytes = strlen(text);
    if (bytes < 2u || text[0] != '"' || text[bytes - 1u] != '"') return 0;
    for (size_t i = 1u; i + 1u < bytes; ++i)
        if (text[i] == '"' || (unsigned char)text[i] < 33u) return 0;
    return 1;
}

static size_t take_header(char *data, size_t size, size_t count, void *opaque)
{
    inspect_http *state = opaque;
    if (size != 0u && count > SIZE_MAX / size) return refuse(state, "header size overflow");
    size_t bytes = size * count;
    if (bytes > AOTX_INSPECT_HEADERS - state->headers)
        return refuse(state, "HTTP headers exceed the byte bound");
    state->headers += bytes;
    if (bytes >= 5u && memcmp(data, "HTTP/", 5u) == 0) {
        const char *space = memchr(data, ' ', bytes);
        if (space == NULL || data + bytes - space < 5 || space[1] < '1' || space[1] > '5'
            || space[2] < '0' || space[2] > '9' || space[3] < '0' || space[3] > '9'
            || (space[4] != ' ' && space[4] != '\r' && space[4] != '\n'))
            return refuse(state, "invalid HTTP status line");
        if (state->ready) return refuse(state, "unexpected HTTP response after body headers");
        state->status = (unsigned int)(space[1] - '0') * 100u
                      + (unsigned int)(space[2] - '0') * 10u + (unsigned int)(space[3] - '0');
        state->range_seen = state->length_seen = 0u;
        state->etag[0] = state->modified[0] = state->location[0] = '\0';
        return bytes;
    }
    if ((bytes == 2u && memcmp(data, "\r\n", 2u) == 0) || (bytes == 1u && data[0] == '\n')) {
        if (state->status >= 100u && state->status < 200u && state->status != 101u) return bytes;
        /* Stop redirects and range refusals before any response body is read. */
        if ((state->status == 301u || state->status == 302u || state->status == 303u
             || state->status == 307u || state->status == 308u) && state->location[0] != '\0') {
            state->ready = 2u;
            return 0u;
        }
        if (state->status != 206u) return refuse(state, "HTTP response is not 206 Partial Content");
        if (!state->range_seen || state->first != state->offset
            || state->last < state->first || state->last - state->first != state->wanted - 1u
            || state->last >= state->length || (state->total && state->length != state->total))
            return refuse(state, "Content-Range does not match the requested bytes and source size");
        const char *validator = state->validator_etag ? state->etag : state->modified;
        if (state->validator[0] != '\0') {
            if (strcmp(validator, state->validator) != 0)
                return refuse(state, "the source validator changed or is absent");
        } else if (!strong_etag(state->etag)
                   && (state->modified[0] == '\0' || curl_getdate(state->modified, NULL) < 0)) {
            return refuse(state, "the source has no strong ETag or valid Last-Modified header");
        }
        state->ready = 1u;
        return bytes;
    }
    if (state->ready) return refuse(state, "HTTP trailers are not supported for a byte range");
    const char *colon = memchr(data, ':', bytes);
    if (colon == NULL) return refuse(state, "invalid HTTP header line");
    const char *at = colon + 1, *end = data + bytes;
    while (end > at && (end[-1] == '\r' || end[-1] == '\n' || end[-1] == ' ' || end[-1] == '\t')) --end;
    while (at < end && (*at == ' ' || *at == '\t')) ++at;
    size_t key = (size_t)(colon - data);
    if (key == 13u && strncasecmp(data, "Content-Range", key) == 0) {
        if (state->range_seen++ || end - at < 6 || memcmp(at, "bytes ", 6u) != 0)
            return refuse(state, "invalid or repeated Content-Range");
        at += 6;
        if (number(&at, end, &state->first) || at == end || *at++ != '-'
            || number(&at, end, &state->last) || at == end || *at++ != '/'
            || number(&at, end, &state->length) || at != end || state->length == 0u)
            return refuse(state, "invalid Content-Range numbers");
    } else if (key == 14u && strncasecmp(data, "Content-Length", key) == 0) {
        uint64_t length;
        if (state->length_seen++ || number(&at, end, &length) || at != end)
            return refuse(state, "invalid or repeated Content-Length");
        if (state->status == 206u && length != state->wanted)
            return refuse(state, "Content-Length differs from the requested range");
    } else if (key == 4u && strncasecmp(data, "ETag", key) == 0) {
        if (header_text(state->etag, sizeof state->etag, at, end)) return refuse(state, "invalid or repeated ETag");
    } else if (key == 13u && strncasecmp(data, "Last-Modified", key) == 0) {
        if (header_text(state->modified, sizeof state->modified, at, end)) return refuse(state, "invalid or repeated Last-Modified");
    } else if (key == 8u && strncasecmp(data, "Location", key) == 0) {
        if (header_text(state->location, sizeof state->location, at, end)) return refuse(state, "invalid or repeated redirect location");
    } else if (key == 16u && strncasecmp(data, "Content-Encoding", key) == 0) {
        if (end - at != 8 || strncasecmp(at, "identity", 8u) != 0)
            return refuse(state, "encoded HTTP byte ranges are not supported");
    }
    return bytes;
}

static size_t take_body(char *data, size_t size, size_t count, void *opaque)
{
    inspect_http *state = opaque;
    if (size != 0u && count > SIZE_MAX / size) return refuse(state, "body size overflow");
    size_t bytes = size * count;
    if (!state->ready || bytes > state->wanted - state->used)
        return refuse(state, "HTTP body exceeds the requested range");
    if (bytes > UINT64_MAX - state->received) return refuse(state, "received byte count overflow");
    memcpy(state->buffer + state->used, data, bytes);
    state->used += bytes;
    state->received += bytes;
    return bytes;
}

/* Resolve each redirect with libcurl. Credentials are never sent on a redirect. */
static int http_url(CURLU *url, const char *text)
{
    char *scheme = NULL, *user = NULL;
    if (strlen(text) >= AOTX_INSPECT_URL || curl_url_set(url, CURLUPART_URL, text, 0)) return 1;
    int bad = curl_url_get(url, CURLUPART_SCHEME, &scheme, 0) != CURLUE_OK;
    if (!bad) bad = strcmp(scheme, "http") != 0 && strcmp(scheme, "https") != 0;
    if (curl_url_get(url, CURLUPART_USER, &user, 0) == CURLUE_OK) bad = 1;
    curl_free(scheme);
    curl_free(user);
    return bad;
}

static int read_range(inspect_http *state, uint64_t offset, size_t bytes)
{
    CURLU *url = curl_url();
    struct curl_slist *headers = NULL;
    char range[64], condition[AOTX_INSPECT_VALIDATOR + 16u];
    int bad = 1;
    if (url == NULL) { state->reason = "the URL does not allocate"; return 1; }
    if (bytes == 0u || bytes > AOTX_INSPECT_CHUNK || bytes - 1u > UINT64_MAX - offset
        || http_url(url, state->resource[0] != '\0' ? state->resource : state->source)) {
        state->reason = "invalid HTTP URL or byte range";
        goto done;
    }
    snprintf(range, sizeof range, "%" PRIu64 "-%" PRIu64, offset, offset + bytes - 1u);
    state->offset = offset;
    state->wanted = bytes;
    state->headers = 0u;
    for (unsigned int redirect = 0u; redirect <= 10u; ++redirect) {
        char *address = NULL;
        state->ready = state->status = state->range_seen = state->length_seen = 0u;
        state->used = 0u;
        state->etag[0] = state->modified[0] = state->location[0] = '\0';
        state->reason = NULL;
        curl_slist_free_all(headers);
        headers = NULL;
        if (redirect == 0u && state->auth[0] != '\0' &&
            (state->resource[0] == '\0' || strcmp(state->resource, state->source) == 0)) {
            headers = curl_slist_append(NULL, state->auth);
            if (headers == NULL) { state->reason = "the authorization header does not allocate"; goto done; }
        }
        if (state->validator[0] != '\0') {
            snprintf(condition, sizeof condition, "If-Range: %s", state->validator);
            struct curl_slist *added = curl_slist_append(headers, condition);
            if (added == NULL) { state->reason = "the range header does not allocate"; goto done; }
            headers = added;
        }
        if (curl_url_get(url, CURLUPART_URL, &address, 0) != CURLUE_OK) {
            state->reason = "the HTTP URL does not read";
            goto done;
        }
        CURLcode configured = curl_easy_setopt(state->curl, CURLOPT_URL, address);
        curl_free(address);
        if (configured != CURLE_OK || curl_easy_setopt(state->curl, CURLOPT_RANGE, range) != CURLE_OK
            || curl_easy_setopt(state->curl, CURLOPT_HTTPHEADER, headers) != CURLE_OK) {
            state->reason = "the HTTP request options do not apply";
            goto done;
        }
        CURLcode code = curl_easy_perform(state->curl);
        if (state->ready == 2u && code == CURLE_WRITE_ERROR) {
            if (state->resource[0] != '\0') {
                state->reason = "the resolved HTTP resource changed";
                goto done;
            }
            if (redirect == 10u) { state->reason = "more than ten HTTP redirects"; goto done; }
            if (http_url(url, state->location)) { state->reason = "invalid HTTP redirect URL"; goto done; }
            continue;
        }
        if (code != CURLE_OK || !state->ready || state->used != bytes) {
            if (state->reason == NULL) state->reason = code != CURLE_OK
                ? curl_easy_strerror(code) : "HTTP body is shorter than the requested range";
            goto done;
        }
        if (state->validator[0] == '\0') {
            char *resolved = NULL;
            if (curl_url_get(url, CURLUPART_URL, &resolved, 0) != CURLUE_OK ||
                strlen(resolved) >= sizeof state->resource) {
                curl_free(resolved);
                state->reason = "the resolved HTTP URL does not fit";
                goto done;
            }
            memcpy(state->resource, resolved, strlen(resolved) + 1u);
            curl_free(resolved);
            state->validator_etag = strong_etag(state->etag);
            snprintf(state->validator, sizeof state->validator, "%s",
                     state->validator_etag ? state->etag : state->modified);
            state->total = state->length;
        }
        state->cached = bytes;
        bad = 0;
        break;
    }
 done:
    curl_easy_setopt(state->curl, CURLOPT_HTTPHEADER, NULL);
    curl_slist_free_all(headers);
    curl_url_cleanup(url);
    return bad;
}

static int read_bytes(void *opaque, uint64_t offset, size_t bytes, void *out)
{
    inspect_http *state = opaque;
    unsigned char *target = out;
    if (offset > state->total || bytes > state->total - offset) {
        state->reason = "header read exceeds the source size";
        return 1;
    }
    while (bytes != 0u) {
        if (offset < state->offset || offset - state->offset >= state->cached) {
            uint64_t available = state->total - offset;
            size_t chunk = available > AOTX_INSPECT_CHUNK ? AOTX_INSPECT_CHUNK : (size_t)available;
            if (read_range(state, offset, chunk)) return 1;
        }
        size_t start = (size_t)(offset - state->offset);
        size_t take = state->cached - start;
        if (take > bytes) take = bytes;
        memcpy(target, state->buffer + start, take);
        offset += take;
        target += take;
        bytes -= take;
    }
    return 0;
}

int aotx_model_inspect_remote(const char *source, aotx_modelfile **file, uint64_t *received)
{
    inspect_http state = {0};
    state.source = source;
    *file = NULL;
    *received = 0u;
    state.curl = curl_easy_init();
    state.buffer = malloc(AOTX_INSPECT_CHUNK);
    int result = 1;
    if (state.curl == NULL || state.buffer == NULL) {
        state.reason = "the HTTP read buffer or handle does not allocate";
        goto done;
    }
    /* HF_TOKEN belongs only to the HTTPS Hub origin, not an arbitrary source URL. */
    CURLU *url = curl_url();
    char *host = NULL, *scheme = NULL, *port = NULL;
    if (url == NULL || http_url(url, source)) {
        if (url != NULL) curl_url_cleanup(url);
        state.reason = "an HTTP or HTTPS URL without user information is required";
        goto done;
    }
    curl_url_get(url, CURLUPART_HOST, &host, 0);
    curl_url_get(url, CURLUPART_SCHEME, &scheme, 0);
    curl_url_get(url, CURLUPART_PORT, &port, CURLU_DEFAULT_PORT);
    const char *token = getenv("HF_TOKEN");
    if (host != NULL && scheme != NULL && port != NULL && strcmp(host, "huggingface.co") == 0
        && strcmp(scheme, "https") == 0 && strcmp(port, "443") == 0
        && token != NULL && token[0] != '\0') {
        int wrote = snprintf(state.auth, sizeof state.auth, "Authorization: Bearer %s", token);
        if (wrote < 0 || (size_t)wrote >= sizeof state.auth || strpbrk(token, "\r\n") != NULL)
            state.reason = "HF_TOKEN does not fit a request header";
    }
    curl_free(host);
    curl_free(scheme);
    curl_free(port);
    curl_url_cleanup(url);
    if (state.reason != NULL) goto done;
#define AOTX_HTTP_SET(option, value) \
    do { if (curl_easy_setopt(state.curl, option, value) != CURLE_OK) { \
        state.reason = "the HTTP options do not apply"; goto done; } } while (0)
    AOTX_HTTP_SET(CURLOPT_FOLLOWLOCATION, 0L);
    AOTX_HTTP_SET(CURLOPT_CONNECTTIMEOUT, 30L);
    AOTX_HTTP_SET(CURLOPT_TIMEOUT, 120L);
    AOTX_HTTP_SET(CURLOPT_NOSIGNAL, 1L);
    AOTX_HTTP_SET(CURLOPT_NOPROXY, "*");
    AOTX_HTTP_SET(CURLOPT_SSL_VERIFYPEER, 1L);
    AOTX_HTTP_SET(CURLOPT_SSL_VERIFYHOST, 2L);
    AOTX_HTTP_SET(CURLOPT_NETRC, (long)CURL_NETRC_IGNORED);
    AOTX_HTTP_SET(CURLOPT_USERAGENT, "aotx-models/" AOTX_VERSION);
    AOTX_HTTP_SET(CURLOPT_HEADERFUNCTION, take_header);
    AOTX_HTTP_SET(CURLOPT_HEADERDATA, &state);
    AOTX_HTTP_SET(CURLOPT_WRITEFUNCTION, take_body);
    AOTX_HTTP_SET(CURLOPT_WRITEDATA, &state);
#undef AOTX_HTTP_SET
    if (read_range(&state, 0u, 1u)) goto done;
    result = aotx_modelfile_open_reader(source, state.total, read_bytes, &state, file);
    if (result != 0 && state.reason != NULL) result = 1;
    if (result != 0 && state.reason == NULL) state.reason = "the model header does not read";
 done:
    *received = state.received;
    if (result != 0) fprintf(stderr, "inspect: %s: %s\n", source,
                             state.reason != NULL ? state.reason : "HTTP read failed");
    if (state.curl != NULL) curl_easy_cleanup(state.curl);
    free(state.buffer);
    return result;
}
#else
int aotx_model_inspect_remote(const char *source, aotx_modelfile **file, uint64_t *received)
{
    *file = NULL;
    *received = 0u;
    fprintf(stderr, "inspect: %s: HTTP support is disabled in this build\n", source);
    return 1;
}
#endif
