/* Purpose: Check whole, resumed, restarted, rejected, and redirected model fetches.
 * Owns: Loopback servers and one temporary store for each case.
 * Threading: One server process beside one fetch process.
 * Lifetime: The test. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/models/models.h"
#include "disk/wire/diskwire.h"
#include "tests/disk_fake.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <time.h>

#ifdef AOTX_FETCH_TEST

#define DATA_BYTES (1024u * 1024u)

enum server_mode {
    SERVER_WHOLE,
    SERVER_RANGE,
    SERVER_IGNORE_RANGE,
    SERVER_SLOW
};

static unsigned char data[DATA_BYTES];

static int write_all(int fd, const void *bytes, size_t count)
{
    const unsigned char *at = (const unsigned char *)bytes;
    while (count != 0u) {
        ssize_t wrote = write(fd, at, count);
        if (wrote < 0 && errno == EINTR) {
            continue;
        }
        if (wrote <= 0) {
            return -1;
        }
        at += wrote;
        count -= (size_t)wrote;
    }
    return 0;
}

static int open_server(unsigned short *port)
{
    struct sockaddr_in address;
    socklen_t bytes = sizeof(address);
    int one = 1;
    int fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        return -1;
    }
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    if (bind(fd, (struct sockaddr *)&address, sizeof(address)) != 0 ||
        listen(fd, 4) != 0 ||
        getsockname(fd, (struct sockaddr *)&address, &bytes) != 0) {
        close(fd);
        return -1;
    }
    *port = ntohs(address.sin_port);
    return fd;
}

static int read_request(int fd, char *out, size_t bytes)
{
    size_t fill = 0u;
    while (fill + 1u < bytes) {
        ssize_t got = read(fd, out + fill, bytes - fill - 1u);
        if (got <= 0) {
            return -1;
        }
        fill += (size_t)got;
        out[fill] = '\0';
        if (strstr(out, "\r\n\r\n") != NULL) {
            return 0;
        }
    }
    return -1;
}

static uint64_t range_start(const char *request)
{
    const char *at = strstr(request, "Range: bytes=");
    uint64_t value = 0u;
    if (at == NULL) {
        return 0u;
    }
    at += strlen("Range: bytes=");
    while (*at >= '0' && *at <= '9') {
        value = value * 10u + (uint64_t)(*at - '0');
        at++;
    }
    return value;
}

static int answer(int fd, enum server_mode mode, const unsigned char *body, size_t bytes,
                  int expect_range)
{
    char request[8192] = "";
    char head[256];
    uint64_t start;
    size_t at;
    if (read_request(fd, request, sizeof(request)) != 0) {
        return 1;
    }
    if (strstr(request, "User-Agent: aotx-models/" AOTX_VERSION) == NULL) {
        return 4;
    }
    start = range_start(request);
    if ((expect_range && strstr(request, "Range: bytes=") == NULL) ||
        (!expect_range && mode != SERVER_IGNORE_RANGE &&
         strstr(request, "Range: bytes=") != NULL)) {
        return 2;
    }
    if (mode == SERVER_RANGE) {
        int wrote = snprintf(head, sizeof(head),
                             "HTTP/1.1 206 Partial Content\r\nContent-Length: %zu\r\n"
                             "Content-Range: bytes %llu-%zu/%zu\r\nConnection: close\r\n\r\n",
                             bytes - (size_t)start, (unsigned long long)start,
                             bytes - 1u, bytes);
        if (write_all(fd, head, (size_t)wrote) != 0 ||
            write_all(fd, body + start, bytes - (size_t)start) != 0) {
            return 3;
        }
        return 0;
    }
    {
        int wrote = snprintf(head, sizeof(head),
                             "HTTP/1.1 200 OK\r\nContent-Length: %zu\r\n"
                             "Connection: close\r\n\r\n", bytes);
        if (write_all(fd, head, (size_t)wrote) != 0) {
            return 3;
        }
    }
    if (mode != SERVER_SLOW) {
        int state = write_all(fd, body, bytes);
        return (state == 0 || mode == SERVER_IGNORE_RANGE) ? 0 : 3;
    }
    for (at = 0u; at < bytes; at += 4096u) {
        size_t take = bytes - at;
        if (take > 4096u) {
            take = 4096u;
        }
        if (write_all(fd, body + at, take) != 0) {
            return 0;
        }
        usleep(2000u);
    }
    return 0;
}

static int serve(unsigned short *port, enum server_mode mode, const unsigned char *body,
                 size_t bytes)
{
    int listen_fd = open_server(port);
    int pid;
    if (listen_fd < 0) {
        return -1;
    }
    pid = fork();
    if (pid == 0) {
        int client;
        int state = 0;
        signal(SIGPIPE, SIG_IGN);
        client = accept4(listen_fd, NULL, NULL, SOCK_CLOEXEC);
        if (client < 0) {
            _exit(10);
        }
        state = answer(client, mode, body, bytes,
                       mode == SERVER_RANGE || mode == SERVER_IGNORE_RANGE);
        close(client);
        if (mode == SERVER_IGNORE_RANGE) {
            client = accept4(listen_fd, NULL, NULL, SOCK_CLOEXEC);
            if (client < 0) {
                _exit(11);
            }
            state |= answer(client, SERVER_WHOLE, body, bytes, 0);
            close(client);
        }
        close(listen_fd);
        _exit(state);
    }
    close(listen_fd);
    return pid;
}

static void digest_of(const void *bytes, size_t count, char out[AOTX_SHA256_HEX])
{
    aotx_sha256 state;
    unsigned char raw[AOTX_SHA256_DIGEST];
    aotx_sha256_init(&state);
    aotx_sha256_update(&state, bytes, count);
    aotx_sha256_final(&state, raw);
    aotx_sha256_text(raw, out);
}

static void entry_for(aotx_model_catalog_entry *entry, unsigned short port,
                      const char *name, const char *file, size_t bytes)
{
    memset(entry, 0, sizeof(*entry));
    snprintf(entry->name, sizeof(entry->name), "%s", name);
    snprintf(entry->role, sizeof(entry->role), "language");
    snprintf(entry->repository, sizeof(entry->repository), "http://127.0.0.1:%u", port);
    snprintf(entry->file, sizeof(entry->file), "%s", file);
    snprintf(entry->revision, sizeof(entry->revision), "revision");
    entry->bytes = bytes;
    digest_of(data, bytes, entry->sha256);
    snprintf(entry->license, sizeof(entry->license), "Apache-2.0");
    snprintf(entry->quant, sizeof(entry->quant), "Q8_0");
    snprintf(entry->profiles, sizeof(entry->profiles), "8g");
    snprintf(entry->source, sizeof(entry->source), "loopback");
}

static void check_file(const char *dir, const aotx_model_catalog_entry *entry)
{
    char path[AOTX_MODEL_PATH];
    char got[AOTX_SHA256_HEX];
    unsigned char buffer[4096];
    uint64_t bytes = 0u;
    snprintf(path, sizeof(path), "%s/%s", dir, entry->file);
    CHECK(aotx_sha256_file(path, got, &bytes, buffer, sizeof(buffer)) == 0,
          "the fetched file does not hash");
    CHECK(bytes == entry->bytes && strcmp(got, entry->sha256) == 0,
          "the fetched file has %llu bytes and digest %s", (unsigned long long)bytes, got);
}

static void whole_case(int rows)
{
    aotx_model_catalog_entry entry;
    char dir[128];
    char reason[512];
    unsigned short port;
    size_t bytes = (size_t)rows * 4096u;
    int server;
    size_t i;
    for (i = 0u; i < bytes; i++) {
        data[i] = (unsigned char)((i + (size_t)rows) % 251u);
    }
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the whole store does not open");
    server = serve(&port, SERVER_WHOLE, data, bytes);
    entry_for(&entry, port, "whole", "whole.gguf", bytes);
    CHECK(aotx_model_fetch(dir, &entry, AOTX_MODEL_FETCH_TIMEOUT,
                           reason, sizeof(reason)) == 0,
          "the whole fetch failed: %s", reason);
    CHECK(aotx_wait(server) == 0, "the whole server failed");
    check_file(dir, &entry);
    aotx_remove_tree(dir);
    printf("whole fetch %d: %zu bytes\n", rows, bytes);
}

static void resume_case(void)
{
    aotx_model_catalog_entry entry;
    char dir[128];
    char part[AOTX_MODEL_PATH];
    char reason[512];
    unsigned short port;
    int server;
    size_t half = DATA_BYTES / 2u;
    size_t i;
    for (i = 0u; i < DATA_BYTES; i++) {
        data[i] = (unsigned char)((i * 7u) % 251u);
    }
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the resume store does not open");
    server = serve(&port, SERVER_RANGE, data, DATA_BYTES);
    entry_for(&entry, port, "resume", "resume.gguf", DATA_BYTES);
    snprintf(part, sizeof(part), "%s/%s.part", dir, entry.file);
    {
        int fd = open(part, O_WRONLY | O_CREAT | O_TRUNC, 0600);
        CHECK(fd >= 0 && write_all(fd, data, half) == 0, "the resume part does not write");
        if (fd >= 0) close(fd);
    }
    CHECK(aotx_model_fetch(dir, &entry, AOTX_MODEL_FETCH_TIMEOUT,
                           reason, sizeof(reason)) == 0,
          "the resumed fetch failed: %s", reason);
    CHECK(aotx_wait(server) == 0, "the range server failed");
    check_file(dir, &entry);
    aotx_remove_tree(dir);
}

static void ignored_range_case(void)
{
    aotx_model_catalog_entry entry;
    char dir[128];
    char part[AOTX_MODEL_PATH];
    char reason[512];
    unsigned short port;
    int server;
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the restart store does not open");
    server = serve(&port, SERVER_IGNORE_RANGE, data, DATA_BYTES);
    entry_for(&entry, port, "restart", "restart.gguf", DATA_BYTES);
    snprintf(part, sizeof(part), "%s/%s.part", dir, entry.file);
    {
        int fd = open(part, O_WRONLY | O_CREAT | O_TRUNC, 0600);
        CHECK(fd >= 0 && write_all(fd, data, 4096u) == 0, "the restart part does not write");
        if (fd >= 0) close(fd);
    }
    CHECK(aotx_model_fetch(dir, &entry, AOTX_MODEL_FETCH_TIMEOUT,
                           reason, sizeof(reason)) == 0,
          "the restarted fetch failed: %s", reason);
    CHECK(aotx_wait(server) == 0, "the ignored-range server failed");
    check_file(dir, &entry);
    aotx_remove_tree(dir);
}

static void wrong_digest_case(void)
{
    aotx_model_catalog_entry entry;
    char dir[128];
    char part[AOTX_MODEL_PATH];
    char reason[512];
    unsigned short port;
    int server;
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the mismatch store does not open");
    server = serve(&port, SERVER_WHOLE, data, DATA_BYTES);
    entry_for(&entry, port, "mismatch", "mismatch.gguf", DATA_BYTES);
    entry.sha256[0] = (entry.sha256[0] == '0') ? '1' : '0';
    CHECK(aotx_model_fetch(dir, &entry, AOTX_MODEL_FETCH_TIMEOUT,
                           reason, sizeof(reason)) != 0,
          "the wrong expected digest was accepted");
    CHECK(strstr(reason, "expected") != NULL && strstr(reason, "received") != NULL,
          "the mismatch reason does not give both digests: %s", reason);
    CHECK(aotx_wait(server) == 0, "the mismatch server failed");
    snprintf(part, sizeof(part), "%s/%s.part", dir, entry.file);
    CHECK(access(part, F_OK) != 0, "the digest mismatch left its part file");
    aotx_remove_tree(dir);
}

static void wrong_part_case(void)
{
    aotx_model_catalog_entry entry;
    char dir[128];
    char part[AOTX_MODEL_PATH];
    char reason[512];
    unsigned short port;
    size_t half = DATA_BYTES / 2u;
    int server;
    int fd;
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the wrong-part store does not open");
    server = serve(&port, SERVER_RANGE, data, DATA_BYTES);
    entry_for(&entry, port, "wrong-part", "wrong-part.gguf", DATA_BYTES);
    snprintf(part, sizeof(part), "%s/%s.part", dir, entry.file);
    fd = open(part, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    CHECK(fd >= 0 && write_all(fd, data, half) == 0, "the wrong part does not write");
    if (fd >= 0) {
        CHECK(pwrite(fd, "x", 1u, 17) == 1, "the wrong part does not change");
        close(fd);
    }
    CHECK(aotx_model_fetch(dir, &entry, AOTX_MODEL_FETCH_TIMEOUT,
                           reason, sizeof(reason)) != 0,
          "a wrong resumed prefix was accepted");
    CHECK(strstr(reason, "expected") != NULL && strstr(reason, "received") != NULL,
          "the wrong-prefix reason does not give both digests: %s", reason);
    CHECK(aotx_wait(server) == 0, "the wrong-prefix server failed");
    CHECK(access(part, F_OK) != 0, "the wrong resumed prefix was not deleted");
    aotx_remove_tree(dir);
}

static void killed_case(void)
{
    aotx_model_catalog_entry entry;
    char dir[128];
    char part[AOTX_MODEL_PATH];
    char reason[512];
    unsigned short port;
    int server;
    int fetcher;
    struct stat info;
    int tries;
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the killed store does not open");
    server = serve(&port, SERVER_SLOW, data, DATA_BYTES);
    entry_for(&entry, port, "killed", "killed.gguf", DATA_BYTES);
    snprintf(part, sizeof(part), "%s/%s.part", dir, entry.file);
    fetcher = fork();
    if (fetcher == 0) {
        _exit(aotx_model_fetch(dir, &entry, AOTX_MODEL_FETCH_TIMEOUT,
                               reason, sizeof(reason)) == 0 ? 0 : 1);
    }
    for (tries = 0; tries < 400; tries++) {
        if (stat(part, &info) == 0 && info.st_size >= (off_t)(DATA_BYTES / 2u)) {
            break;
        }
        usleep(5000u);
    }
    CHECK(tries < 400, "the killed fetch did not reach half");
    kill(fetcher, SIGKILL);
    waitpid(fetcher, NULL, 0);
    waitpid(server, NULL, 0);
    CHECK(stat(part, &info) == 0 && info.st_size > 0 && info.st_size < DATA_BYTES,
          "the killed fetch left no partial file");
    server = serve(&port, SERVER_RANGE, data, DATA_BYTES);
    snprintf(entry.repository, sizeof(entry.repository), "http://127.0.0.1:%u", port);
    CHECK(aotx_model_fetch(dir, &entry, AOTX_MODEL_FETCH_TIMEOUT,
                           reason, sizeof(reason)) == 0,
          "the killed fetch did not resume: %s", reason);
    CHECK(aotx_wait(server) == 0, "the killed-fetch range server failed");
    check_file(dir, &entry);
    aotx_remove_tree(dir);
}

static int redirect_server(unsigned short *first_port, unsigned short second_port)
{
    int listen_fd = open_server(first_port);
    int pid = fork();
    if (pid == 0) {
        char request[8192] = "";
        char answer_text[512];
        int client = accept4(listen_fd, NULL, NULL, SOCK_CLOEXEC);
        int wrote;
        if (client < 0 || read_request(client, request, sizeof(request)) != 0 ||
            strstr(request, "Authorization: Bearer fetch-test-token") == NULL ||
            strstr(request, "User-Agent: aotx-models/" AOTX_VERSION) == NULL) {
            _exit(1);
        }
        wrote = snprintf(answer_text, sizeof(answer_text),
                         "HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:%u/revision/token.gguf\r\n"
                         "Content-Length: 0\r\nConnection: close\r\n\r\n", second_port);
        write_all(client, answer_text, (size_t)wrote);
        close(client);
        close(listen_fd);
        _exit(0);
    }
    close(listen_fd);
    return pid;
}

static int content_server(unsigned short *port)
{
    int listen_fd = open_server(port);
    int pid = fork();
    if (pid == 0) {
        char request[8192] = "";
        char head[256];
        int client = accept4(listen_fd, NULL, NULL, SOCK_CLOEXEC);
        int wrote;
        if (client < 0 || read_request(client, request, sizeof(request)) != 0 ||
            strstr(request, "Authorization:") != NULL ||
            strstr(request, "User-Agent: aotx-models/" AOTX_VERSION) == NULL) {
            _exit(1);
        }
        wrote = snprintf(head, sizeof(head),
                         "HTTP/1.1 200 OK\r\nContent-Length: %u\r\nConnection: close\r\n\r\n",
                         4096u);
        write_all(client, head, (size_t)wrote);
        write_all(client, data, 4096u);
        close(client);
        close(listen_fd);
        _exit(0);
    }
    close(listen_fd);
    return pid;
}

static void token_redirect_case(void)
{
    aotx_model_catalog_entry entry;
    char dir[128];
    char reason[512];
    unsigned short first_port;
    unsigned short second_port;
    int first;
    int second;
    char path[AOTX_MODEL_PATH];
    char store[2048] = "";
    int fd;
    ssize_t got;
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the token store does not open");
    second = content_server(&second_port);
    first = redirect_server(&first_port, second_port);
    entry_for(&entry, first_port, "token", "token.gguf", 4096u);
    setenv("HF_TOKEN", "fetch-test-token", 1);
    CHECK(aotx_model_fetch(dir, &entry, AOTX_MODEL_FETCH_TIMEOUT,
                           reason, sizeof(reason)) == 0,
          "the redirected fetch failed: %s", reason);
    unsetenv("HF_TOKEN");
    CHECK(aotx_wait(first) == 0, "the first host did not receive the token");
    CHECK(aotx_wait(second) == 0, "the redirect host received the token");
    check_file(dir, &entry);
    snprintf(path, sizeof(path), "%s/store.jsonl", dir);
    fd = open(path, O_RDONLY | O_CLOEXEC);
    got = (fd >= 0) ? read(fd, store, sizeof(store) - 1u) : -1;
    if (fd >= 0) {
        close(fd);
    }
    CHECK(got > 0 && strstr(store, "fetch-test-token") == NULL,
          "the local store holds the bearer token");
    aotx_remove_tree(dir);
}

static void link_refusal_case(void)
{
    aotx_model_catalog_entry entry;
    char dir[128];
    char path[AOTX_MODEL_PATH];
    char reason[512] = "";
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the link store does not open");
    entry_for(&entry, 9u, "link", "link.gguf", 4096u);
    snprintf(path, sizeof(path), "%s/%s.part", dir, entry.file);
    CHECK(symlink("/dev/full", path) == 0, "the part link does not form");
    CHECK(aotx_model_fetch(dir, &entry, AOTX_MODEL_FETCH_TIMEOUT,
                           reason, sizeof(reason)) != 0,
          "a part link was accepted");
    CHECK(strstr(reason, "regular file owned") != NULL,
          "the part link reason is not exact: %s", reason);
    unlink(path);
    snprintf(path, sizeof(path), "%s/%s", dir, entry.file);
    CHECK(symlink("/dev/full", path) == 0, "the target link does not form");
    reason[0] = '\0';
    CHECK(aotx_model_fetch(dir, &entry, AOTX_MODEL_FETCH_TIMEOUT,
                           reason, sizeof(reason)) != 0,
          "a rename target link was accepted");
    CHECK(strstr(reason, "regular file owned") != NULL,
          "the target link reason is not exact: %s", reason);
    aotx_remove_tree(dir);
}

static int redirect_bound_server(unsigned short *port)
{
    int listen_fd = open_server(port);
    int pid = fork();
    if (pid == 0) {
        int state = 0;
        signal(SIGPIPE, SIG_IGN);
        for (int hop = 0; hop < 11; ++hop) {
            char request[8192] = "";
            char answer_text[512];
            int client = accept4(listen_fd, NULL, NULL, SOCK_CLOEXEC);
            int wrote;
            if (client < 0 || read_request(client, request, sizeof(request)) != 0) {
                state = 1;
                break;
            }
            wrote = snprintf(answer_text, sizeof(answer_text),
                             "HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:%u/hop-%d\r\n"
                             "Content-Length: 0\r\nConnection: close\r\n\r\n",
                             *port, hop + 1);
            if (write_all(client, answer_text, (size_t)wrote) != 0) {
                state = 1;
            }
            close(client);
        }
        close(listen_fd);
        _exit(state);
    }
    close(listen_fd);
    return pid;
}

static void redirect_bound_case(void)
{
    aotx_model_catalog_entry entry;
    char dir[128];
    char reason[512] = "";
    unsigned short port;
    int server;
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the redirect store does not open");
    server = redirect_bound_server(&port);
    entry_for(&entry, port, "redirect-bound", "redirect-bound.gguf", 4096u);
    CHECK(aotx_model_fetch(dir, &entry, AOTX_MODEL_FETCH_TIMEOUT,
                           reason, sizeof(reason)) != 0,
          "an eleven-hop redirect was accepted");
    CHECK(strstr(reason, "redirect") != NULL,
          "the redirect reason does not name the bound: %s", reason);
    CHECK(aotx_wait(server) == 0, "the redirect-bound server failed");
    aotx_remove_tree(dir);
}

static long long clock_ms(void)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (long long)now.tv_sec * 1000ll + (long long)now.tv_nsec / 1000000ll;
}

static void connect_timeout_case(void)
{
    aotx_model_catalog_entry entry;
    char dir[128];
    char reason[512] = "";
    long long began;
    long long spent;
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the timeout store does not open");
    entry_for(&entry, 81u, "timeout", "timeout.gguf", 4096u);
    snprintf(entry.repository, sizeof(entry.repository), "http://192.0.2.1:81");
    began = clock_ms();
    CHECK(aotx_model_fetch(dir, &entry, 1u, reason, sizeof(reason)) != 0,
          "a non-routable connection was accepted");
    spent = clock_ms() - began;
    CHECK(spent <= 3000ll, "the one-second connect bound took %lld ms", spent);
    printf("connect timeout: %lld ms, %s\n", spent, reason);
    aotx_remove_tree(dir);
}

static int make_certificate(const char *cert, const char *key)
{
    int child = fork();
    if (child == 0) {
        int null_fd = open("/dev/null", O_WRONLY | O_CLOEXEC);
        if (null_fd >= 0) {
            dup2(null_fd, 1);
            dup2(null_fd, 2);
        }
        execlp("openssl", "openssl", "req", "-x509", "-newkey", "rsa:2048",
               "-keyout", key, "-out", cert, "-sha256", "-days", "1", "-nodes",
               "-subj", "/CN=localhost", (char *)NULL);
        _exit(127);
    }
    return aotx_wait(child);
}

static int tls_server(unsigned short port, const char *cert, const char *key)
{
    int child = fork();
    if (child == 0) {
        char service[16];
        int null_fd = open("/dev/null", O_WRONLY | O_CLOEXEC);
        snprintf(service, sizeof(service), "%u", port);
        if (null_fd >= 0) {
            dup2(null_fd, 1);
            dup2(null_fd, 2);
        }
        execlp("openssl", "openssl", "s_server", "-quiet", "-accept", service,
               "-cert", cert, "-key", key, "-www", "-naccept", "1", (char *)NULL);
        _exit(127);
    }
    return child;
}

static void tls_refusal_case(void)
{
    aotx_model_catalog_entry entry;
    char dir[128];
    char cert[AOTX_MODEL_PATH];
    char key[AOTX_MODEL_PATH];
    char reason[512] = "";
    unsigned short port;
    int socket_fd;
    int server;
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the TLS store does not open");
    snprintf(cert, sizeof(cert), "%s/cert.pem", dir);
    snprintf(key, sizeof(key), "%s/key.pem", dir);
    CHECK(make_certificate(cert, key) == 0, "the self-signed certificate does not form");
    socket_fd = open_server(&port);
    CHECK(socket_fd >= 0, "the TLS port does not open");
    if (socket_fd >= 0) {
        close(socket_fd);
    }
    server = tls_server(port, cert, key);
    usleep(250000u);
    entry_for(&entry, port, "tls", "tls.gguf", 4096u);
    snprintf(entry.repository, sizeof(entry.repository), "https://127.0.0.1:%u", port);
    CHECK(aotx_model_fetch(dir, &entry, 2u, reason, sizeof(reason)) != 0,
          "a self-signed certificate was accepted");
    CHECK(strstr(reason, "certificate") != NULL || strstr(reason, "SSL") != NULL,
          "the TLS reason does not name verification: %s", reason);
    waitpid(server, NULL, 0);
    aotx_remove_tree(dir);
}

int main(void)
{
    whole_case(1);
    whole_case(64);
    resume_case();
    ignored_range_case();
    wrong_digest_case();
    wrong_part_case();
    killed_case();
    token_redirect_case();
    link_refusal_case();
    redirect_bound_case();
    connect_timeout_case();
    tls_refusal_case();
    return aotx_report("fetch_test", 50);
}

#else

int main(void)
{
    aotx_model_catalog_entry entry;
    char reason[192] = "";
    memset(&entry, 0, sizeof(entry));
    CHECK(aotx_model_fetch(".", &entry, 1u, reason, sizeof(reason)) != 0,
          "a build without libcurl accepted a fetch");
    CHECK(strstr(reason, "libcurl") != NULL,
          "the fetch refusal does not name libcurl: %s", reason);
    return aotx_report("fetch_test", 2);
}

#endif
