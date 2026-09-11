/* Purpose: Read image file bytes and publish bounded image producer frames.
 * Owns: A file lease and transfer buffer; no pixels or model features run here.
 * Threading: One feeder thread; stop and ring closure interrupt every wait.
 * Lifetime: One explicit local-file command. */
#include "disk/feed/media_io.h"
#include "cuda/media/profile.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <unistd.h>

static int aotx_media_word(const unsigned char *line, unsigned length, unsigned *at, const char *word)
{
    unsigned n = (unsigned)strlen(word);
    if (*at > length || n > length - *at || memcmp(line + *at, word, n) ||
        (*at + n < length && line[*at+n] != ' ' && line[*at+n] != '\t')) return 0;
    *at += n;
    while (*at < length && (line[*at] == ' ' || line[*at] == '\t')) ++*at;
    return 1;
}
static int aotx_media_number(const unsigned char *line, unsigned length, unsigned *at, unsigned *value)
{
    uint64_t n = 0; unsigned first = *at;
    while (*at < length && line[*at] >= '0' && line[*at] <= '9') {
        n = n * 10u + line[(*at)++] - '0';
        if (n > UINT32_MAX) return 0;
    }
    if (*at == first || *at == length || (line[*at] != ' ' && line[*at] != '\t')) return 0;
    while (*at < length && (line[*at] == ' ' || line[*at] == '\t')) ++*at;
    *value = (unsigned)n; return 1;
}
static int aotx_media_control(const aotx_inbound_ring *r, const volatile sig_atomic_t *stop)
{
    uint64_t target = aotx_load_acquire(&r->pre->head);
    while (aotx_load_acquire(&r->pre->consumed) < target) {
        if ((stop && *stop) || r->pre->closed) return -1;
        usleep(1000);
    }
    return 0;
}
static int aotx_media_read(int fd, unsigned char *out, unsigned n, uint64_t at)
{
    unsigned done = 0;
    while (done < n) {
        ssize_t got = pread(fd, out + done, n - done, (off_t)(at + done));
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return 1;
        done += (unsigned)got;
    }
    return 0;
}
static int aotx_media_piece(int fd, unsigned char *out, unsigned n, uint64_t at,
                             const unsigned char *header, unsigned head)
{
    unsigned prefix = at < head ? (unsigned)(head - at) : 0;
    if (prefix > n) prefix = n;
    if (prefix) memcpy(out, header + at, prefix);
    return prefix < n ? aotx_media_read(fd, out + prefix, n - prefix, at + prefix - head) : 0;
}
static int aotx_media_file(aotx_media_producer *out, const char *path, unsigned slot,
                            unsigned scope, unsigned format, unsigned width, unsigned height,
                            const volatile sig_atomic_t *stop)
{
    int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    struct stat before, after;
    if (fd < 0) return 1;
    unsigned head = format == 2u ? AOTX_MEDIA_RGB_HEAD : 0;
    if (flock(fd, LOCK_SH | LOCK_NB) || fstat(fd, &before) || !S_ISREG(before.st_mode) ||
        before.st_size <= 0 || AOTX_MEDIA_BYTES < head || (uint64_t)before.st_size > AOTX_MEDIA_BYTES - head ||
        (format == 2u && (!width || !height ||
            (uint64_t)width * height != (uint64_t)before.st_size / 3u || before.st_size % 3))) {
        close(fd); return 1;
    }
    unsigned char *frame = calloc(1, AOTX_MEDIA_FRAME_BYTES);
    if (!frame) { close(fd); return 1; }
    unsigned char digest[32], transfer[16] = {0};
    int rc = getrandom(transfer, sizeof transfer, 0) != sizeof transfer;
    aotx_sha256 sha; aotx_sha256_init(&sha);
    unsigned char header[AOTX_MEDIA_RGB_HEAD] = {0};
    memcpy(header, "AOTXRGB1", 8); aotx_media_put(header + 8, width, 4);
    aotx_media_put(header + 12, height, 4); aotx_media_put(header + 16, (uint64_t)before.st_size, 8);
    uint64_t bytes = (uint64_t)before.st_size + head;
    for (uint64_t at = 0; !rc && at < bytes;) {
        unsigned take = bytes - at < AOTX_MEDIA_FRAME_DATA ? (unsigned)(bytes - at) : AOTX_MEDIA_FRAME_DATA;
        if ((stop && *stop) || aotx_media_piece(fd, frame + 64, take, at, header, head)) { rc = 1; break; }
        aotx_sha256_update(&sha, frame + 64, take); at += take;
    }
    aotx_sha256_final(&sha, digest);
    memset(frame, 0, AOTX_MEDIA_FRAME_BYTES);
    aotx_media_put(frame, AOTX_MEDIA_SCHEMA, 4); aotx_media_put(frame + 4, AOTX_MEDIA_BEGIN, 4);
    memcpy(frame + 8, transfer, 16); aotx_media_put(frame + 24, bytes, 8);
    aotx_media_put(frame + 40, 48, 4); aotx_media_put(frame + 44, slot, 4);
    aotx_media_put(frame + 48, scope, 4); aotx_media_put(frame + 64, format, 4);
    aotx_media_put(frame + 68, width, 4); aotx_media_put(frame + 72, height, 4);
    memcpy(frame + 80, digest, 32);
    int begun = 0;
    if (!rc) { rc = aotx_media_producer_put(out, frame, stop); begun = !rc; }
    if (!rc) rc = aotx_media_producer_wait(out, stop);
    for (uint64_t at = 0; !rc && at < bytes;) {
        unsigned take = bytes - at < AOTX_MEDIA_FRAME_DATA ? (unsigned)(bytes - at) : AOTX_MEDIA_FRAME_DATA;
        if (aotx_media_piece(fd, frame + 64, take, at, header, head)) { rc = 1; break; }
        aotx_media_put(frame + 4, AOTX_MEDIA_CHUNK, 4);
        aotx_media_put(frame + 32, at, 8); aotx_media_put(frame + 40, take, 4);
        rc = aotx_media_producer_put(out, frame, stop); at += take;
    }
    if (!rc && (fstat(fd, &after) || after.st_size != before.st_size ||
        after.st_mtim.tv_sec != before.st_mtim.tv_sec || after.st_mtim.tv_nsec != before.st_mtim.tv_nsec ||
        after.st_ctim.tv_sec != before.st_ctim.tv_sec || after.st_ctim.tv_nsec != before.st_ctim.tv_nsec)) rc = 1;
    if (begun) {
        aotx_media_put(frame + 4, rc ? AOTX_MEDIA_CANCEL : AOTX_MEDIA_END, 4);
        aotx_media_put(frame + 32, bytes, 8); aotx_media_put(frame + 40, 0, 4);
        int sent = aotx_media_producer_put(out, frame, stop);
        if (sent) rc = sent;
    }
    if (!rc) rc = aotx_media_producer_wait(out, stop);
    if (!rc) {
        char text[65], id[33]; aotx_sha256_text(digest, text);
        for (unsigned i = 0; i < 16; ++i) snprintf(id + 2u*i, 3, "%02x", transfer[i]);
        fprintf(stderr, "image: uploaded [image:%s] transfer %s\n", text, id);
    }
    free(frame); close(fd); return rc;
}
int aotx_media_feed_line(aotx_media_producer *out, const unsigned char *line, unsigned length,
                          const aotx_inbound_ring *control, const volatile sig_atomic_t *stop)
{
    unsigned at = 0, slot = 0, scope = 0, format = 0, width = 0, height = 0;
    while (at < length && (line[at] == ' ' || line[at] == '\t')) ++at;
    if (!aotx_media_word(line, length, &at, "image")) return 0;
    if (aotx_media_word(line, length, &at, "cancel")) {
        unsigned char *frame = calloc(1, AOTX_MEDIA_FRAME_BYTES);
        int valid = frame && out->pre && aotx_media_number(line, length, &at, &slot) && length-at == 32;
        for (unsigned i=0; valid && i<16; ++i) {
            unsigned value=0;
            for (unsigned j=0; j<2; ++j) {
                unsigned c=line[at+2*i+j];
                if (c>='0' && c<='9') c-='0';
                else if (c>='a' && c<='f') c=c-'a'+10;
                else {valid=0;break;}
                value=value*16+c;
            }
            frame[8+i]=(unsigned char)value;
        }
        int rc=1;
        if (valid) {
            aotx_media_put(frame,AOTX_MEDIA_SCHEMA,4); aotx_media_put(frame+4,AOTX_MEDIA_CANCEL,4);
            aotx_media_put(frame+44,slot,4);
            rc=aotx_media_control(control,stop);
            if (!rc) rc=aotx_media_producer_put(out,frame,stop);
            if (!rc) rc=aotx_media_producer_wait(out,stop);
        }
        fprintf(stderr,rc ? "image: cancel is refused\n" : "image: cancel is complete\n");
        free(frame);return rc<0 ? -1 : 1;
    }
    if (!aotx_media_word(line, length, &at, "load")) return 0;
    int valid = aotx_media_number(line, length, &at, &slot);
    if (aotx_media_word(line, length, &at, "private")) scope = AOTX_MEDIA_PRIVATE;
    else if (aotx_media_word(line, length, &at, "room")) scope = AOTX_MEDIA_ROOM;
    else if (aotx_media_word(line, length, &at, "shared")) scope = AOTX_MEDIA_SHARED;
    else valid = 0;
    if (aotx_media_word(line, length, &at, "jpeg")) format = 1;
    else if (aotx_media_word(line, length, &at, "rgb8")) {
        format = 2;
        valid &= aotx_media_number(line, length, &at, &width);
        valid &= aotx_media_number(line, length, &at, &height);
    } else valid = 0;
    char path[PATH_MAX];
    if (at >= length || length - at >= sizeof path) valid = 0;
    for (unsigned i = at; i < length; ++i) if (line[i] < 32 || line[i] == 127) valid = 0;
    if (!valid || !out->pre) {
        fprintf(stderr, "image: give load SLOT private|room|shared jpeg PATH, or rgb8 WIDTH HEIGHT PATH\n");
        return 1;
    }
    memcpy(path, line + at, length - at); path[length-at] = 0;
    if (aotx_media_control(control, stop)) return -1;
    int rc = aotx_media_file(out, path, slot, scope, format, width, height, stop);
    if (rc > 0) fprintf(stderr, "image: the source file is refused\n");
    return rc < 0 ? -1 : 1;
}
