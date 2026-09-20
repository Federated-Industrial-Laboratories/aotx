/* Purpose: Read GPU-selected cold payload batches without blocking journal drain.
 * Owns: One pinned source descriptor, read worker and response publication.
 * Threading: One worker reads; the drain starts and joins it without waiting for IO.
 * Lifetime: A read retains its source inode through file replacement. */
#include "disk/cognitive/cold_io.h"
#include "cognitive/checkpoint_io.h"
#include "disk/ccir/internal.h"
#include <pthread.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct aotx_cold_worker {
    pthread_t thread;
    aotx_ccir_view view;
    aotx_cold_catalog catalog;
    aotx_cold_transport *ring;
    uint64_t request;
    unsigned done;
} aotx_cold_worker;

static void cleanup(void *data) {
    aotx_cold_worker *w = data;
    aotx_cold_catalog_close(&w->catalog);
    close(w->view.fd);
}
static void *read_batch(void *data) {
    aotx_cold_worker *w = data;
    aotx_cold_transport *r = w->ring;
    uint64_t used = 0;
    int rc;
    pthread_cleanup_push(cleanup, w);
    rc = aotx_cold_catalog_open(&w->view, &w->catalog);
    for (uint32_t i = 0; !rc && i < r->count; ++i) {
        const unsigned char *row = r->rows[i];
        uint64_t n = aotx_ccir_u64(row + AOTX_CO_BYTES);
        int at = aotx_cold_catalog_find(&w->catalog, row);
        if (at < 0 || !n || n > AOTX_COG_PAYLOAD - used) { rc = AOTX_CCIR_INVALID; break; }
        aotx_sha256 hash; aotx_sha256_init(&hash);
        for (uint64_t pos = 0; !rc && pos < n;) {
            size_t take = n - pos < AOTX_COLD_COPY ? (size_t)(n - pos) : AOTX_COLD_COPY;
            rc = aotx_cold_catalog_read(&w->catalog, (uint32_t)at, pos, take, r->payload + used + pos);
            if (!rc) aotx_sha256_update(&hash, r->payload + used + pos, take);
            pos += take;
        }
        unsigned char digest[32]; aotx_sha256_final(&hash, digest);
        if (!rc && memcmp(digest, w->catalog.rows + (uint32_t)at * AOTX_COLD_EXTENT_ROW + AOTX_COG_OBJECT, 32))
            rc = AOTX_CCIR_INVALID;
        used += n;
    }
    r->bytes = rc ? 0 : (uint32_t)used; r->status = (uint32_t)rc;
    __atomic_store_n(&r->response, w->request, __ATOMIC_RELEASE);
    pthread_cleanup_pop(1);
    __atomic_store_n(&w->done, 1, __ATOMIC_RELEASE);
    return NULL;
}
int aotx_cold_disk_pass(aotx_checkpoint_disk *d) {
    if (!d->ring) return 0;
    if (d->cold_worker) {
        if (!__atomic_load_n(&d->cold_worker->done, __ATOMIC_ACQUIRE)) return 0;
        pthread_join(d->cold_worker->thread, NULL);
        free(d->cold_worker); d->cold_worker = NULL;
    }
    aotx_cold_transport *r = (aotx_cold_transport *)((unsigned char *)d->ring + AOTX_CP_COLD_OFFSET);
    uint64_t request = __atomic_load_n(&r->request, __ATOMIC_ACQUIRE);
    if (request == __atomic_load_n(&r->response, __ATOMIC_ACQUIRE)) return 0;
    int valid = request && r->boot == d->ring->boot && r->count && r->count <= AOTX_COG_OBJECTS &&
        !r->reserved && d->view.fd >= 0 && r->generation == d->view.generation &&
        !memcmp(r->incarnation, d->view.incarnation, 16);
    aotx_cold_worker *w = valid ? calloc(1, sizeof(*w)) : NULL;
    if (w) {
        w->view = d->view; w->view.fd = fcntl(d->view.fd, F_DUPFD_CLOEXEC, 0);
        w->ring = r; w->request = request;
        if (w->view.fd >= 0 && !pthread_create(&w->thread, NULL, read_batch, w)) {
            d->cold_worker = w; return 1;
        }
        if (w->view.fd >= 0) close(w->view.fd);
        free(w);
    }
    r->bytes = 0; r->status = valid ? AOTX_CCIR_IO : AOTX_CCIR_CHANGED;
    __atomic_store_n(&r->response, request, __ATOMIC_RELEASE);
    return 1;
}
void aotx_cold_disk_close(aotx_checkpoint_disk *d) {
    if (!d->cold_worker) return;
    pthread_cancel(d->cold_worker->thread);
    pthread_join(d->cold_worker->thread, NULL);
    free(d->cold_worker); d->cold_worker = NULL;
}
