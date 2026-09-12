/* Purpose: Define local service broker state and grant-file loading.
 * Owns: Socket and file transport state only.
 * Threading: One nonblocking transfer owner.
 * Lifetime: One broker process. */
#ifndef AOTX_SERVICE_BROKER_H
#define AOTX_SERVICE_BROKER_H
#include "service/wire.h"
#include "service/profile.h"
#include <stddef.h>
typedef struct aotx_service_peer {
    int fd, pending;
    uint64_t touched;
} aotx_service_peer;
int aotx_service_grant_file(const char *path, aotx_service_mailbox *mailbox);
int aotx_service_listen(const char *journal, char *path, size_t bytes, int *lock);
uint64_t aotx_service_now(void);
#endif
