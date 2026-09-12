/* Purpose: Expose the bounded image producer transport and local file commands.
 * Owns: The mapped producer ring; CUDA owns all image interpretation.
 * Threading: One feeder owns publication and waits for copied-frame acknowledgments.
 * Lifetime: Feeder startup through shutdown. */
#ifndef AOTX_FEED_MEDIA_IO_H
#define AOTX_FEED_MEDIA_IO_H
#include "disk/wire/diskwire.h"
#include "cuda/media/wire.h"
typedef struct aotx_media_producer {
    aotx_map map;
    aotx_media_preamble *pre;
    unsigned char *frames;
} aotx_media_producer;
int aotx_media_producer_open(int fd, aotx_media_producer *out);
void aotx_media_producer_close(aotx_media_producer *out);
int aotx_media_producer_put(aotx_media_producer *out, const unsigned char *frame,
                             const volatile sig_atomic_t *stop);
int aotx_media_producer_wait(aotx_media_producer *out, const volatile sig_atomic_t *stop);
/* Zero is another command; one is handled; minus one is a closed transport. */
int aotx_media_feed_line(aotx_media_producer *out, const unsigned char *line, unsigned length,
                          const aotx_inbound_ring *control, const volatile sig_atomic_t *stop);
#endif
