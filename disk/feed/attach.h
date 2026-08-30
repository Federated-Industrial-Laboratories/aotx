/* Purpose: Declare the socket that terminal programs attach to, and the frames it takes.
 * Owns: Nothing; the caller owns the state that the functions fill.
 * Threading: One thread, the feeder's loop; no function here blocks on a terminal.
 * Lifetime: From the open of the socket to the close of the feeder. */
#ifndef AOTX_FEED_ATTACH_H
#define AOTX_FEED_ATTACH_H

#include <poll.h>
#include <signal.h>
#include <stddef.h>
#include <stdint.h>

#include "cuda/seam/wire.h"
#include "cuda/ui/mirror.h"
#include "disk/feed/line.h"
#include "disk/wire/diskwire.h"

/* The socket sits in the journal directory beside the segment files. */
#define AOTX_ATTACH_NAME     "aotx.sock"
#define AOTX_ATTACH_MAX      8u    /* terminal programs at one time */

/* Each message of a terminal starts with one kind byte. A key frame carries the 16 bytes
 * of aotx_key_body. The feeder publishes that frame as a KEY record. A line carries four
 * bytes of length, least significant byte first, and then the bytes of the line. The
 * feeder publishes the line as an INPUT_LINE record. A message the feeder refuses gets a
 * reason frame back, which carries the same length prefix. */
#define AOTX_ATTACH_KEY      'K'
#define AOTX_ATTACH_LINE     'L'
#define AOTX_ATTACH_REASON   'R'

/* The one byte of payload that carries the mirror descriptor. A message with no payload
 * carries no control data on a socket of this kind. */
#define AOTX_ATTACH_MIRROR   'M'

/* The bytes of one message that a read did not complete: the kind byte, the four length
 * bytes, and the body. */
#define AOTX_ATTACH_PART     (AOTX_INPUT_LINE_BYTES + 5u)

typedef struct aotx_attach_client {
    int           fd;
    uint32_t      fill;                     /* bytes of a message that is not complete */
    unsigned char part[AOTX_ATTACH_PART];
} aotx_attach_client;

typedef struct aotx_attach {
    int                 listen_fd;          /* the socket that waits, or -1 */
    int                 dir_fd;             /* the journal directory */
    int                 mirror_fd;          /* the mirror to send, or -1 */
    aotx_mirror_preamble *preamble;         /* the mapped head of the mirror, or NULL */
    size_t              map_bytes;
    unsigned int        clients;
    aotx_attach_client  client[AOTX_ATTACH_MAX];
    char                path[AOTX_PATH_BYTES]; /* the path for a report only */
    uint64_t            joined;             /* terminals that attached */
    uint64_t            left;               /* terminals that went away */
    uint64_t            keys;               /* key frames published */
    uint64_t            lines;              /* lines published */
    uint64_t            refused;            /* messages and peers refused */
} aotx_attach;

/* Makes the socket at <dir>/aotx.sock with mode 0600 and maps the head of the mirror. A
 * mirror descriptor of -1 gives a socket that refuses every terminal with a reason, so a
 * terminal gets an answer and not a silence. Returns 0, or -1 with a line on the standard
 * error. A directory of NULL opens nothing and gives 0. */
int aotx_attach_open(aotx_attach *a, const char *dir, int mirror_fd);

/* Closes every terminal, takes the socket file away, and clears the attached count. */
void aotx_attach_close(aotx_attach *a);

/* Adds the socket and each terminal to the poll set. Returns the count of entries that
 * the call wrote, which is at most AOTX_ATTACH_MAX + 1. */
unsigned int aotx_attach_poll_set(const aotx_attach *a, struct pollfd *fds, unsigned int most);

/* Takes what the poll reported: accepts a terminal, sends it the mirror, and reads the
 * frames of every terminal that has bytes. Publishes one KEY record for each key frame and
 * one INPUT_LINE record for each line. Returns 0, or -1 when the ring closed or the stop
 * flag went to one. */
int aotx_attach_take(aotx_attach *a, const struct pollfd *fds, unsigned int count,
                     const aotx_inbound_ring *ring, const volatile sig_atomic_t *stop);

/* Judges one peer credential as the accept does. Returns 0 when the uid is the one given,
 * and 1 when it is not. The check of the socket calls this, and so does its test. */
int aotx_attach_peer_allowed(unsigned int peer_uid, unsigned int own_uid);

#endif
