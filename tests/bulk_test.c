/* Purpose: Check that the drain writes every payload of the bulk ring with its checksum.
 * Owns: One temporary journal, one journal ring and one bulk ring for each case.
 * Threading: Two processes; the test writes both rings while the drain reads them.
 * Lifetime: The run of the program. */
#include "tests/disk_fake.h"

#include <fcntl.h>

#define AOTX_JOURNAL_BYTES 262144u
#define AOTX_INDEX_MAX     8192
#define AOTX_PAYLOAD_MAX   2048
#define AOTX_WAIT_NS       15000000000ull
#define AOTX_HOLD_NS       200000000ull

static char **arguments;
static char index_text[AOTX_INDEX_MAX];
static unsigned char payload[AOTX_PAYLOAD_MAX];
static unsigned char readback[AOTX_PAYLOAD_MAX];

typedef struct ctx {
    aotx_map journal_map;
    aotx_map bulk_map;
    aotx_host_ring journal_ring;
    aotx_host_ring bulk_ring;
    aotx_fake_device journal_device;
    aotx_fake_device bulk_device;
    char dir[256];
    int echo_fd;
    int child;
} ctx;

static int slurp(const char *path, void *out, size_t bytes)
{
    int fd = open(path, O_RDONLY);
    ssize_t got;
    if (fd < 0) {
        return -1;
    }
    got = read(fd, out, bytes);
    close(fd);
    return (int)got;
}

static void start(ctx *c, uint64_t boot_id, uint64_t bulk_bytes, const char *derive)
{
    char ring_text[16];
    char bulk_text[16];
    char path[512];
    char *args[10];
    int n = 0;
    memset(c, 0, sizeof(*c));
    CHECK(aotx_temp_dir(c->dir, sizeof(c->dir)) == 0, "the temporary directory does not open");
    CHECK(aotx_host_ring_create(AOTX_JOURNAL_BYTES, boot_id, &c->journal_map, &c->journal_ring) == 0,
          "the journal ring does not open");
    CHECK(aotx_host_ring_create(bulk_bytes, boot_id, &c->bulk_map, &c->bulk_ring) == 0,
          "the bulk ring does not open");
    snprintf(ring_text, sizeof(ring_text), "%d", c->journal_map.fd);
    snprintf(bulk_text, sizeof(bulk_text), "%d", c->bulk_map.fd);
    snprintf(path, sizeof(path), "%s/echo.txt", c->dir);
    c->echo_fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    CHECK(c->echo_fd >= 0, "the echo file does not open");
    args[n++] = arguments[1];
    args[n++] = (char *)"--ring-fd";
    args[n++] = ring_text;
    args[n++] = (char *)"--bulk-fd";
    args[n++] = bulk_text;
    args[n++] = (char *)"--journal";
    args[n++] = c->dir;
    if (derive != NULL) {
        args[n++] = (char *)"--derive";
        args[n++] = (char *)derive;
    }
    args[n] = NULL;
    c->child = aotx_spawn(args, -1, c->echo_fd);
    CHECK(c->child > 0, "the drain does not start");
    aotx_fake_start(&c->journal_device, &c->journal_ring, boot_id);
    aotx_fake_start(&c->bulk_device, &c->bulk_ring, boot_id);
}

/* Ends the drain and gives its exit status. The rings stay mapped, so a case can still read
 * the cursor that the drain left. */
static int finish(ctx *c)
{
    aotx_store_release16(&c->bulk_ring.pre->closed, 1);
    aotx_store_release16(&c->journal_ring.pre->closed, 1);
    return aotx_wait(c->child);
}

static void release(ctx *c)
{
    close(c->echo_fd);
    aotx_map_release(&c->journal_map);
    aotx_map_release(&c->bulk_map);
    aotx_remove_tree(c->dir);
}

/* Gives the payload of one handle a content that no other handle holds. */
static uint32_t fill(int i)
{
    uint32_t len = (uint32_t)(13 + (i * 37) % 500);
    uint32_t j;
    for (j = 0; j < len; j++) {
        payload[j] = (unsigned char)((i * 131 + (int)j * 7) & 0xff);
    }
    return len;
}

/* Reads one row of the index. The first line names the columns. */
static int index_row(const char *tsv, int row, uint64_t *handle, uint64_t *tick, uint32_t *length,
                     uint32_t *crc)
{
    const char *at = tsv;
    int i;
    for (i = 0; i <= row; i++) {
        at = strchr(at, '\n');
        if (at == NULL) {
            return -1;
        }
        at++;
    }
    if (*at == '\0') {
        return -1;
    }
    {
        unsigned long long h = 0;
        unsigned long long t = 0;
        if (sscanf(at, "%llx\t%llu\t%u\t%x", &h, &t, length, crc) != 4) {
            return -1;
        }
        *handle = (uint64_t)h;
        *tick = (uint64_t)t;
    }
    return 0;
}

static void batch(int n)
{
    ctx c;
    char path[512];
    uint64_t handles[64];
    uint32_t lengths[64];
    uint32_t rounded[64];
    int i;

    CHECK(n <= 64, "the case holds at most 64 payloads");
    /* A bulk ring of this size holds a few payloads only, so the device writes pad blocks. */
    start(&c, 0x00b41c0000000001ull + (uint64_t)n, 8192u, NULL);
    for (i = 0; i < n; i++) {
        aotx_bulk_body body;
        uint32_t len = fill(i);
        handles[i] = aotx_fake_payload(&c.bulk_device, payload, len, 0);
        lengths[i] = len;
        rounded[i] = (len + 7u) & ~7u;
        memset(&body, 0, sizeof(body));
        body.handle = handles[i];
        body.length = len;
        body.kind = 1;
        c.journal_device.writer = AOTX_WRITER_SYSTEM;
        aotx_fake_record(&c.journal_device, AOTX_CLASS_B, AOTX_REC_BULK, &body, sizeof(body));
    }
    {
        aotx_commit_body commit;
        memset(&commit, 0, sizeof(commit));
        aotx_fake_record(&c.journal_device, AOTX_CLASS_A, AOTX_REC_TICK_COMMIT, &commit,
                         sizeof(commit));
        aotx_fake_commit(&c.journal_device, 0);
    }
    CHECK(finish(&c) == 0, "the drain does not end with a clean status");
    CHECK(aotx_host_ring_cursor(&c.bulk_ring) == aotx_host_ring_head(&c.bulk_ring),
          "the drain did not reach the head of the bulk ring");

    snprintf(path, sizeof(path), "%s/bulk/index.tsv", c.dir);
    memset(index_text, 0, sizeof(index_text));
    CHECK(slurp(path, index_text, sizeof(index_text) - 1) > 0, "the index does not read");
    CHECK(strncmp(index_text, "handle\ttick\tlength\tcrc\n", 23) == 0,
          "the index does not name its columns");
    for (i = 0; i < n; i++) {
        uint64_t handle = 0;
        uint64_t tick = 0;
        uint32_t length = 0;
        uint32_t crc = 0;
        uint32_t want = fill(i);
        int got;
        snprintf(path, sizeof(path), "%s/bulk/%016llx", c.dir, (unsigned long long)handles[i]);
        got = slurp(path, readback, sizeof(readback));
        CHECK(got == (int)rounded[i], "payload %d holds %d bytes and %u were written", i, got,
              rounded[i]);
        CHECK(got >= 0 && memcmp(readback, payload, want) == 0, "payload %d holds other bytes", i);
        {
            /* The block rounds the payload up to eight bytes, so the file can hold up to
             * seven bytes more than the record gives. Those bytes are zero. */
            uint32_t j;
            int clean = 1;
            for (j = lengths[i]; j < rounded[i]; j++) {
                clean = clean && (readback[j] == 0);
            }
            CHECK(clean == 1, "payload %d does not end with zero bytes", i);
        }
        CHECK(index_row(index_text, i, &handle, &tick, &length, &crc) == 0,
              "row %d of the index does not read", i);
        CHECK(handle == handles[i], "row %d names handle %llx and the file is %llx", i,
              (unsigned long long)handle, (unsigned long long)handles[i]);
        CHECK(length == rounded[i], "row %d gives length %u and %u were written", i, length,
              rounded[i]);
        CHECK(tick > 0, "row %d gives no tick", i);
        CHECK(crc == aotx_crc32c(readback, (size_t)rounded[i], 0),
              "row %d gives a checksum that the file does not give", i);
    }
    /* One byte of a payload changes, so the checksum of the index no longer matches. The
     * damage of a file that is already written is visible from the index alone. */
    if (n > 0) {
        uint64_t handle = 0;
        uint64_t tick = 0;
        uint32_t length = 0;
        uint32_t crc = 0;
        int fd;
        unsigned char bad = 0x5au;
        snprintf(path, sizeof(path), "%s/bulk/%016llx", c.dir, (unsigned long long)handles[0]);
        CHECK(index_row(index_text, 0, &handle, &tick, &length, &crc) == 0, "row 0 does not read");
        fd = open(path, O_WRONLY);
        CHECK(fd >= 0, "the payload does not open");
        CHECK(pwrite(fd, &bad, 1, 3) == 1, "the damage does not write");
        close(fd);
        CHECK(slurp(path, readback, sizeof(readback)) == (int)rounded[0],
              "the damaged payload does not read");
        CHECK(crc != aotx_crc32c(readback, (size_t)rounded[0], 0),
              "a damaged payload must not match the checksum of the index");
    }
    printf("batch %d: payloads %d, ring bytes 8192\n", n, n);
    release(&c);
}

/* A block whose sequence is still zero is under write, so the drain must leave it. */
static void double_load(void)
{
    ctx c;
    char path[512];
    uint32_t len = fill(3);
    uint32_t byte_len = AOTX_BLOCK_HEADER_BYTES + ((len + 7u) & ~7u);
    uint64_t handle;
    uint64_t offset;
    uint64_t deadline;
    struct stat st;
    start(&c, 0x00b41c0000000002ull, 65536u, NULL);
    handle = aotx_fake_payload(&c.bulk_device, payload, len, 1);
    offset = c.bulk_device.head - byte_len;
    snprintf(path, sizeof(path), "%s/bulk/%016llx", c.dir, (unsigned long long)handle);
    deadline = aotx_wall_ns() + AOTX_HOLD_NS;
    while (aotx_wall_ns() < deadline) {
        uint64_t backoff = 0;
        aotx_pause(&backoff);
    }
    CHECK(stat(path, &st) != 0, "a block that is under write must not reach a file");
    aotx_fake_release(&c.bulk_device, offset, handle);
    deadline = aotx_wall_ns() + AOTX_WAIT_NS;
    while (stat(path, &st) != 0 && aotx_wall_ns() < deadline) {
        uint64_t backoff = 0;
        aotx_pause(&backoff);
    }
    CHECK(stat(path, &st) == 0, "the block does not reach a file after it is published");
    CHECK(finish(&c) == 0, "the drain does not end with a clean status");
    printf("double load: handle %llx, bytes %lld\n", (unsigned long long)handle,
           (long long)st.st_size);
    release(&c);
}

/* Writes a block header with fields that the caller gives, to prove a guard. */
static void put_header(ctx *c, uint16_t kind, uint32_t byte_len, uint32_t record_count)
{
    unsigned char block[AOTX_BLOCK_HEADER_BYTES];
    aotx_block_header *h = (aotx_block_header *)block;
    memset(block, 0, sizeof(block));
    h->magic = AOTX_BLOCK_MAGIC;
    h->layout = AOTX_WIRE_LAYOUT;
    h->kind = kind;
    h->boot_id = c->bulk_device.boot_id;
    h->tick = 7;
    h->first_seq = c->bulk_device.block_seq + 1;
    h->record_count = record_count;
    h->byte_len = byte_len;
    c->bulk_device.block_seq++;
    aotx_fake_write(&c->bulk_device, block, AOTX_BLOCK_HEADER_BYTES, byte_len,
                    c->bulk_device.block_seq, 0);
}

/* The bulk ring carries payloads only, and a payload block holds no record. */
static void refuse_block(uint16_t kind, uint32_t byte_len, uint32_t record_count,
                         const char *what)
{
    ctx c;
    start(&c, 0x00b41c0000000003ull, 65536u, NULL);
    put_header(&c, kind, byte_len, record_count);
    CHECK(finish(&c) == AOTX_EXIT_FAULT, "%s must stop the drain", what);
    release(&c);
}

/* The switch without the payload type must still take the blocks, or the ring fills. */
static void filter(void)
{
    ctx c;
    char path[512];
    struct stat st;
    uint32_t len = fill(9);
    uint64_t handle;
    start(&c, 0x00b41c0000000004ull, 65536u, "console,note,bus");
    handle = aotx_fake_payload(&c.bulk_device, payload, len, 0);
    CHECK(finish(&c) == 0, "the drain does not end with a clean status");
    CHECK(aotx_host_ring_cursor(&c.bulk_ring) == aotx_host_ring_head(&c.bulk_ring),
          "the drain must take the blocks even when it writes no file");
    snprintf(path, sizeof(path), "%s/bulk/%016llx", c.dir, (unsigned long long)handle);
    CHECK(stat(path, &st) != 0, "the switch without the payload type must write no file");
    printf("switch console,note,bus: cursor at the head, no file\n");
    release(&c);
}

int main(int argc, char **argv)
{
    arguments = argv;
    if (argc < 2) {
        printf("usage: bulk_test <drain program>\n");
        return 1;
    }
    batch(1);
    batch(64);
    double_load();
    refuse_block(0, AOTX_BLOCK_HEADER_BYTES, 0, "a block of records on the bulk ring");
    refuse_block(AOTX_BLOCK_BULK, AOTX_BLOCK_HEADER_BYTES + 4u, 0, "a payload length of four");
    refuse_block(AOTX_BLOCK_BULK, AOTX_BLOCK_HEADER_BYTES + 8u, 1, "a payload block with a record");
    filter();
    return aotx_report("bulk_test", 40);
}
