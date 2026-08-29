/* Purpose: Check the mirror node, the rule a terminal reads it by and the frame thread.
 * Owns: The rings of the check, the reader thread and the counts of the cases.
 * Launch shape: The mirror node is one block; the paint kernel takes the whole grid.
 * Lifetime: One run of the test program. */
#include <pthread.h>
#include <fcntl.h>
#include <poll.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>

#include <cuda.h>

#include "agent/agent.cuh"
#include "boot/check.h"
#include "sched/sched.cuh"
#include "cli/agents.cuh"
#include "mem/mem.cuh"
#include "settings/settings.cuh"
#include "tool/tool.cuh"
#include "ui/mirror.cuh"

#include "catalog_feed.h"

extern "C" {
#include "disk/feed/attach.h"
}

/* Frames of the long run, and frames of the run under the slow reader. */
#define AOTX_TEST_FRAMES  1000u
#define AOTX_TEST_RACE    3000u

/* Microseconds the reader waits between two reads. The writer makes many frames in that
 * time, so a reader that waits loses frames. A reader that waits for nothing meets the
 * writer inside a slot far more often, which is where a torn frame would come from. */
#define AOTX_TEST_SLOW_US 300

static unsigned int aotx_test_applied;
static unsigned int aotx_test_failed;
static unsigned long long aotx_test_boot_id = 0x0f1de5c0ull;
static aotx_seam_rings aotx_test_rings;
static aotx_ui_panel aotx_test_panels[AOTX_UI_PANELS];
static aotx_ui_cell aotx_test_cells[AOTX_UI_CELLS];
__device__ unsigned long long aotx_test_worker_sink[16];

static unsigned long long aotx_test_published(void);

static void aotx_test_check(int ok, const char *what)
{
    aotx_test_applied += 1u;
    if (!ok) {
        aotx_test_failed += 1u;
        printf("mirror: FAILED %s\n", what);
    }
}

static long long aotx_test_now_ns(void)
{
    struct timespec at;
    clock_gettime(CLOCK_MONOTONIC, &at);
    return (long long)at.tv_sec * 1000000000ll + (long long)at.tv_nsec;
}

/* The name of one number setting, from the same list the device reads. */
static const char *aotx_test_setting_name(unsigned int index)
{
    switch (index) {
#define AOTX_TEST_ONE(symbol, name, side, effect, ...) case symbol: return name;
    AOTX_SETTING_NUMBERS(AOTX_TEST_ONE)
#undef AOTX_TEST_ONE
    default: return "";
    }
}

/* Paint every cell from one tag. A snapshot that mixes two frames holds a cell that no
 * one tag gives. That is what the reader of the race case looks for. */
__global__ void aotx_test_paint(unsigned int tag)
{
    unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int at = blockIdx.x * blockDim.x + threadIdx.x; at < AOTX_UI_CELLS;
         at += stride) {
        aotx_ui_grid[at].glyph = (unsigned char)((tag + at) % AOTX_UI_GLYPHS);
        aotx_ui_grid[at].attr = (unsigned char)(tag % AOTX_UI_ATTRS);
    }
}

/* Sixteen blocks stand for sixteen workers. Each block takes one row of the sink, and
 * each path takes the same batch of threads and operations. */
__global__ void aotx_test_worker_tick(unsigned int tick)
{
    unsigned long long value = (unsigned long long)tick + blockIdx.x + threadIdx.x;
    for (unsigned int i = 0u; i < 4096u; ++i) {
        value = value * 2862933555777941757ull + 3037000493ull;
    }
    if (threadIdx.x == 0u) {
        aotx_test_worker_sink[blockIdx.x] = value;
    }
}

/* Put count agents of a role in the table and free every slot after them. */
__global__ void aotx_test_agents_fill(unsigned int count, unsigned int role)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= AOTX_SLOTS) {
        return;
    }
    aotx_agent *agent = &aotx_agents.agent[id];
    agent->request = 0u;
    agent->tool = AOTX_ROLE_NONE;
    if (id < count) {
        agent->state = AOTX_AGENT_STATE_IDLE;
        agent->role = role;
        agent->task = id;
        agent->turn = id + 1u;
    } else {
        agent->state = AOTX_AGENT_STATE_FREE;
        agent->role = AOTX_ROLE_NONE;
        agent->task = 0xffffffffu;
        agent->turn = 0u;
    }
    if (id == 0u) {
        aotx_agents.live = count;
    }
}

/* Put count requests that wait for the operator in the table, the lowest number first. */
__global__ void aotx_test_requests_fill(unsigned int count, unsigned int tool)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= AOTX_SLOTS) {
        return;
    }
    aotx_request *slot = &aotx_requests.slot[id];
    if (id < count) {
        slot->request = id + 1u;
        slot->agent = id;
        slot->entry = tool;
        slot->auth = AOTX_AUTH_PENDING;
        slot->arg_len = 5u;
        for (unsigned int i = 0u; i < 5u; ++i) {
            slot->arg[i] = (char)('a' + (char)(id % 20u));
        }
    } else {
        slot->request = 0u;
        slot->auth = 0u;
        slot->arg_len = 0u;
    }
}

/* Put the editor cursor and the focus where the case asks for them. */
__global__ void aotx_test_editor(unsigned int length, unsigned int cursor,
                                 unsigned int focus)
{
    for (unsigned int i = 0u; i < length && i < AOTX_BODY_BYTES; ++i) {
        aotx_cli.line[i] = (unsigned char)'x';
    }
    aotx_cli.length = length;
    aotx_cli.cursor = cursor;
    aotx_cli_focus = focus;
}

/* Put the figures of the status line in the device state the head reads. */
__global__ void aotx_test_status(unsigned long long held, unsigned int live)
{
    aotx_sched.held_count = held;
    aotx_agents.live = live;
}

/* Give the frame rate of the mirror a value and publish the control page. */
__global__ void aotx_test_rate_set(unsigned int hz)
{
    aotx_setting_table.row[AOTX_SET_MIRROR_HZ].value = (long long)hz;
    aotx_settings_publish();
}

static const aotx_mirror_preamble *aotx_test_preamble(void)
{
    return (const aotx_mirror_preamble *)aotx_test_rings.mirror_map;
}

static const aotx_mirror_snapshot *aotx_test_slot(unsigned int which)
{
    const aotx_mirror_preamble *preamble = aotx_test_preamble();
    return (const aotx_mirror_snapshot *)(aotx_test_rings.mirror_map
                                          + sizeof(aotx_mirror_preamble)
                                          + (unsigned long long)which
                                            * preamble->slot_bytes);
}

static unsigned long long aotx_test_acquire(const uint64_t *at)
{
    return __atomic_load_n(at, __ATOMIC_ACQUIRE);
}

/* Set the count of terminals the feeder says are attached. The feeder writes this field in
 * a run; the check writes it, because no feeder runs here. */
static void aotx_test_attach(unsigned int count)
{
    aotx_mirror_preamble *preamble = (aotx_mirror_preamble *)aotx_test_rings.mirror_map;
    __atomic_store_n(&preamble->attached, count, __ATOMIC_RELEASE);
}

/* Take one slot by the rule of the mirror: an acquire load of the sequence, the copy, a
 * second acquire load. The copy is kept when both loads agree on a value that is not zero.
 * The return is the sequence of the copy, or zero. */
static unsigned long long aotx_test_take(unsigned int which, aotx_mirror_snapshot *out)
{
    const aotx_mirror_snapshot *at = aotx_test_slot(which);
    unsigned long long first = aotx_test_acquire(&at->head.sequence);
    if (first == 0ull) {
        return 0ull;
    }
    memcpy(out, at, sizeof *out);
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    unsigned long long second = aotx_test_acquire(&at->head.sequence);
    return (first == second) ? first : 0ull;
}

/* Take the newest slot the mirror holds. */
static unsigned long long aotx_test_newest(aotx_mirror_snapshot *out)
{
    unsigned long long best = 0ull;
    unsigned int which = 0u;
    for (unsigned int i = 0u; i < AOTX_MIRROR_SLOTS; ++i) {
        unsigned long long seq = aotx_test_acquire(&aotx_test_slot(i)->head.sequence);
        if (seq > best) {
            best = seq;
            which = i;
        }
    }
    if (best == 0ull) {
        return 0ull;
    }
    return aotx_test_take(which, out);
}

/* Report whether every cell of a snapshot comes from one tag. */
static int aotx_test_one_tag(const aotx_mirror_snapshot *snapshot)
{
    unsigned int tag = snapshot->cell[0].glyph;
    unsigned int attr = snapshot->cell[0].attribute;
    for (unsigned int at = 0u; at < AOTX_MIRROR_CELLS; ++at) {
        if (snapshot->cell[at].glyph != (unsigned char)((tag + at) % AOTX_UI_GLYPHS)
            || snapshot->cell[at].attribute != (unsigned char)attr) {
            return 0;
        }
    }
    return 1;
}

/* Run one frame of the mirror node alone, with the grid painted from the tag. */
static void aotx_test_frame(unsigned int tag)
{
    aotx_test_paint<<<64, 256>>>(tag);
    aotx_ui_mirror<<<1, AOTX_MIRROR_THREADS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

/* The preamble states the shape of a snapshot, and a snapshot fits one slot. */
static void aotx_test_layout(void)
{
    const aotx_mirror_preamble *preamble = aotx_test_preamble();
    aotx_test_check(preamble->magic == AOTX_MIRROR_MAGIC, "the preamble magic");
    aotx_test_check(preamble->layout == AOTX_MIRROR_LAYOUT, "the preamble layout");
    aotx_test_check(preamble->slots == AOTX_MIRROR_SLOTS, "the slot count");
    aotx_test_check(preamble->cols == AOTX_MIRROR_COLS
                    && preamble->rows == AOTX_MIRROR_ROWS, "the grid shape");
    aotx_test_check(preamble->slot_bytes >= sizeof(aotx_mirror_snapshot),
                    "a slot holds a snapshot");
    aotx_test_check((preamble->slot_bytes % 16u) == 0u, "a slot is a run of 16-byte stores");
    aotx_test_check(aotx_test_rings.mirror_bytes
                    >= sizeof(aotx_mirror_preamble)
                       + AOTX_MIRROR_SLOTS * (unsigned long long)preamble->slot_bytes,
                    "the file holds the preamble and every slot");
    aotx_test_check(aotx_test_rings.mirror_fd > 2, "the mirror has a descriptor");
}

/* The head carries the figures of the status line and the rectangle of each panel. */
static void aotx_test_head(void)
{
    aotx_mirror_snapshot snapshot;
    unsigned long long tick = 0ull;
    aotx_test_editor<<<1, 1>>>(6u, 4u, AOTX_CLI_FOCUS_AGENTS);
    aotx_test_status<<<1, 1>>>(9ull, 3u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_frame(7u);
    aotx_check_runtime(cudaMemcpyFromSymbol(&tick, aotx_time_tick, sizeof tick),
                       "cudaMemcpyFromSymbol");
    aotx_test_check(aotx_test_newest(&snapshot) != 0ull, "the head case takes a snapshot");
    aotx_test_check(snapshot.head.tick == tick, "the tick of the head");
    aotx_test_check(snapshot.head.boot_id == aotx_test_boot_id, "the boot id of the head");
    aotx_test_check(snapshot.head.slots == (unsigned int)AOTX_SLOTS, "the slot count");
    aotx_test_check(strcmp(snapshot.head.profile, AOTX_PROFILE_NAME) == 0, "the profile");
    char arch[8];
    snprintf(arch, sizeof arch, "sm_%d", (int)AOTX_ARCH);
    aotx_test_check(strcmp(snapshot.head.arch, arch) == 0, "the architecture");
    aotx_test_check(snapshot.head.focus == AOTX_CLI_FOCUS_AGENTS, "the focus");
    aotx_test_check(snapshot.head.held == 9ull, "the held ticks of the head");
    aotx_test_check(snapshot.head.agents_live == 3u, "the agents that are live");
    aotx_test_check(snapshot.head.requests_waiting == 0u, "no request waits");
    /* With no drain and no block in the host ring the disk side is not behind. */
    aotx_test_check(snapshot.head.drain_lag_ms == 0ull, "the drain lag of an empty ring");
    aotx_test_check(snapshot.head.language == 0xffffffffu, "no language model is resident");

    /* The cursor cell is the cell the window shows bright: the row of the command line and
     * the column of the prompt and the cursor. */
    const aotx_ui_panel *console = &aotx_test_panels[AOTX_UI_CONSOLE];
    aotx_test_check(snapshot.head.cursor_row
                    == (unsigned int)console->row + (unsigned int)console->rows - 1u,
                    "the cursor row");
    aotx_test_check(snapshot.head.cursor_col == (unsigned int)console->col + 3u + 4u,
                    "the cursor column");

    int panels = 1;
    for (unsigned int i = 0u; i < AOTX_MIRROR_PANELS; ++i) {
        panels = panels && snapshot.head.panel[i].row == aotx_test_panels[i].row
                        && snapshot.head.panel[i].col == aotx_test_panels[i].col
                        && snapshot.head.panel[i].rows == aotx_test_panels[i].rows
                        && snapshot.head.panel[i].cols == aotx_test_panels[i].cols
                        && snapshot.head.panel[i].name[0] != '\0';
    }
    aotx_test_check(panels != 0, "the panel table names the rectangle of every panel");
    aotx_test_check(strcmp(snapshot.head.panel[AOTX_UI_CONSOLE].name, "console") == 0,
                    "the name of the first panel");
}

/* Every frame of a long run holds the cells of the grid it was made from. */
static void aotx_test_long_run(unsigned int frames)
{
    aotx_mirror_snapshot snapshot;
    unsigned int wrong_seq = 0u;
    unsigned int wrong_cells = 0u;
    unsigned int lost = 0u;
    unsigned long long first = 0ull;
    aotx_check_runtime(cudaMemcpyFromSymbol(&first, aotx_mirror, sizeof first,
                                            offsetof(aotx_mirror_state, frame)),
                       "cudaMemcpyFromSymbol");
    for (unsigned int i = 1u; i <= frames; ++i) {
        aotx_test_frame(i);
        unsigned long long seq = aotx_test_newest(&snapshot);
        if (seq == 0ull) {
            lost += 1u;
            continue;
        }
        if (seq != first + (unsigned long long)i) {
            wrong_seq += 1u;
        }
        aotx_check_runtime(cudaMemcpyFromSymbol(aotx_test_cells, aotx_ui_grid,
                                                sizeof aotx_test_cells),
                           "cudaMemcpyFromSymbol");
        if (memcmp(snapshot.cell, aotx_test_cells, sizeof aotx_test_cells) != 0) {
            wrong_cells += 1u;
        }
    }
    printf("mirror: %u frames, %u lost, %u out of order, %u unlike the grid\n",
           frames, lost, wrong_seq, wrong_cells);
    aotx_test_check(lost == 0u, "no frame of the long run is refused");
    aotx_test_check(wrong_seq == 0u, "the sequence counts every frame");
    aotx_test_check(wrong_cells == 0u, "every frame holds the cells of the grid");
}

/* What the reader thread of the race case keeps. */
typedef struct aotx_test_reader {
    volatile int stop;
    int check_tag;              /* the synthetic writer gives every cell one tag */
    unsigned int wait_us;        /* microseconds between two reads */
    unsigned long long taken;    /* copies the rule kept */
    unsigned long long refused;  /* copies the rule refused */
    unsigned long long torn;     /* copies that mix two frames */
    unsigned long long gaps;     /* frames the reader did not see */
} aotx_test_reader;

static void *aotx_test_read_slow(void *state)
{
    aotx_test_reader *reader = (aotx_test_reader *)state;
    aotx_mirror_snapshot *snapshot = (aotx_mirror_snapshot *)malloc(sizeof *snapshot);
    unsigned long long last = 0ull;
    if (snapshot == NULL) {
        return NULL;
    }
    while (reader->stop == 0) {
        unsigned long long seq = aotx_test_newest(snapshot);
        if (seq == 0ull) {
            reader->refused += 1ull;
        } else {
            reader->taken += 1ull;
            if ((reader->check_tag != 0 && aotx_test_one_tag(snapshot) == 0)
                || (reader->check_tag == 0 && snapshot->head.sequence != seq)) {
                reader->torn += 1ull;
            }
            if (last != 0ull && seq > last + 1ull) {
                reader->gaps += seq - last - 1ull;
            }
            last = seq;
        }
        if (reader->wait_us != 0u) {
            usleep(reader->wait_us);
        }
    }
    free(snapshot);
    return NULL;
}

/* A reader never sees a torn frame, whatever its pace. The writer runs frames as fast as
 * the card takes them. A reader that waits loses frames. A reader that does not wait meets
 * the writer inside a slot, which is the state the sequence of the slot guards. */
static void aotx_test_slow_reader(unsigned int frames, unsigned int wait_us)
{
    aotx_test_reader reader;
    pthread_t thread;
    memset(&reader, 0, sizeof reader);
    reader.check_tag = 1;
    reader.wait_us = wait_us;
    if (pthread_create(&thread, NULL, aotx_test_read_slow, &reader) != 0) {
        aotx_test_check(0, "the reader thread starts");
        return;
    }
    for (unsigned int i = 1u; i <= frames; ++i) {
        aotx_test_frame(i);
    }
    reader.stop = 1;
    pthread_join(thread, NULL);
    printf("mirror: a reader that waits %u us took %llu, refused %llu, lost %llu of %u"
           " frames, torn %llu\n", wait_us, reader.taken, reader.refused, reader.gaps,
           frames, reader.torn);
    aotx_test_check(reader.taken > 0ull, "the reader keeps frames");
    aotx_test_check(reader.gaps > 0ull, "the reader loses frames");
    aotx_test_check(reader.torn == 0ull, "the reader sees no torn frame");
}

static int aotx_test_connect(const char *dir)
{
    struct sockaddr_un address;
    char path[sizeof(address.sun_path)];
    int dir_fd = open(dir, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (dir_fd < 0 || fd < 0) {
        return -1;
    }
    snprintf(path, sizeof(path), "/proc/self/fd/%d/%s", dir_fd, AOTX_ATTACH_NAME);
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    memcpy(address.sun_path, path, strlen(path));
    int state = connect(fd, (struct sockaddr *)&address, sizeof(address));
    close(dir_fd);
    if (state != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static void aotx_test_attach_drive(aotx_attach *attach, const aotx_inbound_ring *ring)
{
    struct pollfd fds[AOTX_ATTACH_MAX + 1u];
    unsigned int count = aotx_attach_poll_set(attach, fds, AOTX_ATTACH_MAX + 1u);
    if (count > 0u && poll(fds, (nfds_t)count, 20) > 0) {
        aotx_attach_take(attach, fds, count, ring, NULL);
    }
}

static int aotx_test_take_descriptor(int fd)
{
    struct msghdr message;
    struct iovec io;
    struct cmsghdr *control;
    union {
        char bytes[CMSG_SPACE(sizeof(int))];
        struct cmsghdr align;
    } room;
    char payload = 0;
    int mirror = -1;
    memset(&message, 0, sizeof(message));
    memset(&room, 0, sizeof(room));
    io.iov_base = &payload;
    io.iov_len = 1u;
    message.msg_iov = &io;
    message.msg_iovlen = 1u;
    message.msg_control = room.bytes;
    message.msg_controllen = sizeof(room.bytes);
    if (recvmsg(fd, &message, 0) != 1) {
        return -1;
    }
    control = CMSG_FIRSTHDR(&message);
    if (control != NULL && control->cmsg_type == SCM_RIGHTS) {
        memcpy(&mirror, CMSG_DATA(control), sizeof(mirror));
    }
    return mirror;
}

static void aotx_test_send_key(int fd, unsigned int code)
{
    unsigned char frame[1u + sizeof(aotx_key_body)];
    aotx_key_body body;
    memset(&body, 0, sizeof(body));
    body.key = code;
    body.action = 1u;
    frame[0] = (unsigned char)AOTX_ATTACH_KEY;
    memcpy(frame + 1u, &body, sizeof(body));
    if (write(fd, frame, sizeof(frame)) != (ssize_t)sizeof(frame)) {
        aotx_test_check(0, "a key frame goes to the attached mirror");
    }
}

static unsigned long long aotx_test_worker_run(unsigned int ticks, int client,
                                                aotx_attach *attach,
                                                const aotx_inbound_ring *ring)
{
    unsigned long long cost = 0ull;
    for (unsigned int tick = 0u; tick < ticks; ++tick) {
        long long period = aotx_test_now_ns();
        if (client >= 0) {
            aotx_test_send_key(client, 400u + tick);
        }
        long long started = aotx_test_now_ns();
        aotx_test_worker_tick<<<16, 128>>>(tick);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        cost += (unsigned long long)(aotx_test_now_ns() - started);
        if (client >= 0) {
            aotx_test_attach_drive(attach, ring);
        }
        long long left = 10000000ll - (aotx_test_now_ns() - period);
        if (left > 0ll) {
            usleep((useconds_t)(left / 1000ll));
        }
    }
    return cost;
}

/* Three hundred paced ticks compare sixteen workers with and without an attached mirror.
 * A key goes through the socket on every loaded tick, and a reader checks every frame. */
static void aotx_test_attached_load(void)
{
    const unsigned int ticks = 300u;
    char dir[] = "/tmp/aotx-mirror-load-XXXXXX";
    aotx_map map;
    aotx_inbound_ring ring;
    aotx_attach attach;
    aotx_test_reader reader;
    pthread_t reader_thread;
    unsigned long long frames_before;
    unsigned long long frames_after;
    unsigned long long plain;
    unsigned long long live;
    unsigned int keys = 0u;
    int client;
    int descriptor;

    map.base = aotx_test_rings.inbound_map;
    map.bytes = (size_t)aotx_test_rings.inbound_bytes;
    map.fd = aotx_test_rings.inbound_fd;
    aotx_test_check(aotx_inbound_attach(&map, &ring) == 0, "the load takes the inbound ring");
    aotx_test_worker_run(30u, -1, NULL, NULL);
    plain = aotx_test_worker_run(ticks, -1, NULL, NULL);
    aotx_test_check(mkdtemp(dir) != NULL, "the directory of the attached load opens");
    aotx_test_check(aotx_attach_open(&attach, dir, aotx_test_rings.mirror_fd) == 0,
                    "the socket of the attached load opens");
    client = aotx_test_connect(dir);
    aotx_test_check(client >= 0, "the load terminal connects");
    aotx_test_attach_drive(&attach, &ring);
    descriptor = aotx_test_take_descriptor(client);
    aotx_test_check(descriptor >= 0, "the load terminal receives the mirror");
    if (descriptor >= 0) {
        close(descriptor);
    }
    memset(&reader, 0, sizeof(reader));
    aotx_test_check(pthread_create(&reader_thread, NULL, aotx_test_read_slow, &reader) == 0,
                    "the frame reader of the load starts");
    frames_before = aotx_test_published();
    live = aotx_test_worker_run(ticks, client, &attach, &ring);
    frames_after = aotx_test_published();
    reader.stop = 1;
    pthread_join(reader_thread, NULL);
    for (uint64_t at = 0u; at < aotx_inbound_head(&ring); ++at) {
        const aotx_record_header *record = (const aotx_record_header *)(
            ring.slots + (at & ring.mask) * AOTX_SLOT_BYTES);
        if (record->type == AOTX_REC_KEY) {
            keys++;
        }
    }
    double overhead = (plain > 0ull) ? ((double)live / (double)plain - 1.0) * 100.0 : 100.0;
    printf("mirror load: 16 workers, %u ticks, plain %.1f us, attached %.1f us, "
           "overhead %.2f percent, %llu frames, %u of %u keys, %llu torn\n",
           ticks, (double)plain / ticks / 1000.0, (double)live / ticks / 1000.0,
           overhead, frames_after - frames_before, keys, ticks, reader.torn);
    aotx_test_check(live <= plain + plain * 8ull / 100ull,
                    "the attached tick cost is within eight percent");
    aotx_test_check(frames_after > frames_before, "the load reads frames at 30 Hz");
    aotx_test_check(reader.torn == 0ull, "the load reader sees no torn frame");
    aotx_test_check(keys == ticks, "every key of the load becomes a KEY record");
    close(client);
    aotx_attach_close(&attach);
    rmdir(dir);
}

/* The node stores zero into the sequence of a slot before it writes the bytes of it. The
 * store reaches the host before those bytes, so a reader that takes the slot at that
 * moment drops the copy. The host polls the slot while the node runs and counts the
 * launches at which it read the zero. */
static void aotx_test_zero_window(unsigned int frames)
{
    unsigned int saw = 0u;
    for (unsigned int i = 0u; i < frames; ++i) {
        unsigned long long frame = 0ull;
        aotx_check_runtime(cudaMemcpyFromSymbol(&frame, aotx_mirror, sizeof frame,
                                                offsetof(aotx_mirror_state, frame)),
                           "cudaMemcpyFromSymbol");
        /* The frame that follows takes the slot of its own number. */
        const aotx_mirror_snapshot *at =
            aotx_test_slot((unsigned int)(frame % (unsigned long long)AOTX_MIRROR_SLOTS));
        aotx_test_paint<<<64, 256>>>(i + 1u);
        aotx_ui_mirror<<<1, AOTX_MIRROR_THREADS>>>();
        for (;;) {
            if (aotx_test_acquire(&at->head.sequence) == 0ull) {
                saw += 1u;
                break;
            }
            if (cudaStreamQuery(0) == cudaSuccess) {
                break;
            }
        }
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }
    printf("mirror: the slot read zero at %u of %u launches\n", saw, frames);
    aotx_test_check(saw > 0u, "the slot reads zero while the node writes it");
}

/* The tables of a snapshot hold the rows of the device tables. */
static void aotx_test_tables(unsigned int agents, unsigned int requests)
{
    aotx_mirror_snapshot snapshot;
    unsigned int role = aotx_test_catalog_role_at(0u);
    aotx_test_agents_fill<<<1, AOTX_SLOTS>>>(agents, role);
    aotx_test_requests_fill<<<1, AOTX_SLOTS>>>(requests, role);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    /* Every slot takes the tables at the start of a run of frames, so two frames give both
     * slots the rows the device holds now. */
    for (unsigned int i = 0u; i < AOTX_MIRROR_TABLES_EVERY + AOTX_MIRROR_SLOTS; ++i) {
        aotx_test_frame(i + 1u);
    }
    aotx_test_check(aotx_test_newest(&snapshot) != 0ull, "the table case takes a snapshot");

    unsigned int live = 0u;
    int rows = 1;
    for (unsigned int i = 0u; i < AOTX_MIRROR_AGENT_ROWS; ++i) {
        const aotx_mirror_agent_row *row = &snapshot.tables.agent[i];
        if (row->role_name[0] == '\0') {
            continue;
        }
        live += 1u;
        rows = rows && row->id == i && row->state == AOTX_AGENT_STATE_IDLE
                    && row->task == i && row->turn == i + 1u
                    && row->pages == AOTX_KV_PAGES_EACH;
    }
    aotx_test_check(live == agents, "one agent row for each agent that is not free");
    aotx_test_check(rows != 0, "the agent rows hold the fields of the agent table");

    unsigned int waiting = 0u;
    int order = 1;
    for (unsigned int i = 0u; i < AOTX_MIRROR_REQUEST_ROWS; ++i) {
        const aotx_mirror_request_row *row = &snapshot.tables.request[i];
        if (row->request == 0u) {
            continue;
        }
        waiting += 1u;
        order = order && row->request == i + 1u && row->agent == i
                      && row->tool_name[0] != '\0';
    }
    unsigned int expect = (requests < AOTX_MIRROR_REQUEST_ROWS) ? requests
                                                                : AOTX_MIRROR_REQUEST_ROWS;
    aotx_test_check(waiting == expect, "one request row for each request that waits");
    aotx_test_check(order != 0, "the request rows hold the lowest numbers in order");

    int models = 1;
    for (unsigned int i = 0u; i < AOTX_MIRROR_MODEL_ROWS; ++i) {
        models = models && snapshot.tables.model[i].role == i
                        && snapshot.tables.model[i].resident == 0u;
    }
    aotx_test_check(models != 0, "a model row for each role, and none is resident");

    unsigned int installed = 0u;
    int modules = 1;
    for (unsigned int i = 0u; i < AOTX_MIRROR_MODULE_ROWS; ++i) {
        const aotx_mirror_module_row *row = &snapshot.tables.module[i];
        if (row->name[0] == '\0') {
            continue;
        }
        installed += 1u;
        modules = modules && (row->state == AOTX_CATALOG_INSTALLED
                              || row->reason[0] != '\0');
    }
    aotx_test_check(installed > 0u, "the module rows hold the catalog");
    aotx_test_check(modules != 0, "a refused module row states its reason");

    int settings = 1;
    for (unsigned int i = 0u; i < (unsigned int)AOTX_SETTING_NUMBER_COUNT; ++i) {
        settings = settings
                && strcmp(snapshot.tables.setting[i].key, aotx_test_setting_name(i)) == 0
                && snapshot.tables.setting[i].value == aotx_settings_default(i);
    }
    aotx_test_check(settings != 0, "the setting rows hold the key and the value of each row");
}

/* The tables go over every AOTX_MIRROR_TABLES_EVERY frames, and each slot says which frame
 * its own tables came from. */
static void aotx_test_tables_rate(void)
{
    aotx_mirror_snapshot snapshot;
    unsigned long long first = 0ull;
    unsigned int fresh = 0u;
    unsigned int stale = 0u;
    aotx_check_runtime(cudaMemcpyFromSymbol(&first, aotx_mirror, sizeof first,
                                            offsetof(aotx_mirror_state, frame)),
                       "cudaMemcpyFromSymbol");
    for (unsigned int i = 1u; i <= 2u * AOTX_MIRROR_TABLES_EVERY; ++i) {
        aotx_test_frame(i);
        if (aotx_test_newest(&snapshot) == 0ull) {
            continue;
        }
        unsigned long long frame = first + (unsigned long long)i;
        if (snapshot.head.tables_sequence == (unsigned int)frame) {
            fresh += 1u;
        } else {
            stale += 1u;
        }
    }
    printf("mirror: %u frames of %u carried the tables\n", fresh,
           2u * AOTX_MIRROR_TABLES_EVERY);
    aotx_test_check(fresh == 2u * AOTX_MIRROR_SLOTS,
                    "each slot takes the tables once in every run of frames");
    aotx_test_check(stale == 2u * AOTX_MIRROR_TABLES_EVERY - 2u * AOTX_MIRROR_SLOTS,
                    "a frame that carries no table keeps the frame of the tables it holds");
}

/* Read the frames the node published. */
static unsigned long long aotx_test_published(void)
{
    aotx_mirror_report report;
    aotx_mirror_read(&report);
    return report.frames;
}

/* The thread launches nothing while no terminal reads, and it launches at the rate of the
 * setting while one does. */
static void aotx_test_thread(unsigned int hz)
{
    aotx_test_rate_set<<<1, 1>>>(hz);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_check(aotx_settings_mirror_hz() == (unsigned long long)hz,
                    "the control page carries the frame rate");

    unsigned long long before = aotx_test_published();
    usleep(300000);
    aotx_test_check(aotx_test_published() == before,
                    "the thread launches nothing while no terminal is attached");

    aotx_test_attach(1u);
    long long started = aotx_test_now_ns();
    usleep(1000000);
    unsigned long long made = aotx_test_published() - before;
    long long spent = aotx_test_now_ns() - started;
    aotx_test_attach(0u);
    double rate = (double)made * 1e9 / (double)spent;
    printf("mirror: the thread made %llu frames in %lld ms at a set rate of %u Hz\n",
           made, spent / 1000000ll, hz);
    aotx_test_check(rate > (double)hz * 0.7 && rate < (double)hz * 1.3,
                    "the sequence advances at the rate of the setting");

    unsigned long long after = aotx_test_published();
    usleep(300000);
    aotx_test_check(aotx_test_published() == after,
                    "the thread stops when the last terminal leaves");
}

int main(void)
{
    CUdevice device;
    CUcontext context;
    aotx_mem_map map;

    aotx_check_driver(cuInit(0), "cuInit");
    aotx_check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    aotx_check_driver(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    aotx_check_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    if (aotx_mem_reserve(&map) != 0
        || aotx_seam_open(&aotx_test_rings, aotx_test_boot_id) != 0) {
        printf("mirror: the memory map or the rings did not open\n");
        return 1;
    }
    aotx_seam_bind(&aotx_test_rings, map.ring, map.ring_bytes, aotx_test_boot_id);
    if (aotx_mirror_bind(&aotx_test_rings) != 0) {
        printf("mirror: the mirror did not bind\n");
        return 1;
    }
    if (aotx_settings_page_open() != 0) {
        printf("mirror: the control page did not open\n");
        return 1;
    }
    if (aotx_test_catalog_setup() != 0) {
        printf("mirror: the catalog did not take the built-in tools and the roles\n");
        return 1;
    }
    aotx_check_runtime(cudaMemcpyFromSymbol(aotx_test_panels, aotx_ui_panel_table,
                                            sizeof aotx_test_panels), "cudaMemcpyFromSymbol");

    aotx_test_layout();
    aotx_test_head();
    aotx_test_long_run(AOTX_TEST_FRAMES);
    aotx_test_slow_reader(AOTX_TEST_RACE, AOTX_TEST_SLOW_US);
    aotx_test_slow_reader(AOTX_TEST_RACE, 0u);
    aotx_test_zero_window(200u);
    aotx_test_tables(1u, 1u);
    aotx_test_tables(AOTX_SLOTS, AOTX_SLOTS);
    aotx_test_tables_rate();
    if (aotx_mirror_start(&aotx_test_rings) != 0) {
        printf("mirror: the thread of the mirror did not start\n");
        return 1;
    }
    aotx_test_thread(30u);
    aotx_test_attached_load();
    aotx_test_thread(60u);
    aotx_mirror_stop();
    aotx_mirror_report_line();

    aotx_seam_close(&aotx_test_rings);
    aotx_mem_release(&map);
    aotx_settings_page_close();
    cuDevicePrimaryCtxRelease(device);
    printf("mirror: %u cases, %u failed\n", aotx_test_applied, aotx_test_failed);
    return (aotx_test_failed == 0u) ? 0 : 1;
}
