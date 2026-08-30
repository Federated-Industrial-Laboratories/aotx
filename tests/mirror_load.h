/* Purpose: Check an attached mirror under the load of sixteen workers.
 * Owns: The socket, reader thread and trial figures of the loaded case.
 * Threading: One reader thread beside the host thread that launches worker batches.
 * Lifetime: One loaded case of the mirror check. */
#ifndef AOTX_TEST_MIRROR_LOAD_H
#define AOTX_TEST_MIRROR_LOAD_H

#define AOTX_TEST_LOAD_TRIALS 3u

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

static unsigned long long aotx_test_median3(const unsigned long long value[3])
{
    unsigned long long a = value[0];
    unsigned long long b = value[1];
    unsigned long long c = value[2];
    if (a > b) { unsigned long long t = a; a = b; b = t; }
    if (b > c) { unsigned long long t = b; b = c; c = t; }
    if (a > b) { unsigned long long t = a; a = b; b = t; }
    return b;
}

/* Three trials compare sixteen workers with and without an attached mirror. A key goes
 * through the socket on every loaded tick, and a reader checks every frame at 30 Hz. */
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
    unsigned long long plain[AOTX_TEST_LOAD_TRIALS];
    unsigned long long live[AOTX_TEST_LOAD_TRIALS];
    unsigned int keys = 0u;
    int client;
    int descriptor;

    map.base = aotx_test_rings.inbound_map;
    map.bytes = (size_t)aotx_test_rings.inbound_bytes;
    map.fd = aotx_test_rings.inbound_fd;
    aotx_test_check(aotx_inbound_attach(&map, &ring) == 0, "the load takes the inbound ring");
    aotx_test_worker_run(30u, -1, NULL, NULL);
    for (unsigned int trial = 0u; trial < AOTX_TEST_LOAD_TRIALS; ++trial) {
        plain[trial] = aotx_test_worker_run(ticks, -1, NULL, NULL);
    }
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
    for (unsigned int trial = 0u; trial < AOTX_TEST_LOAD_TRIALS; ++trial) {
        live[trial] = aotx_test_worker_run(ticks, client, &attach, &ring);
        printf("mirror load trial %u: plain %.1f us, attached %.1f us\n", trial + 1u,
               (double)plain[trial] / ticks / 1000.0,
               (double)live[trial] / ticks / 1000.0);
    }
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
    unsigned long long plain_median = aotx_test_median3(plain);
    unsigned long long live_median = aotx_test_median3(live);
    double overhead = (plain_median > 0ull)
                    ? ((double)live_median / (double)plain_median - 1.0) * 100.0 : 100.0;
    printf("mirror load median: 16 workers, %u trials of %u ticks, plain %.1f us, "
           "attached %.1f us, overhead %.2f percent, %llu frames, %u keys, %llu lost, "
           "%llu torn\n", AOTX_TEST_LOAD_TRIALS, ticks,
           (double)plain_median / ticks / 1000.0,
           (double)live_median / ticks / 1000.0, overhead,
           frames_after - frames_before, keys, reader.gaps, reader.torn);
    aotx_test_check(live_median <= plain_median + plain_median * 8ull / 100ull,
                    "the median attached tick cost is within eight percent");
    aotx_test_check(frames_after > frames_before, "the load reads frames at 30 Hz");
    aotx_test_check(reader.gaps == 0ull, "the load reader loses no frame at 30 Hz");
    aotx_test_check(reader.torn == 0ull, "the load reader sees no torn frame");
    aotx_test_check(keys == ticks * AOTX_TEST_LOAD_TRIALS,
                    "every key of the load becomes a KEY record");
    close(client);
    aotx_attach_close(&attach);
    rmdir(dir);
}

#endif
