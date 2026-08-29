/* Purpose: Check the shapes that the two sides of the seam agree on, over the real drain
 *   and the real feeder.
 * Owns: One temporary tree, one host ring and one inbound ring for each case.
 * Threading: Three processes; the check drives a device that publishes blocks, the drain
 *   derives the requests file, and the feeder answers it.
 * Lifetime: The run of one case. */
#ifndef AOTX_TESTS_FEED_SEAM_H
#define AOTX_TESTS_FEED_SEAM_H

/* The bytes one line of the requests file may take. */
#define AOTX_CHAIN_TEXT 2048

/* Reads a whole file of the fixture back. Returns the byte count, or -1. */
static long read_back(const char *path, char *out, size_t bytes)
{
    ssize_t n;
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return -1;
    }
    n = read(fd, out, bytes - 1u);
    close(fd);
    if (n < 0) {
        return -1;
    }
    out[n] = '\0';
    return (long)n;
}

/* The bytes of the argument text, stated here as a figure and not as the output of the
 * writer under check. The device writes these bytes into the request body and the feeder
 * reads them. A writer on either side that drifts from this shape fails here. */
static void argument_shape(void)
{
    static const char one[] = "\x1fpath=hello.txt";
    static const char two[] = "\x1fpath=out.txt\x1ftext=hi";
    const char *keys[2];
    const char *values[2];
    char raw[AOTX_TOOL_ARG_BYTES + 1];
    char text[256];
    aotx_args split;
    const char *reason = "";
    uint32_t len;

    keys[0] = "path";
    values[0] = "hello.txt";
    len = aotx_args_join(raw, sizeof(raw), keys, values, 1u);
    CHECK(len == (uint32_t)sizeof(one) - 1u && memcmp(raw, one, len) == 0,
          "one argument gives %u bytes and %u were asked for", len,
          (unsigned)sizeof(one) - 1u);
    aotx_json_write(text, sizeof(text), (const unsigned char *)raw, len);
    CHECK(strcmp(text, "\\u001fpath=hello.txt") == 0, "the line of one argument reads %s",
          text);
    CHECK(aotx_args_split(&split, raw, keys, 1u, &reason) == 1, "the split refuses: %s",
          reason);
    CHECK(split.count == 1u && strcmp(aotx_args_value(&split, "path"), "hello.txt") == 0,
          "the split of one argument gives %u pairs", split.count);

    keys[1] = "text";
    values[0] = "out.txt";
    values[1] = "hi";
    len = aotx_args_join(raw, sizeof(raw), keys, values, 2u);
    CHECK(len == (uint32_t)sizeof(two) - 1u && memcmp(raw, two, len) == 0,
          "two arguments give %u bytes and %u were asked for", len,
          (unsigned)sizeof(two) - 1u);
    CHECK(aotx_args_split(&split, raw, keys, 2u, &reason) == 1, "the split refuses: %s",
          reason);
    CHECK(split.count == 2u && strcmp(aotx_args_value(&split, "path"), "out.txt") == 0 &&
          strcmp(aotx_args_value(&split, "text"), "hi") == 0,
          "the split of two arguments gives %u pairs", split.count);
    /* A value that holds an equal sign is not a key, because the separator marks a pair. */
    values[0] = "a=b";
    len = aotx_args_join(raw, sizeof(raw), keys, values, 1u);
    CHECK(aotx_args_split(&split, raw, keys, 1u, &reason) == 1, "the split refuses: %s",
          reason);
    CHECK(strcmp(aotx_args_value(&split, "path"), "a=b") == 0,
          "a value with an equal sign reads %s", aotx_args_value(&split, "path"));
    printf("argument shape: one %u bytes, two %u bytes\n", (unsigned)sizeof(one) - 1u,
           (unsigned)sizeof(two) - 1u);
}

/* The one writer of a line of the requests file. The drain writes that file and the check
 * program of a module writes one line of it. The shape stands here one time, and both of
 * them are held to it. */
static void request_shape(void)
{
    aotx_tool_request_body r;
    const char *keys[1];
    const char *values[1];
    char line[AOTX_CHAIN_TEXT];
    size_t used;

    keys[0] = "path";
    values[0] = "hello.txt";
    memset(&r, 0, sizeof(r));
    r.agent = 2u;
    r.turn = 5u;
    r.tool = AOTX_TOOL_FS_READ;
    r.request = 41u;
    r.deadline = 500u;
    r.arg_len = aotx_args_join(r.arg, AOTX_TOOL_ARG_BYTES, keys, values, 1u);
    used = aotx_request_line(line, sizeof(line), &r, 7u, AOTX_AUTH_NONE);
    CHECK(used > 0, "the line of a request does not fit");
    CHECK(strcmp(line, "{\"request\":41,\"agent\":2,\"turn\":5,\"tool\":\"fs_read\","
                       "\"side\":\"host\",\"number\":3,\"arg\":\"\\u001fpath=hello.txt\","
                       "\"deadline\":500,\"auth\":\"none\",\"tick\":7}\n") == 0,
          "the line of a request reads %s", line);
    /* The feeder runs the tool that reads a module directory, so the line names the host. */
    r.tool = AOTX_TOOL_IMPORT;
    used = aotx_request_line(line, sizeof(line), &r, 7u, AOTX_AUTH_GRANTED);
    CHECK(used > 0 && strstr(line, "\"tool\":\"import\",\"side\":\"host\"") != NULL,
          "the line of an import reads %s", line);
    CHECK(strstr(line, "\"auth\":\"granted\"") != NULL, "the line names the authorization"
          " %s", line);
    /* A tool of the catalog has no name in the record, so the line carries the number. */
    r.tool = AOTX_TOOL_MODULE_BASE + 1u;
    used = aotx_request_line(line, sizeof(line), &r, 7u, AOTX_AUTH_NONE);
    CHECK(used > 0 && strstr(line, "\"tool\":\"module\",\"side\":\"module\",\"number\":17")
          != NULL, "the line of a module tool reads %s", line);
    printf("request shape: %d bytes\n", (int)used);
}

/* The line that the drain writes is the line the feeder reads. This case runs both
 * programs over one file, so a change to the format of one shows here. The feeder starts
 * first, as it does at a boot, and the drain makes the file at its first request. */
static void loop(int n)
{
    aotx_map map;
    aotx_host_ring ring;
    aotx_fake_device device;
    aotx_map imap;
    aotx_inbound_ring iring;
    replies *got = &collected;
    char dir[256];
    char root[320];
    char path[512];
    char requests[512];
    char fd_text[16];
    char *args[8];
    int feeder;
    int drain;
    int i;

    memset(got, 0, sizeof(*got));
    CHECK(aotx_temp_dir(dir, sizeof(dir)) == 0, "the temporary directory does not open");
    snprintf(root, sizeof(root), "%s/root", dir);
    CHECK(aotx_make_dir(root) == 0, "the root does not open");
    for (i = 0; i < n; i++) {
        char body[64];
        int bytes = snprintf(body, sizeof(body), "the bytes of file %d", i);
        snprintf(path, sizeof(path), "%s/file-%d.txt", root, i);
        write_file(path, (const unsigned char *)body, (size_t)bytes);
    }
    snprintf(requests, sizeof(requests), "%s/requests.jsonl", dir);

    CHECK(aotx_inbound_create(256u, &imap, &iring) == 0, "the inbound ring does not open");
    snprintf(fd_text, sizeof(fd_text), "%d", imap.fd);
    args[0] = arguments[1];
    args[1] = (char *)"--inbound-fd";
    args[2] = fd_text;
    args[3] = (char *)"--root";
    args[4] = root;
    args[5] = (char *)"--requests";
    args[6] = requests;
    args[7] = NULL;
    feeder = aotx_spawn(args, -1, -1);
    CHECK(feeder > 0, "the feeder does not start");
    wait_for_start(&iring);

    CHECK(aotx_host_ring_create(262144u, 0x00100b0000000001ull + (uint64_t)n, &map, &ring) == 0,
          "the host ring does not open");
    snprintf(fd_text, sizeof(fd_text), "%d", map.fd);
    args[0] = arguments[2];
    args[1] = (char *)"--ring-fd";
    args[2] = fd_text;
    args[3] = (char *)"--journal";
    args[4] = dir;
    args[5] = NULL;
    drain = aotx_spawn(args, -1, -1);
    CHECK(drain > 0, "the drain does not start");
    aotx_fake_start(&device, &ring, 0x00100b0000000001ull + (uint64_t)n);
    for (i = 0; i < n; i++) {
        aotx_tool_request_body r;
        const char *keys[2];
        const char *values[2];
        char name[64];
        char text[64];
        char command[64];
        device.writer = AOTX_WRITER_AGENT_BASE + (uint32_t)(i % 3);
        /* Every argument text is the text the device writes: one separator byte before
         * each key, then the key, an equal sign and the value. The check builds it with
         * the one writer of that shape. A drift between the two sides of the seam thus
         * shows here and not in a run. */
        keys[0] = "path";
        snprintf(name, sizeof(name), "file-%d.txt", i);
        values[0] = name;
        /* One request needs no authorization; the other waits for the operator and is
         * granted, and the grant carries no argument of its own. */
        aotx_fake_request(i, AOTX_AUTH_NONE, &r);
        r.tool = AOTX_TOOL_FS_READ;
        r.arg_len = aotx_args_join(r.arg, AOTX_TOOL_ARG_BYTES, keys, values, 1u);
        CHECK(r.arg_len > 0, "the argument text of request %d does not fit", i);
        aotx_fake_record(&device, AOTX_CLASS_B, AOTX_REC_TOOL_REQUEST, &r, sizeof(r));
        aotx_fake_request(2000 + i, AOTX_AUTH_PENDING, &r);
        r.tool = AOTX_TOOL_FS_READ;
        r.arg_len = aotx_args_join(r.arg, AOTX_TOOL_ARG_BYTES, keys, values, 1u);
        aotx_fake_record(&device, AOTX_CLASS_B, AOTX_REC_TOOL_REQUEST, &r, sizeof(r));
        aotx_fake_request(2000 + i, AOTX_AUTH_GRANTED, &r);
        r.tool = AOTX_TOOL_FS_READ;
        r.arg_len = 0;
        aotx_fake_record(&device, AOTX_CLASS_B, AOTX_REC_TOOL_REQUEST, &r, sizeof(r));
        /* A tool of two arguments, which the operator must permit. The grant carries no
         * argument, so the line takes both keys of the request that was held. */
        keys[1] = "text";
        snprintf(name, sizeof(name), "written-%d.txt", i);
        snprintf(text, sizeof(text), "the text of write %d", i);
        values[1] = text;
        aotx_fake_request(4000 + i, AOTX_AUTH_PENDING, &r);
        r.tool = AOTX_TOOL_FS_WRITE;
        r.arg_len = aotx_args_join(r.arg, AOTX_TOOL_ARG_BYTES, keys, values, 2u);
        CHECK(r.arg_len > 0, "the argument text of write %d does not fit", i);
        aotx_fake_record(&device, AOTX_CLASS_B, AOTX_REC_TOOL_REQUEST, &r, sizeof(r));
        aotx_fake_request(4000 + i, AOTX_AUTH_GRANTED, &r);
        r.tool = AOTX_TOOL_FS_WRITE;
        r.arg_len = 0;
        aotx_fake_record(&device, AOTX_CLASS_B, AOTX_REC_TOOL_REQUEST, &r, sizeof(r));
        /* A command line, which runs a program under the root. */
        keys[0] = "command";
        snprintf(command, sizeof(command), "echo ran-%d", i);
        values[0] = command;
        aotx_fake_request(6000 + i, AOTX_AUTH_NONE, &r);
        r.tool = AOTX_TOOL_RUN;
        r.arg_len = aotx_args_join(r.arg, AOTX_TOOL_ARG_BYTES, keys, values, 1u);
        aotx_fake_record(&device, AOTX_CLASS_B, AOTX_REC_TOOL_REQUEST, &r, sizeof(r));
        keys[0] = "path";
        if ((i + 1) % 32 == 0) {
            aotx_fake_commit(&device, 0);
        }
        collect(&iring, got);
    }
    aotx_fake_commit(&device, 0);
    aotx_store_release16(&ring.pre->closed, 1);
    CHECK(aotx_wait(drain) == 0, "the drain does not end with a clean status");

    wait_for(&iring, got, 4 * n);
    CHECK(got->count == 4 * n, "the feeder answered %d requests and %d were asked for",
          got->count, 4 * n);
    for (i = 0; i < n; i++) {
        char want[64];
        char got_text[128];
        reply *plain = entry_of(got, (uint32_t)(1000 + i));
        reply *granted = entry_of(got, (uint32_t)(3000 + i));
        reply *written = entry_of(got, (uint32_t)(5000 + i));
        reply *ran = entry_of(got, (uint32_t)(7000 + i));
        int bytes = snprintf(want, sizeof(want), "the bytes of file %d", i);
        CHECK(plain != NULL && granted != NULL && written != NULL && ran != NULL,
              "the reply table is full at request %d", i);
        if (plain == NULL || granted == NULL || written == NULL || ran == NULL) {
            continue;
        }
        CHECK(plain->status == AOTX_TOOL_OK && (int)plain->len == bytes,
              "the request that needs no authorization gives %u bytes, and the reason"
              " reads %s", plain->len, plain->reason);
        CHECK(memcmp(plain->bytes, want, (size_t)bytes) == 0,
              "the bytes of request %d are not the bytes of the file", i);
        /* The grant carried no path, so the line came from the request that was held. */
        CHECK(granted->status == AOTX_TOOL_OK && (int)granted->len == bytes,
              "the granted request gives %u bytes", granted->len);
        CHECK(memcmp(granted->bytes, want, (size_t)bytes) == 0,
              "the bytes of the granted request %d are not the bytes of the file", i);
        /* The tool of two arguments wrote the file that its second key names. */
        CHECK(written->status == AOTX_TOOL_OK, "the granted write %d gives the status %u"
              " and the reason %s", i, written->status, written->reason);
        snprintf(path, sizeof(path), "%s/written-%d.txt", root, i);
        snprintf(want, sizeof(want), "the text of write %d", i);
        CHECK(read_back(path, got_text, sizeof(got_text)) == (long)strlen(want) &&
              strcmp(got_text, want) == 0, "the write %d left %s", i, got_text);
        /* The command line ran under the root and its output came back. */
        snprintf(want, sizeof(want), "ran-%d\n", i);
        CHECK(ran->status == AOTX_TOOL_OK, "the command %d gives the status %u and the"
              " reason %s", i, ran->status, ran->reason);
        CHECK((int)ran->len == (int)strlen(want) &&
              memcmp(ran->bytes, want, strlen(want)) == 0,
              "the command %d wrote %u bytes", i, ran->len);
    }
    aotx_store_release16(&iring.pre->closed, 1);
    CHECK(aotx_wait(feeder) == 0, "the feeder does not end with a clean status");
    printf("loop %d: replies %d\n", n, got->count);
    aotx_map_release(&map);
    aotx_map_release(&imap);
    aotx_remove_tree(dir);
}

#endif
