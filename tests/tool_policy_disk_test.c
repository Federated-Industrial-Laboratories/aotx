/* Purpose: Check the real drain path for typed policy and console derivation selection.
 * Owns: One temporary journal and a synthetic host ring for each batch.
 * Threading: The test publishes blocks while the drain process consumes them.
 * Lifetime: One test process; every child and temporary directory is closed. */
#include "tests/disk_fake.h"
#include "cuda/tool/policy.h"
#include <fcntl.h>

static void aotx_policy_disk_batch(const char *drain, unsigned count, int enabled)
{
    aotx_map map;
    aotx_host_ring ring;
    aotx_fake_device device;
    char directory[256], path[512], descriptor[32], text[256];
    uint64_t boot = 900u + count + (enabled ? 0u : 1000u);
    CHECK(aotx_temp_dir(directory, sizeof directory) == 0, "temporary directory failed");
    CHECK(aotx_host_ring_create(262144u, boot, &map, &ring) == 0, "host ring failed");
    snprintf(descriptor, sizeof descriptor, "%d", map.fd);
    int sink = open("/dev/null", O_WRONLY);
    char *args[] = {(char *)drain, "--ring-fd", descriptor, "--journal", directory,
                    "--derive", enabled ? "console" : "none", NULL};
    int child = aotx_spawn(args, -1, sink);
    CHECK(child > 0, "drain did not start");
    aotx_fake_start(&device, &ring, boot);
    for (unsigned agent = 0u; agent < count; ++agent) {
        unsigned group = agent % AOTX_TOOL_POLICY_GROUPS;
        aotx_tool_policy_body body = {agent, 0u, 2u << (group * 2u), 1u << group, 1u << group};
        device.writer = AOTX_WRITER_SYSTEM;
        aotx_fake_record(&device, AOTX_CLASS_B, AOTX_REC_TOOL_POLICY, &body, sizeof body);
        device.writer = AOTX_WRITER_AGENT_BASE + agent;
        snprintf(text, sizeof text, "tools: agent %u defaults 0 choices 0 selected 0 effective 0", agent);
        aotx_fake_record(&device, AOTX_CLASS_B, AOTX_REC_CONSOLE, text, (uint32_t)strlen(text));
        aotx_fake_record(&device, AOTX_CLASS_B, AOTX_REC_TOOL_POLICY, &body, sizeof body);
    }
    aotx_commit_body commit = {0};
    device.writer = AOTX_WRITER_SYSTEM;
    aotx_fake_record(&device, AOTX_CLASS_A, AOTX_REC_TICK_COMMIT, &commit, sizeof commit);
    aotx_fake_commit(&device, 0);
    aotx_store_release16(&ring.pre->closed, 1);
    CHECK(aotx_wait(child) == 0, "drain exit failed");
    CHECK(aotx_host_ring_cursor(&ring) == aotx_host_ring_head(&ring), "drain did not consume the batch");
    snprintf(path, sizeof path, "%s/%016llx/tools.jsonl", directory, (unsigned long long)boot);
    FILE *file = fopen(path, "r");
    if (!enabled) {
        CHECK(file == NULL, "disabled derivation wrote a policy file");
    } else {
        CHECK(file != NULL, "policy file is missing");
        unsigned lines = 0u;
        if (file != NULL) {
            while (fgets(text, sizeof text, file) != NULL) {
                char expected[256];
                unsigned group = lines % AOTX_TOOL_POLICY_GROUPS;
                snprintf(expected, sizeof expected,
                    "{\"tick\":1,\"seq\":%u,\"agent\":%u,\"defaults\":0,\"choices\":%u,\"selected\":%u,\"effective\":%u}\n",
                    lines * 3u + 1u, lines, 2u << (group * 2u), 1u << group, 1u << group);
                CHECK(strcmp(text, expected) == 0, "typed status differs at agent %u", lines);
                ++lines;
            }
        }
        CHECK(lines == count, "console or invalid-writer records changed the line count");
    }
    if (file != NULL) fclose(file);
    close(sink);
    aotx_map_release(&map);
    aotx_remove_tree(directory);
}

int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    aotx_policy_disk_batch(argv[1], 1u, 1);
    aotx_policy_disk_batch(argv[1], 64u, 1);
    aotx_policy_disk_batch(argv[1], 1u, 0);
    aotx_policy_disk_batch(argv[1], 64u, 0);
    return aotx_report("disk tool policy", 91);
}
