/* Purpose: Write one line of the requests file, which is the format of a host tool request.
 * Owns: Nothing; the caller owns the buffer that takes the line.
 * Threading: One thread; the functions hold no state.
 * Lifetime: The call.
 *
 * The drain writes this file from the records of the device, and the check program writes
 * one line of it for a module under test. Both link this file, so the format has one
 * writer and no second shape exists. */
#include "disk/wire/diskwire.h"

#include <stdio.h>
#include <string.h>

const char *aotx_tool_name(uint32_t tool)
{
    static const char *names[9] = { "none", "memory_recall", "memory_write", "fs_read",
                                    "fs_list", "fs_write", "fs_update", "run", "skill_use" };
    if (tool == AOTX_TOOL_IMPORT) {
        /* The name comes from the constant and not from a place in the table, so the
         * number of the tool can change in one header. */
        return "import";
    }
    if (tool <= 8u) {
        return names[tool];
    }
    /* A tool of the catalog has no name in a record. The feeder holds the name and the
     * program under the number, which the line states beside this word. */
    return (tool >= AOTX_TOOL_MODULE_BASE) ? "module" : "other";
}

const char *aotx_tool_side(uint32_t tool)
{
    if (tool >= AOTX_TOOL_MODULE_BASE) {
        return "module";
    }
    if (tool >= AOTX_TOOL_FS_READ && tool <= AOTX_TOOL_RUN) {
        return "host";
    }
    if (tool == AOTX_TOOL_IMPORT) {
        /* The feeder reads the module directory, so the tool runs on the host beside the
         * file tools. */
        return "host";
    }
    /* A device tool makes no line of this file. The word states where the tool runs. */
    return "device";
}

const char *aotx_auth_name(uint32_t auth)
{
    static const char *names[4] = { "none", "pending", "granted", "refused" };
    return (auth <= 3u) ? names[auth] : "other";
}

size_t aotx_request_line(char *out, size_t out_bytes, const aotx_tool_request_body *r,
                         uint64_t tick, uint32_t auth)
{
    char arg[AOTX_TOOL_ARG_BYTES * 6 + 8];
    uint32_t len = r->arg_len;
    int used;
    if (len > AOTX_TOOL_ARG_BYTES) {
        len = AOTX_TOOL_ARG_BYTES;
    }
    aotx_json_write(arg, sizeof(arg), (const unsigned char *)r->arg, len);
    used = snprintf(out, out_bytes,
                    "{\"request\":%u,\"agent\":%u,\"turn\":%u,\"tool\":\"%s\","
                    "\"side\":\"%s\",\"number\":%u,\"arg\":\"%s\","
                    "\"deadline\":%llu,\"auth\":\"%s\",\"tick\":%llu}\n",
                    r->request, r->agent, r->turn, aotx_tool_name(r->tool),
                    aotx_tool_side(r->tool), r->tool, arg,
                    (unsigned long long)r->deadline, aotx_auth_name(auth),
                    (unsigned long long)tick);
    if (used < 0 || (size_t)used >= out_bytes) {
        return 0;
    }
    return (size_t)used;
}
