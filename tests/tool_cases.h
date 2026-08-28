/* Purpose: Give the parser cases and the note texts that the tool check reads.
 * Owns: The case tables of the check.
 * Threading: One host thread builds the tables; the device reads a copy of them.
 * Lifetime: One run of the test program. */
#ifndef AOTX_TESTS_TOOL_CASES_H
#define AOTX_TESTS_TOOL_CASES_H

#include "tool/tool_state.cuh"

/* Bytes of one case text of the parser check. */
#define AOTX_TOOL_CASE_BYTES  512u

/* Cases of the parser check: 64 that hold a call and 16 that hold none. */
#define AOTX_TOOL_CASE_GOOD   64u
#define AOTX_TOOL_CASE_BAD    16u
#define AOTX_TOOL_CASES       (AOTX_TOOL_CASE_GOOD + AOTX_TOOL_CASE_BAD)

/* The sixteen shapes the parser must refuse. Each one names the defect it carries. */
static const char *aotx_tool_bad_case[AOTX_TOOL_CASE_BAD] = {
    "{\"name\": \"memory_recall\", \"arguments\": {\"text\": \"a\"}}",
    "<tool_call>\n{\"name\": \"memory_recall\", \"arguments\": {\"text\": \"a\"}}\n",
    "<tool_call>\n\"name\": \"memory_recall\", \"arguments\": {\"text\": \"a\"}\n</tool_call>",
    "<tool_call>\n{\"arguments\": {\"text\": \"a\"}}\n</tool_call>",
    "<tool_call>\n{\"name\": memory_recall, \"arguments\": {\"text\": \"a\"}}\n</tool_call>",
    "<tool_call>\n{\"name\": \"fs_write\", \"arguments\": {\"path\": \"a\"}}\n</tool_call>",
    "<tool_call>\n{\"name\": \"memory_recall\"}\n</tool_call>",
    "<tool_call>\n{\"name\": \"memory_recall\", \"arguments\": \"text\"}\n</tool_call>",
    "<tool_call>\n{\"name\": \"memory_recall\", \"arguments\": {\"text\": 7}}\n</tool_call>",
    "<tool_call>\n{\"name\": \"memory_recall\", \"arguments\": {\"text\": \"a}}\n</tool_call>",
    "<tool_call>\n{\"name\" \"memory_recall\", \"arguments\": {\"text\": \"a\"}}\n</tool_call>",
    "<tool_call>\n{\"name\": \"memory_recall\" \"arguments\": {\"text\": \"a\"}}\n</tool_call>",
    "<tool_call>\n{\"name\": \"memory_recall\", \"arguments\": {\"query\": \"a\"}}\n</tool_call>",
    "<tool_call>\n{\"name\": \"memory_recall\", \"arguments\": {}}\n</tool_call>",
    "<tool_call>\n{\"name\": \"memory_write\", \"arguments\": {\"provenance\": \"guessed\","
        " \"text\": \"a\"}}\n</tool_call>",
    "<tool_call>\n</tool_call>"
};

/* What each bad case carries, for the report. */
static const char *aotx_tool_bad_why[AOTX_TOOL_CASE_BAD] = {
    "no call tag", "no closing tag", "no object", "no name", "a name that is not a string",
    "a tool that is not in the table", "no arguments", "arguments that are not an object",
    "a value that is not a string", "a string with no end", "no colon after a key",
    "no comma between the members", "a key the tool does not take", "no argument",
    "a provenance that is not one of the four", "an empty call"
};

/* The three tools of the good cases, and the words the check writes into them. */
static const char *aotx_tool_noun[AOTX_TOOL_CASE_GOOD] = {
    "otter", "walrus", "heron", "badger", "lynx", "ibex", "marmot", "puffin",
    "kestrel", "gannet", "adder", "newt", "shrew", "stoat", "weasel", "polecat",
    "curlew", "dunlin", "godwit", "avocet", "wigeon", "teal", "pochard", "eider",
    "cello", "oboe", "bassoon", "clarinet", "timpani", "harpsichord", "viola", "lute",
    "granite", "basalt", "gabbro", "gneiss", "schist", "shale", "chalk", "flint",
    "bergamot", "juniper", "cardamom", "coriander", "fennel", "tarragon", "sorrel", "chervil",
    "harbour", "quarry", "foundry", "windmill", "aqueduct", "lighthouse", "causeway", "weir",
    "sextant", "theodolite", "barometer", "anemometer", "chronometer", "caliper", "lathe",
    "kiln"
};

/* Build the text of one good case. The three tools take their turn, so every tool of the
 * table is in the batch. The white space of the shape changes from case to case. */
static unsigned int aotx_tool_good_case(unsigned int i, char *out, unsigned int max)
{
    const char *noun = aotx_tool_noun[i % AOTX_TOOL_CASE_GOOD];
    const char *space = ((i % 3u) == 0u) ? "" : (((i % 3u) == 1u) ? "  " : "\n  ");
    unsigned int made = 0u;
    if ((i % 3u) == 0u) {
        made = (unsigned int)snprintf(out, max,
            "I will look this up.\n<tool_call>\n{\"name\": \"memory_recall\", "
            "\"arguments\": {\"text\": \"the %s of the list\"}}\n</tool_call>", noun);
    } else if ((i % 3u) == 1u) {
        made = (unsigned int)snprintf(out, max,
            "<tool_call>%s{%s\"name\":%s\"memory_write\",%s\"arguments\":%s"
            "{\"provenance\": \"computed\", \"text\": \"the %s is on the list\"}%s}%s"
            "</tool_call>", space, space, space, space, space, noun, space, space);
    } else {
        made = (unsigned int)snprintf(out, max,
            "<tool_call>\n{\"name\": \"fs_read\", \"arguments\": "
            "{\"path\": \"notes/%s.txt\"}}\n</tool_call>\nthat is the file.", noun);
    }
    return (made < max) ? made : (max - 1u);
}

/* The tool that a good case names. */
static unsigned int aotx_tool_good_tool(unsigned int i)
{
    if ((i % 3u) == 0u) {
        return AOTX_TOOL_MEMORY_RECALL;
    }
    if ((i % 3u) == 1u) {
        return AOTX_TOOL_MEMORY_WRITE;
    }
    return AOTX_TOOL_FS_READ;
}

/* The argument that a good case gives. */
static unsigned int aotx_tool_good_arg(unsigned int i, char *out, unsigned int max)
{
    const char *noun = aotx_tool_noun[i % AOTX_TOOL_CASE_GOOD];
    unsigned int made = 0u;
    if ((i % 3u) == 0u) {
        made = (unsigned int)snprintf(out, max, "the %s of the list", noun);
    } else if ((i % 3u) == 1u) {
        made = (unsigned int)snprintf(out, max, "the %s is on the list", noun);
    } else {
        made = (unsigned int)snprintf(out, max, "notes/%s.txt", noun);
    }
    return (made < max) ? made : (max - 1u);
}

/* The text of one note of the memory check, and the query that must find it. */
static unsigned int aotx_tool_note_text(unsigned int i, char *out, unsigned int max)
{
    return (unsigned int)snprintf(out, max, "the %s stands in row %u of the table",
                                  aotx_tool_noun[i % AOTX_TOOL_CASE_GOOD], i);
}

static unsigned int aotx_tool_note_query(unsigned int i, char *out, unsigned int max)
{
    return (unsigned int)snprintf(out, max, "which row of the table holds the %s",
                                  aotx_tool_noun[i % AOTX_TOOL_CASE_GOOD]);
}

#endif
