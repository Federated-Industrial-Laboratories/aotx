/* Purpose: Give the catalog check its cases: the reader, the import path and the lists.
 * Owns: The module texts of each case.
 * Threading: One host thread drives the cases one at a time.
 * Lifetime: One run of the test program.
 *
 * The file is a part of the catalog check. It reads the kernels of that check, so it
 * comes after them in the same translation unit. */
#ifndef AOTX_TESTS_CATALOG_CASES_H
#define AOTX_TESTS_CATALOG_CASES_H

#include "catalog_kernels.h"
#include "catalog_more.h"
#include "catalog_fixes.h"

/* One case of the reader: a manifest text, the kind of the import, the name the head
 * gave, and the reason the reader must give. */
typedef struct aotx_catalog_test_case {
    const char  *text;
    unsigned int kind;
    const char  *head;
    unsigned int why;
    const char  *what;
} aotx_catalog_test_case;

/* The reader case. Each pair is a text the reader takes and a text it refuses, so every
 * guard of the reader is shown able to fail. */
static void aotx_catalog_test_reader(unsigned int *applied, unsigned int *failed)
{
    static const aotx_catalog_test_case cases[] = {
        { AOTX_CATALOG_TEST_TOOL, AOTX_MODULE_TOOL, "word_count",
          AOTX_CATALOG_WHY_NONE, "a tool manifest of every key" },
        { AOTX_CATALOG_TEST_ROLE, AOTX_MODULE_ROLE, "scribe",
          AOTX_CATALOG_WHY_NONE, "a role manifest of every key" },
        { AOTX_CATALOG_TEST_SKILL, AOTX_MODULE_SKILL, "how_to_count",
          AOTX_CATALOG_WHY_NONE, "a skill manifest of every key" },
        { "# a note\n\nkind: skill\nname: quiet\n\n# the tail\n", AOTX_MODULE_SKILL,
          "quiet", AOTX_CATALOG_WHY_NONE, "comments and empty lines are stepped over" },
        { "kind: skill\nname: red\ncolor: red\n", AOTX_MODULE_SKILL, "red",
          AOTX_CATALOG_WHY_KEY, "a key the kind does not take" },
        { "kind: skill\nname: bare\nthis line holds no mark\n", AOTX_MODULE_SKILL, "bare",
          AOTX_CATALOG_WHY_SHAPE, "a line that is not a key and a value" },
        { "name: nokind\ndescription: none\nversion: 1\n", AOTX_MODULE_SKILL, "nokind",
          AOTX_CATALOG_WHY_MISSING, "a manifest with no kind" },
        { "name: read_me\ndescription: Read this first.\n", AOTX_MODULE_SKILL, "read_me",
          AOTX_CATALOG_WHY_NONE, "the two keys of a skill head with no fence" },
        { "kind: skill\ndescription: none\n", AOTX_MODULE_SKILL, "",
          AOTX_CATALOG_WHY_MISSING, "a manifest with no name" },
        { "kind: skill\nname: Bad-Name\n", AOTX_MODULE_SKILL, "",
          AOTX_CATALOG_WHY_NAME, "a name outside the letters the catalog takes" },
        { "kind: role\nname: mixed\n", AOTX_MODULE_SKILL, "mixed",
          AOTX_CATALOG_WHY_KIND, "a kind that is not the kind of the import" },
        { "kind: tool\nname: five\narguments: a,b,c,d,e\n", AOTX_MODULE_TOOL, "five",
          AOTX_CATALOG_WHY_ARGS, "a tool of more argument keys than the bound" },
        { "kind: tool\nname: odd\nside: sideways\n", AOTX_MODULE_TOOL, "odd",
          AOTX_CATALOG_WHY_VALUE, "a side that is neither device nor host" },
        { "kind: tool\nname: slow\ndeadline: soon\n", AOTX_MODULE_TOOL, "slow",
          AOTX_CATALOG_WHY_VALUE, "a deadline that is not a count" },
        { "kind: role\nname: giant\nmodel: enormous\n", AOTX_MODULE_ROLE, "giant",
          AOTX_CATALOG_WHY_VALUE, "a model that is not a role of the model files" },
        { "kind: role\nname: small\nmodel: embedding\n", AOTX_MODULE_ROLE, "small",
          AOTX_CATALOG_WHY_NONE, "a model that is a role name of the model file list" },
        { "kind: skill\nname: other\n", AOTX_MODULE_SKILL, "expected",
          AOTX_CATALOG_WHY_HEAD, "a name that is not the name of the directory" },
    };
    const unsigned int count = (unsigned int)(sizeof cases / sizeof cases[0]);
    unsigned int *out = (unsigned int *)aotx_catalog_test_take(16u * sizeof(unsigned int));
    char *text = (char *)aotx_catalog_test_take(4096u);
    char *head = (char *)aotx_catalog_test_take(AOTX_CATALOG_NAME_BYTES);
    unsigned int fields[16];
    unsigned int took = 0u;
    unsigned int refused = 0u;

    for (unsigned int i = 0u; i < count; ++i) {
        unsigned int length = (unsigned int)strlen(cases[i].text);
        unsigned int head_len = (unsigned int)strlen(cases[i].head);
        aotx_check_runtime(cudaMemcpy(text, cases[i].text, length, cudaMemcpyHostToDevice),
                           "cudaMemcpy");
        aotx_check_runtime(cudaMemcpy(head, cases[i].head, head_len + 1u,
                                      cudaMemcpyHostToDevice), "cudaMemcpy");
        aotx_catalog_test_read_one<<<1, 1>>>(text, length, cases[i].kind, head, head_len,
                                             out);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_check_runtime(cudaMemcpy(fields, out, sizeof fields, cudaMemcpyDeviceToHost),
                           "cudaMemcpy");
        if (fields[0] != cases[i].why) {
            printf("catalog: the reader gave %u for %s, not %u\n", fields[0],
                   cases[i].what, cases[i].why);
        }
        aotx_catalog_test_check(fields[0] == cases[i].why, cases[i].what, applied, failed);
        took += (cases[i].why == AOTX_CATALOG_WHY_NONE) ? 1u : 0u;
        refused += (cases[i].why != AOTX_CATALOG_WHY_NONE && fields[0] == cases[i].why)
                 ? 1u : 0u;
        if (i == 0u) {
            /* The tool of every key gives every field of the tool row. */
            aotx_catalog_test_check(fields[2] == 1u && fields[7] == 400u
                                    && fields[8] == 20u
                                    && fields[9] == AOTX_CATALOG_SIDE_DEVICE
                                    && fields[10] == AOTX_CATALOG_AUTH_ALWAYS
                                    && fields[4] != 0u && fields[5] != 0u,
                                    "the tool row holds every value of its manifest",
                                    applied, failed);
        }
        if (i == 1u) {
            /* The role of every key gives every field of the role row. The two tools it
             * names are built-in tools, so no name of the list is unknown. */
            aotx_catalog_test_check(fields[11] == 5u && fields[13] == 3u
                                    && fields[12] == AOTX_MODEL_LANGUAGE_Q4
                                    && fields[6] == 0u,
                                    "the role row holds every value of its manifest",
                                    applied, failed);
        }
    }
    printf("catalog: the reader took %u manifests and refused %u of %u with the reason "
           "each one asks for\n", took, refused, count - took);
    cudaFree(out);
    cudaFree(text);
    cudaFree(head);
}

/* The head of a skill file: two keys and no other. */
static void aotx_catalog_test_head(unsigned int *applied, unsigned int *failed)
{
    static const aotx_catalog_test_case cases[] = {
        { "---\nname: read_me\ndescription: Read this before the work.\n---\n",
          AOTX_MODULE_SKILL, "read_me", AOTX_CATALOG_WHY_NONE,
          "the head of a skill file gives the name and the description" },
        { "---\nname: read_me\ndescription: Read this.\nlicense: none\n---\n",
          AOTX_MODULE_SKILL, "read_me", AOTX_CATALOG_WHY_KEY,
          "the head of a skill file takes no third key" },
        { "---\nname: read_me\ndescription: Read this.\n---\n",
          AOTX_MODULE_ROLE, "read_me", AOTX_CATALOG_WHY_KIND,
          "the head of a skill file is not the manifest of a role" },
    };
    const unsigned int count = (unsigned int)(sizeof cases / sizeof cases[0]);
    unsigned int *out = (unsigned int *)aotx_catalog_test_take(16u * sizeof(unsigned int));
    char *text = (char *)aotx_catalog_test_take(1024u);
    char *head = (char *)aotx_catalog_test_take(AOTX_CATALOG_NAME_BYTES);
    unsigned int fields[16];
    for (unsigned int i = 0u; i < count; ++i) {
        unsigned int length = (unsigned int)strlen(cases[i].text);
        unsigned int head_len = (unsigned int)strlen(cases[i].head);
        aotx_check_runtime(cudaMemcpy(text, cases[i].text, length, cudaMemcpyHostToDevice),
                           "cudaMemcpy");
        aotx_check_runtime(cudaMemcpy(head, cases[i].head, head_len + 1u,
                                      cudaMemcpyHostToDevice), "cudaMemcpy");
        aotx_catalog_test_read_one<<<1, 1>>>(text, length, cases[i].kind, head, head_len,
                                             out);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_check_runtime(cudaMemcpy(fields, out, sizeof fields, cudaMemcpyDeviceToHost),
                           "cudaMemcpy");
        if (fields[0] != cases[i].why) {
            printf("catalog: the head reader gave %u for %s, not %u\n", fields[0],
                   cases[i].what, cases[i].why);
        }
        aotx_catalog_test_check(fields[0] == cases[i].why, cases[i].what, applied, failed);
    }
    printf("catalog: the head of a skill file gives two keys and refuses a third\n");
    cudaFree(out);
    cudaFree(text);
    cudaFree(head);
}

#endif
