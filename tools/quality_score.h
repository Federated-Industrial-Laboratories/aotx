/* Purpose: Share the chat wrap and the pair mode entry between the two modes of the score tool.
 * Owns: Nothing; constants, the reader of a pairs line and one declaration.
 * Launch shape: Host glue; no kernel.
 * Lifetime: One program run. */
#ifndef AOTX_TOOLS_QUALITY_SCORE_H
#define AOTX_TOOLS_QUALITY_SCORE_H

#include "tools/steer_set.h"

/* The pieces of the chat wrap the say path puts around a text, with thinking off
 * (cli/prompt.cuh and agent/overlays.cuh). A user block holds a text between the user
 * head and the block end. An assistant block holds a reply between the assistant head and
 * the block end. A query ends with the assistant head, so the last row of the query holds
 * the logits of the first token of the answer. */
#define AOTX_SCORE_USER_HEAD      "<|im_start|>user\n"
#define AOTX_SCORE_BLOCK_END      "<|im_end|>\n"
#define AOTX_SCORE_ASSISTANT_HEAD "<|im_start|>assistant\n<think>\n\n</think>\n\n"

/* The bounds of a pairs file and a rubric. */
#define AOTX_PAIR_TURNS    16u
#define AOTX_PAIR_ITEMS    16u
#define AOTX_PAIR_ID       32u
#define AOTX_PAIR_NAME     128u
#define AOTX_PAIR_QUESTION 512u
#define AOTX_PAIR_LINE     (64u * 1024u)

typedef struct aotx_pair_turn { char *user, *a, *b; } aotx_pair_turn;
typedef struct aotx_pair { char name[AOTX_PAIR_NAME]; aotx_pair_turn turn[AOTX_PAIR_TURNS]; unsigned int turns, cut; } aotx_pair;
typedef struct aotx_pair_set {
    aotx_pair *pair; unsigned int pairs, items;
    char id[AOTX_PAIR_ITEMS][AOTX_PAIR_ID], question[AOTX_PAIR_ITEMS][AOTX_PAIR_QUESTION];
} aotx_pair_set;

static void json_space(const char **at) { while (**at == ' ' || **at == '\t' || **at == '\r' || **at == '\n') *at += 1; }
/* Read one JSON string into out as UTF-8. An escape of the form \uXXXX, with a surrogate
 * pair for a point above the plane, gives its bytes. The return is 0, or 1 for a string
 * that is malformed or does not fit. */
static int json_string(const char **at, char *out, size_t max)
{
    const char *s = *at; size_t n = 0;
    if (*s++ != '"') return 1;
    while (*s != '\0' && *s != '"') {
        unsigned int point = (unsigned char)*s; size_t need = 1;
        if (*s == '\\') {
            static const char from[] = "\"\\/bfnrt", to[] = "\"\\/\b\f\n\r\t";
            const char *k = strchr(from, s[1]);
            if (s[1] == 'u') {
                char *end = 0; char hex[5]; memcpy(hex, s + 2, 4); hex[4] = '\0';
                point = (unsigned int)strtoul(hex, &end, 16);
                if (end != hex + 4) return 1;
                s += 6;
                if (point >= 0xD800u && point < 0xDC00u && s[0] == '\\' && s[1] == 'u') {
                    memcpy(hex, s + 2, 4); unsigned int low = (unsigned int)strtoul(hex, &end, 16);
                    if (end != hex + 4 || low < 0xDC00u || low > 0xDFFFu) return 1;
                    point = 0x10000u + ((point - 0xD800u) << 10) + (low - 0xDC00u); s += 6;
                }
                need = (point < 0x80u) ? 1 : ((point < 0x800u) ? 2 : ((point < 0x10000u) ? 3 : 4));
            } else if (k != 0 && s[1] != '\0') { point = (unsigned char)to[k - from]; s += 2; }
            else return 1;
        } else s += 1;
        static const unsigned char lead[5] = { 0u, 0u, 0xC0u, 0xE0u, 0xF0u };
        if (n + need >= max) return 1;
        out[n++] = (need == 1) ? (char)point : (char)(lead[need] | (point >> (6u * (need - 1u))));
        for (size_t k = need - 1u; k > 0u; --k) out[n++] = (char)(0x80u | ((point >> (6u * (k - 1u))) & 0x3Fu));
    }
    if (*s != '"') return 1;
    out[n] = '\0'; *at = s + 1; return 0;
}

/* Read one pairs line: a name and the turns, each with the user text and the two replies,
 * every value a string. A line of another shape is malformed. */
static int read_pair_line(const char *line, aotx_pair *pair, char *value)
{
    const char *at = line; char key[16]; int named = 0;
    memset(pair, 0, sizeof *pair);
    json_space(&at); if (*at++ != '{') return 1;
    for (;;) {
        json_space(&at); if (json_string(&at, key, sizeof key)) return 1;
        json_space(&at); if (*at++ != ':') return 1; json_space(&at);
        if (!strcmp(key, "name")) { if (json_string(&at, pair->name, sizeof pair->name)) return 1; named = 1; }
        else if (!strcmp(key, "turns")) {
            if (*at++ != '[') return 1;
            for (;;) {
                json_space(&at);
                if (*at == ']') { at += 1; break; }
                if (pair->turns >= AOTX_PAIR_TURNS || *at++ != '{') return 1;
                aotx_pair_turn *turn = &pair->turn[pair->turns];
                for (;;) {
                    json_space(&at); if (json_string(&at, key, sizeof key)) return 1;
                    json_space(&at); if (*at++ != ':') return 1; json_space(&at);
                    if (json_string(&at, value, AOTX_PAIR_LINE)) return 1;
                    char **slot = !strcmp(key, "user") ? &turn->user : (!strcmp(key, "a") ? &turn->a : (!strcmp(key, "b") ? &turn->b : 0));
                    if (slot == 0 || *slot != 0 || (*slot = strdup(value)) == 0) return 1;
                    json_space(&at);
                    if (*at == ',') { at += 1; continue; }
                    if (*at++ != '}') return 1;
                    break;
                }
                if (turn->user == 0 || turn->a == 0 || turn->b == 0) return 1;
                pair->turns += 1u;
                json_space(&at);
                if (*at == ',') at += 1; else if (*at != ']') return 1;
            }
        } else return 1;
        json_space(&at);
        if (*at == ',') { at += 1; continue; }
        if (*at++ != '}') return 1;
        break;
    }
    json_space(&at);
    return !(named && pair->turns != 0u && *at == '\0');
}

/* The pair mode: score the two sides of each conversation of the pairs file on the rubric.
 * The pairs file of the output directory takes the figures. With blind set, the blinded
 * transcripts and their key go beside it. The return is the exit status of the program. */
int aotx_quality_pair(const char *models, const char *role, const char *pairs_path,
                      const char *rubric_path, const char *out_dir, int blind);

#endif
