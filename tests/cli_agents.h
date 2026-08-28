/* Purpose: Check the agent commands, the authorization answers and the focus keystrokes.
 * Owns: The agent and request fixtures of the command line check.
 * Threading: One thread; the command line check calls these one at a time.
 * Lifetime: The program.
 *
 * The file is a part of the command line check. It reads the helpers of that check, so it
 * comes after them in the same translation unit. */
#ifndef AOTX_TEST_CLI_AGENTS_H
#define AOTX_TEST_CLI_AGENTS_H

/* Give the agent table and the request table back empty. */
__global__ void aotx_test_agents_clear(void)
{
    unsigned int at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at < AOTX_SLOTS) {
        aotx_agents.agent[at].state = AOTX_AGENT_STATE_FREE;
        aotx_agents.agent[at].task = ~0u;
        aotx_agents.agent[at].request = 0u;
        aotx_agents.agent[at].tool = 0u;
        aotx_agents.agent[at].turn = 0u;
        aotx_seqs.slot[at].state = AOTX_SEQ_STATE_FREE;
        aotx_seqs.slot[at].sampled = 0u;
    }
    if (at < AOTX_SLOTS) {
        aotx_requests.slot[at].request = 0u;
        aotx_requests.slot[at].auth = AOTX_AUTH_NONE;
    }
    /* The task table keeps its own free list, which is the agent module's. The check
     * therefore leaves it as it stands and reads the tasks it opens by their place after
     * the count that was there before. */
    if (at == 0u) {
        aotx_agents.live = 0u;
        aotx_requests.pending_auth = 0u;
        aotx_agent_gear[0].has_message = 0u;
    }
}

/* Fill the request table with requests that wait for the operator. The number of a request
 * falls as the slot rises. A reader that takes the slot order for the number order then
 * gives another first request, and the case fails. */
__global__ void aotx_test_requests_fill(unsigned int count)
{
    unsigned int at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= AOTX_SLOTS) {
        return;
    }
    aotx_request *slot = &aotx_requests.slot[at];
    if (at >= count) {
        slot->request = 0u;
        slot->auth = AOTX_AUTH_NONE;
        return;
    }
    slot->request = (count - at) * 10u;
    slot->agent = at + 3u;
    slot->tool = AOTX_TOOL_FS_READ;
    slot->auth = AOTX_AUTH_PENDING;
    slot->status = AOTX_TOOL_OK;
    slot->deadline = 0ull;
    /* The argument is longer than the panel shows, so the cut of the row has a figure to
     * cut. Each one differs from the next. */
    unsigned int put = 0u;
    const char *head = "models/notes/request-";
    for (unsigned int i = 0u; head[i] != '\0'; ++i) {
        slot->arg[put++] = head[i];
    }
    put += aotx_text_utoa((unsigned long long)slot->request, slot->arg + put,
                          AOTX_TOOL_ARG_BYTES - put);
    while (put < 60u) {
        slot->arg[put] = (char)('a' + (char)(put % 26u));
        put += 1u;
    }
    slot->arg_len = put;
    if (at == 0u) {
        aotx_requests.pending_auth = count;
    }
}

/* Set the state of one agent, so the busy arm of the task command has a busy agent. */
__global__ void aotx_test_agent_state(unsigned int id, unsigned int state)
{
    if (id < AOTX_SLOTS) {
        aotx_agents.agent[id].state = state;
    }
}

static aotx_agent_table *aotx_test_agent_table(void)
{
    static aotx_agent_table *table = NULL;
    if (table == NULL) {
        table = (aotx_agent_table *)malloc(sizeof *table);
    }
    aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_agents, sizeof *table),
                       "cudaMemcpyFromSymbol");
    return table;
}

static aotx_request_table *aotx_test_request_table(void)
{
    static aotx_request_table *table = NULL;
    if (table == NULL) {
        table = (aotx_request_table *)malloc(sizeof *table);
    }
    aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_requests, sizeof *table),
                       "cudaMemcpyFromSymbol");
    return table;
}

static unsigned int aotx_test_focus(void)
{
    unsigned int focus = 0u;
    aotx_check_runtime(cudaMemcpyFromSymbol(&focus, aotx_cli_focus, sizeof focus),
                       "cudaMemcpyFromSymbol");
    return focus;
}

/* Report whether the newest console line is the text. */
static int aotx_test_last_says(const char *text)
{
    aotx_console_state *console = (aotx_console_state *)malloc(sizeof *console);
    int same = 0;
    aotx_test_console_state(console);
    same = aotx_test_says(aotx_test_at(console, console->count), text);
    if (!same) {
        const aotx_console_line *line = aotx_test_at(console, console->count);
        printf("cli: the last console line is '%.*s', not '%s'\n",
               (line != NULL) ? (int)line->length : 0,
               (line != NULL) ? (const char *)line->text : "", text);
    }
    free(console);
    return same;
}

/* The spawn command makes agents of a role and states the slots they took. The check runs
 * at one agent and at a table that fills, where the spawn after the last one is refused. */
static void aotx_test_spawn(unsigned int count)
{
    const aotx_agent_table *table = NULL;
    char want[128];
    unsigned int at = 0u;
    unsigned int right = 0u;

    aotx_test_agents_clear<<<1, AOTX_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    if (count == 1u) {
        aotx_test_one("spawn conductor");
        table = aotx_test_agent_table();
        aotx_test_check(table->live == 1u && table->agent[0].state == AOTX_AGENT_STATE_IDLE
                        && table->agent[0].role == AOTX_ROLE_CONDUCTOR,
                        "one spawn makes one idle agent of the role on slot 0");
        aotx_test_check(aotx_test_last_says("spawn: conductor on slots 0"),
                        "the spawn command states the slot it took");
        aotx_test_one("spawn conductor");
        aotx_test_check(aotx_test_last_says("spawn: a conductor agent runs already"),
                        "a second conductor is refused with its own reason");
    } else {
        /* Slot 0 is the conductor and no worker takes it. One conductor and the workers of
         * every other slot therefore fill the table. The spawn command takes 8 at the
         * most, so the lines are groups of 8 with the rest in the last line. */
        unsigned int workers = AOTX_SLOTS - 1u;
        unsigned int lines = workers / 8u;
        unsigned int rest = workers % 8u;
        unsigned int expect = lines + ((rest > 0u) ? 1u : 0u);
        aotx_test_one("spawn conductor");
        for (unsigned int line = 0u; line < lines; ++line) {
            at = 0u;
            at += (unsigned int)snprintf(want + at, sizeof want - at,
                                         "spawn: worker on slots");
            for (unsigned int i = 0u; i < 8u; ++i) {
                at += (unsigned int)snprintf(want + at, sizeof want - at, " %u",
                                             1u + line * 8u + i);
            }
            aotx_test_one("spawn worker 8");
            if (aotx_test_last_says(want)) {
                right += 1u;
            }
        }
        if (rest > 0u) {
            char line[32];
            at = 0u;
            at += (unsigned int)snprintf(want + at, sizeof want - at,
                                         "spawn: worker on slots");
            for (unsigned int i = 0u; i < rest; ++i) {
                at += (unsigned int)snprintf(want + at, sizeof want - at, " %u",
                                             1u + lines * 8u + i);
            }
            snprintf(line, sizeof line, "spawn worker %u", rest);
            aotx_test_one(line);
            if (aotx_test_last_says(want)) {
                right += 1u;
            }
        }
        table = aotx_test_agent_table();
        aotx_test_check(right == expect, "every spawn line states the slots it took");
        aotx_test_check(table->live == count, "the spawns fill the agent table");
        aotx_test_one("spawn worker");
        aotx_test_check(aotx_test_last_says("spawn: the agent table is full"),
                        "the spawn after the last slot is refused");
        table = aotx_test_agent_table();
        aotx_test_check(table->live == count, "the refused spawn makes no agent");
    }
    printf("cli: spawn at %u agents left %u agents live\n", count,
           aotx_test_agent_table()->live);
}

/* The spawn command refuses a role it does not know and a count outside 1 to 8. */
static void aotx_test_spawn_refusals(void)
{
    aotx_cli_counts before;
    aotx_cli_counts after;
    static const char *bad[] = { "spawn wibble", "spawn", "spawn worker 9",
                                 "spawn worker 0", "spawn worker two" };
    const unsigned int count = (unsigned int)(sizeof bad / sizeof bad[0]);
    unsigned int live = 0u;

    aotx_test_agents_clear<<<1, AOTX_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    before = aotx_test_counts();
    aotx_test_one(bad[0]);
    aotx_test_check(aotx_test_last_says("spawn: the role is not known; give conductor, "
                                        "worker or verifier"),
                    "a spawn of a role that is not known names the roles");
    for (unsigned int i = 1u; i < count; ++i) {
        aotx_test_one(bad[i]);
    }
    aotx_test_check(aotx_test_last_says("spawn: give a count from 1 to 8"),
                    "a spawn with a count outside the bound names the bound");
    after = aotx_test_counts();
    live = aotx_test_agent_table()->live;
    aotx_test_check(after.refused == before.refused + count,
                    "every bad spawn line is refused");
    aotx_test_check(live == 0u, "a refused spawn makes no agent");
    printf("cli: %u bad spawn lines gave %u refusals and %u agents\n", count,
           after.refused - before.refused, live);
}

/* The task command opens a task for an agent or for a role. The word verify at the end of
 * the text asks for the check of a sibling. The check runs at one task and at 64. */
static void aotx_test_task(unsigned int count)
{
    const aotx_agent_table *table = NULL;
    char line[AOTX_BODY_BYTES];
    char want[AOTX_BODY_BYTES];
    unsigned int matched = 0u;
    unsigned int checked = 0u;

    aotx_test_agents_clear<<<1, AOTX_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_one("spawn worker 8");
    /* The task table keeps its own free list, so the tasks of this case start after the
     * ones an earlier case opened. */
    unsigned int before = aotx_test_agent_table()->tasks;

    for (unsigned int i = 0u; i < count; ++i) {
        snprintf(line, sizeof line, "task worker do the thing %u of %u%s", i, count,
                 (i % 2u == 1u) ? " verify" : "");
        aotx_test_one(line);
    }
    table = aotx_test_agent_table();
    for (unsigned int i = 0u; i < count; ++i) {
        unsigned int length = (unsigned int)snprintf(want, sizeof want,
                                                     "do the thing %u of %u", i, count);
        const aotx_task *task = &table->task[before + i];
        if (task->text_len == length && memcmp(task->text, want, length) == 0) {
            matched += 1u;
        }
        if (task->verify == ((i % 2u == 1u) ? AOTX_VERIFY_SIBLING : AOTX_VERIFY_NONE)) {
            checked += 1u;
        }
    }
    aotx_test_check(table->tasks == before + count,
                    "the task table holds one task for each line");
    aotx_test_check(matched == count, "every task holds the text of its own line");
    aotx_test_check(checked == count,
                    "the word verify at the end of a line asks for the check and no other "
                    "line does");
    printf("cli: %u task lines gave %u tasks, %u texts matched, %u marks matched\n", count,
           table->tasks - before, matched, checked);
}

/* The task command refuses four kinds of line. The first names a slot with no agent. The
 * second names neither an agent nor a role. The third names an agent that is busy. The
 * fourth carries no text. */
static void aotx_test_task_refusals(void)
{
    aotx_cli_counts before;
    aotx_cli_counts after;
    unsigned int tasks = 0u;

    aotx_test_agents_clear<<<1, AOTX_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_one("spawn worker 2");
    unsigned int tasks_before = aotx_test_agent_table()->tasks;
    before = aotx_test_counts();

    /* The first slot outside the table, whatever the profile gives. */
    char outside[AOTX_BODY_BYTES];
    char refusal[AOTX_BODY_BYTES];
    snprintf(outside, sizeof outside, "task %u read the file", (unsigned int)AOTX_SLOTS);
    snprintf(refusal, sizeof refusal,
             "task: the agent or the role is not known; give a slot below %u, or "
             "conductor, worker or verifier", (unsigned int)AOTX_SLOTS);
    aotx_test_one(outside);
    aotx_test_check(aotx_test_last_says(refusal),
                    "a task for a slot outside the table names the slots and the roles");
    aotx_test_one("task wibble read the file");
    aotx_test_check(aotx_test_last_says(refusal),
                    "a task for a name that is neither an agent nor a role is refused");
    aotx_test_one("task 7 read the file");
    aotx_test_check(aotx_test_last_says("task: no agent runs on that slot"),
                    "a task for a slot with no agent says that the slot is empty");
    aotx_test_agent_state<<<1, 1>>>(1u, AOTX_AGENT_STATE_RUN);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_one("task 1 read the file");
    aotx_test_check(aotx_test_last_says("task: the agent is busy"),
                    "a task for an agent that is not idle says that the agent is busy");
    aotx_test_one("task");
    aotx_test_check(aotx_test_last_says("task: give an agent or a role, and a text"),
                    "a task with no name and no text asks for both");
    aotx_test_one("task worker");
    aotx_test_check(aotx_test_last_says("task: the text is missing"),
                    "a task with a role and no text says that the text is missing");
    aotx_test_one("task worker verify");
    aotx_test_check(aotx_test_last_says("task: the text is missing"),
                    "a task of the mark alone has no text and is refused");

    after = aotx_test_counts();
    tasks = aotx_test_agent_table()->tasks - tasks_before;
    aotx_test_check(after.refused == before.refused + 7u, "every bad task line is refused");
    aotx_test_check(tasks == 0u, "a refused task line opens no task");
    printf("cli: 7 bad task lines gave %u refusals and %u tasks\n",
           after.refused - before.refused, tasks);
}

/* The authorise and the refuse commands answer a request that waits. A number that no
 * request holds is refused, and so is a request that took its answer already. */
static void aotx_test_authorise(unsigned int count)
{
    const aotx_request_table *table = NULL;
    aotx_cli_counts before;
    aotx_cli_counts after;
    char line[64];
    char want[80];
    unsigned int granted = 0u;
    unsigned int answered = 0u;

    aotx_test_agents_clear<<<1, AOTX_SLOTS>>>();
    aotx_test_requests_fill<<<1, AOTX_SLOTS>>>(count);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    before = aotx_test_counts();

    /* Every request takes an answer: the even numbers are granted and the odd ones are
     * refused, by the number and not by the slot. */
    for (unsigned int i = 0u; i < count; ++i) {
        unsigned int number = (count - i) * 10u;
        int grant = (i % 2u == 0u);
        snprintf(line, sizeof line, "%s %u", grant ? "authorise" : "refuse", number);
        snprintf(want, sizeof want, "%s: request %u is %s", grant ? "authorise" : "refuse",
                 number, grant ? "granted" : "refused");
        aotx_test_one(line);
        if (aotx_test_last_says(want)) {
            answered += 1u;
        }
        if (grant) {
            granted += 1u;
        }
    }
    table = aotx_test_request_table();
    unsigned int right = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        unsigned int wanted = (i % 2u == 0u) ? AOTX_AUTH_GRANTED : AOTX_AUTH_REFUSED;
        if (table->slot[i].auth == wanted) {
            right += 1u;
        }
    }
    aotx_test_check(answered == count, "every answer names its request and its verdict");
    aotx_test_check(right == count, "every request took the answer its own number was given");
    aotx_test_check(table->pending_auth == 0u, "no request waits after the answers");

    /* A number that no request holds, and a request that took its answer already. */
    aotx_test_one("authorise 999");
    aotx_test_check(aotx_test_last_says("authorise: no request of that number waits"),
                    "an answer to a number that no request holds is refused");
    snprintf(line, sizeof line, "refuse %u", count * 10u);
    aotx_test_one(line);
    aotx_test_check(aotx_test_last_says("refuse: no request of that number waits"),
                    "a second answer to a request is refused");
    aotx_test_one("authorise");
    aotx_test_check(aotx_test_last_says("authorise: no request of that number waits"),
                    "an authorise with no number is refused");
    after = aotx_test_counts();
    aotx_test_check(after.refused == before.refused + 3u,
                    "the three bad answers are the only refusals of this case");
    printf("cli: %u requests took %u answers, %u granted, %u still wait\n", count, answered,
           granted, table->pending_auth);
}

/* The tab key moves the focus. The editor takes no key while the focus is on the panel, and
 * the keys y and n answer the first request that waits. */
static void aotx_test_focus_keys(void)
{
    aotx_key_body keys[64];
    aotx_cli_state *state = (aotx_cli_state *)malloc(sizeof *state);
    const aotx_request_table *table = NULL;
    unsigned int at = 0u;
    unsigned int length = 0u;

    aotx_test_agents_clear<<<1, AOTX_SLOTS>>>();
    aotx_test_requests_fill<<<1, AOTX_SLOTS>>>(3u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    aotx_test_check(aotx_test_focus() == AOTX_CLI_FOCUS_CONSOLE,
                    "a run starts with the focus on the console");
    at = aotx_test_text(keys, 0u, "abc");
    aotx_test_send(keys, at);
    aotx_check_runtime(cudaMemcpyFromSymbol(state, aotx_cli, sizeof *state),
                       "cudaMemcpyFromSymbol");
    length = state->length;
    aotx_test_check(length == 3u, "the editor takes the keys while it has the focus");

    at = aotx_test_key(keys, 0u, AOTX_CLI_KEY_TAB);
    aotx_test_send(keys, at);
    aotx_test_check(aotx_test_focus() == AOTX_CLI_FOCUS_AGENTS,
                    "the tab key moves the focus to the agents panel");
    at = aotx_test_text(keys, 0u, "d");
    aotx_test_send(keys, at);
    aotx_check_runtime(cudaMemcpyFromSymbol(state, aotx_cli, sizeof *state),
                       "cudaMemcpyFromSymbol");
    aotx_test_check(state->length == length,
                    "the editor takes no key while the focus is on the panel");

    /* The keys answer the request of the lowest number, which is the last slot. */
    at = aotx_test_text(keys, 0u, "y");
    aotx_test_send(keys, at);
    table = aotx_test_request_table();
    aotx_test_check(table->slot[2].auth == AOTX_AUTH_GRANTED,
                    "the key y grants the first request that waits");
    aotx_test_check(aotx_test_last_says("authorise: request 10 is granted"),
                    "the console line names the request the key granted");
    at = aotx_test_text(keys, 0u, "n");
    aotx_test_send(keys, at);
    table = aotx_test_request_table();
    aotx_test_check(table->slot[1].auth == AOTX_AUTH_REFUSED,
                    "the key n refuses the request that is first after the grant");
    aotx_test_check(aotx_test_last_says("refuse: request 20 is refused"),
                    "the console line names the request the key refused");
    aotx_test_check(table->slot[0].auth == AOTX_AUTH_PENDING,
                    "the third request still waits");

    at = aotx_test_key(keys, 0u, AOTX_CLI_KEY_TAB);
    aotx_test_send(keys, at);
    aotx_test_check(aotx_test_focus() == AOTX_CLI_FOCUS_CONSOLE,
                    "the tab key gives the focus back to the console");
    at = aotx_test_text(keys, 0u, "e");
    at = aotx_test_key(keys, at, AOTX_CLI_KEY_ENTER);
    aotx_test_send(keys, at);
    aotx_check_runtime(cudaMemcpyFromSymbol(state, aotx_cli, sizeof *state),
                       "cudaMemcpyFromSymbol");
    aotx_test_check(state->length == 0u, "the editor takes the keys again");
    printf("cli: the focus keys answered 2 of 3 requests and the editor line held %u bytes\n",
           length);
    free(state);
}

/* The say command sends its text to the conductor agent. A run with no conductor refuses
 * the command and says so. */
static void aotx_test_say_conductor(void)
{
    const aotx_say_state *say = NULL;
    aotx_agent_work *gear = (aotx_agent_work *)malloc(sizeof *gear);
    aotx_cli_counts before;

    aotx_test_agents_clear<<<1, AOTX_SLOTS>>>();
    aotx_test_model<<<1, 1>>>(36u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    before = aotx_test_counts();

    aotx_test_one("say hello there");
    aotx_test_check(aotx_test_last_says("say: no conductor agent runs; give the spawn "
                                        "command"),
                    "a say with no conductor agent names the command that makes one");
    aotx_test_check(aotx_test_counts().refused == before.refused + 1u,
                    "the say with no conductor is refused");

    aotx_test_one("spawn conductor");
    aotx_test_one("say hello there");
    say = aotx_test_say_state();
    aotx_check_runtime(cudaMemcpyFromSymbol(gear, aotx_agent_gear, sizeof *gear),
                       "cudaMemcpyFromSymbol");
    aotx_test_check(gear->has_message != 0u && gear->message_len == 11u,
                    "the say command gives the message to the conductor agent");
    aotx_test_check(memcmp(gear->message, "hello there", 11u) == 0,
                    "the conductor holds the bytes of the text");
    aotx_test_check(say->slot[0].at != 0ull,
                    "the say command opens the console line the reply grows into");
    aotx_test_one("say again");
    aotx_test_check(aotx_test_last_says("say: a reply runs; give the stop command to end it"),
                    "a say while the conductor holds a message is refused");
    aotx_check_runtime(cudaMemcpyFromSymbol(gear, aotx_agent_gear, sizeof *gear),
                       "cudaMemcpyFromSymbol");
    aotx_test_check(gear->message_len == 11u,
                    "the refused say leaves the first message as it stands");
    printf("cli: the conductor holds a message of %u bytes\n", gear->message_len);
    free(gear);
}

/* The mailbox of an agent and the text of a task hold AOTX_TASK_TEXT_BYTES bytes. A text
 * of exactly that many bytes lands. A text of one byte more is refused, and the line names
 * the bound. The pair binds the bound from both sides, so an off by one cannot pass. */
static void aotx_test_text_bound(void)
{
    char line[AOTX_BODY_BYTES];
    char want[128];
    aotx_agent_work *gear = (aotx_agent_work *)malloc(sizeof *gear);
    const aotx_agent_table *table = NULL;
    aotx_cli_counts before;
    unsigned int tasks = 0u;
    const unsigned int bound = (unsigned int)AOTX_TASK_TEXT_BYTES;

    aotx_test_agents_clear<<<1, AOTX_SLOTS>>>();
    aotx_test_model<<<1, 1>>>(36u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_one("spawn conductor");
    aotx_test_one("spawn worker 1");
    before = aotx_test_counts();

    /* One byte over the bound, on the say command. */
    memset(line, 0, sizeof line);
    memcpy(line, "say ", 4u);
    memset(line + 4, 'a', bound + 1u);
    aotx_test_one(line);
    snprintf(want, sizeof want, "say: the text is too long; give %u bytes at most", bound);
    aotx_test_check(aotx_test_last_says(want),
                    "a say of one byte over the bound is refused and names the bound");
    aotx_check_runtime(cudaMemcpyFromSymbol(gear, aotx_agent_gear, sizeof *gear),
                       "cudaMemcpyFromSymbol");
    aotx_test_check(gear->has_message == 0u,
                    "the refused say gives the conductor no message");

    /* Exactly the bound, on the say command. */
    memset(line, 0, sizeof line);
    memcpy(line, "say ", 4u);
    memset(line + 4, 'b', bound);
    aotx_test_one(line);
    aotx_check_runtime(cudaMemcpyFromSymbol(gear, aotx_agent_gear, sizeof *gear),
                       "cudaMemcpyFromSymbol");
    aotx_test_check(gear->has_message != 0u && gear->message_len == bound,
                    "a say of the bound gives the conductor every byte of the text");

    /* One byte over the bound, on the task command. */
    tasks = aotx_test_agent_table()->tasks;
    memset(line, 0, sizeof line);
    memcpy(line, "task worker ", 12u);
    memset(line + 12, 'c', bound + 1u);
    aotx_test_one(line);
    snprintf(want, sizeof want, "task: the text is too long; give %u bytes at most", bound);
    aotx_test_check(aotx_test_last_says(want),
                    "a task of one byte over the bound is refused and names the bound");
    aotx_test_check(aotx_test_agent_table()->tasks == tasks,
                    "the refused task opens no task");

    /* Exactly the bound, on the task command. */
    memset(line, 0, sizeof line);
    memcpy(line, "task worker ", 12u);
    memset(line + 12, 'd', bound);
    aotx_test_one(line);
    table = aotx_test_agent_table();
    aotx_test_check(table->tasks == tasks + 1u, "a task of the bound opens one task");
    aotx_test_check(table->task[tasks].text_len == bound
                    && table->task[tasks].text[bound - 1u] == 'd',
                    "the task holds every byte of the text of the bound");
    aotx_test_check(aotx_test_counts().refused == before.refused + 2u,
                    "the two texts over the bound are the only refusals of this case");
    printf("cli: the bound of a text is %u bytes; one over it is refused on say and on "
           "task\n", bound);
    free(gear);
}

/* The agents command shows one row for each agent that is not free, with the columns of
 * the agents panel. */
static void aotx_test_agents_list(unsigned int count)
{
    aotx_test_record *found = (aotx_test_record *)malloc(AOTX_TEST_FOUND * sizeof *found);
    char want[128];
    unsigned int before = 0u;
    unsigned int after = 0u;
    unsigned int matched = 0u;
    unsigned int rows = 0u;

    aotx_test_agents_clear<<<1, AOTX_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_test_one("spawn conductor");
    for (unsigned int line = 1u; line < count; line += 8u) {
        aotx_test_one("spawn worker 8");
    }
    before = aotx_test_records(AOTX_REC_CONSOLE, found, AOTX_TEST_FOUND);
    aotx_test_one("agents");
    after = aotx_test_records(AOTX_REC_CONSOLE, found, AOTX_TEST_FOUND);
    /* The allowance of one line bounds the rows, so a full table gives a cut line. */
    rows = after - before;
    for (unsigned int i = before; i < after; ++i) {
        if (aotx_test_same(&found[i],
                           "agents: id role state task tool request turn tokens")) {
            matched += 1u;
        }
    }
    snprintf(want, sizeof want, "  0 conductor idle - - - 0 0");
    aotx_test_check(matched == 1u, "the agents command names its columns once");
    aotx_test_check(rows > 1u, "the agents command writes a row under the column names");
    unsigned int right = 0u;
    for (unsigned int i = before; i < after; ++i) {
        if (aotx_test_same(&found[i], want)) {
            right += 1u;
        }
    }
    aotx_test_check(right == 1u, "the row of the first agent holds its own fields");
    if (count > 1u) {
        /* The allowance covers the command record and the console records together. The
         * last record of the line states the cut, and the allowance keeps its sequence. */
        aotx_test_check(rows == (unsigned int)AOTX_CLI_RECORDS_EACH - 1u,
                        "a table longer than the allowance is cut at the allowance");
        aotx_test_check(after > before && found[after - 1u].length > 14u
                        && memcmp(found[after - 1u].body, "output cut at ", 14u) == 0,
                        "the last record of the cut list says that the output was cut");
    }
    printf("cli: the agents command at %u agents wrote %u console lines\n", count, rows);

    aotx_test_one("agents");
    aotx_test_agents_clear<<<1, AOTX_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    before = aotx_test_records(AOTX_REC_CONSOLE, found, AOTX_TEST_FOUND);
    aotx_test_one("agents");
    after = aotx_test_records(AOTX_REC_CONSOLE, found, AOTX_TEST_FOUND);
    right = 0u;
    for (unsigned int i = before; i < after; ++i) {
        if (aotx_test_same(&found[i], "  no agents")) {
            right += 1u;
        }
    }
    aotx_test_check(right == 1u, "an empty agent table gives one line that says so");
    free(found);
}

#endif
