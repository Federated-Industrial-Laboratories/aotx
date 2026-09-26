/* Purpose: Check persistent shared admission, retry floors, membership and exact result records.
 * Owns: Distinct mapped N=1 and N=64 table fixtures.
 * Launch shape: Production mailbox batches and bounded record emission on the GPU.
 * Lifetime: Each run retains its record prefix for an independent replay check. */
#include "shared_state_fixture.h"
static void run(unsigned n)
{
    fixture f(n); std::vector<std::vector<unsigned char>> p;
    for (unsigned i = 0; i < n; ++i) p.push_back(command(i+1, AOTX_SHARED_REGISTER, 1));
    f.batch(p, 202); f.batch(p, 200);
    p.clear(); for (unsigned i = 0; i < n; ++i) p.push_back(command(i+1, AOTX_SHARED_SPACE, 2, i+100)); f.batch(p, 202);
    p.clear(); for (unsigned i = 0; i < n; ++i) p.push_back(command(i+1, AOTX_SHARED_CONVERSATION, 3, i+1000, i+100)); f.batch(p, 202);
    p.clear(); for (unsigned i = 0; i < n; ++i) p.push_back(command(i+1, AOTX_SHARED_INPUT, 4, i+1000, 0, "source "+std::to_string(i))); f.batch(p, 202);
    f.batch(p, 200);
    auto changed = p; for (auto &row : changed) row.back() ^= 1; f.batch(changed, 409);
    auto future = p; for (auto &row : future) { put(row, AOTX_SERVICE_HEAD+16, 6, 8); put(row, AOTX_SERVICE_HEAD+32, 6, 8); } f.batch(future, 409);
    auto rows = f.receipts(); unsigned char old_id[16]; memcpy(old_id, rows[0].id, 16);
    std::vector<unsigned> requests;
    for (unsigned i = 0; i < rows.size(); ++i) if (rows[i].operation == AOTX_SHARED_INPUT) {
        const auto &r = rows[i]; requests.push_back(i);
        check(r.phase == AOTX_SHARED_QUEUED && r.slot == AOTX_SLOTS && r.order == 1, "conversation admission does not lease a slot");
        check(r.admission_source && !r.saved_admission && r.model_digest[0] == 99, "exact admission is committed before it is saved");
        check(r.pages == 64 && r.sample.top_p == 1 && r.sample.repeat_penalty == 1, "server-derived input limits survive record publication");
    }
    check(requests.size() == n, "every distinct input owns one receipt");
    std::vector<aotx_shared_conversation> conversations(f.shared.conversation_capacity);
    cu(cudaMemcpy(conversations.data(), f.shared.conversations, conversations.size()*sizeof(conversations[0]), cudaMemcpyDeviceToHost));
    for (unsigned i = 0; i < n; ++i) {
        check(conversations[i].active && conversations[i].next_order == 2 && conversations[i].binding.principal[0] == (unsigned char)(i+100),
            "private memory owner is the space identity");
        auto read = read_frame(i == n-1 ? n+1 : i+2, AOTX_SHARED_CONVERSATION_READ, i+1000);
        f.batch({read}, 404);
        auto affect = read_frame(i == n-1 ? n+1 : i+2, AOTX_SHARED_AFFECT_READ, i+1000);
        f.batch({affect}, 404);
        affect = read_frame(i+1, AOTX_SHARED_AFFECT_READ, i+1000);
#ifdef AOTX_AFFECT
        f.batch({affect}, 200);
        const unsigned char *value = f.mailbox[1].bytes + AOTX_SERVICE_HEAD + AOTX_SHARED_REPLY_HEAD;
        check(aotx_service_get(value, 4) == 1 && !aotx_service_get(value+4, 4) &&
            !aotx_service_get(value+8, 8) && !aotx_service_get(value+48, 4), "owned scope exposes disabled state and unavailable probes");
#else
        f.batch({affect}, 501);
#endif
    }
    cu(cudaMemcpyFromSymbol(&f.seam, aotx_seam, sizeof(f.seam)));
    unsigned admission_records = (unsigned)f.seam.dev.tail;
    unsigned char *prefix; cu(cudaMalloc(&prefix, (size_t)admission_records*AOTX_SLOT_BYTES));
    cu(cudaMemcpy(prefix, f.seam.dev.base, (size_t)admission_records*AOTX_SLOT_BYTES, cudaMemcpyDeviceToDevice));
    aotx_shared_test_ack<<<1,1>>>(); cu(cudaDeviceSynchronize());
    for (unsigned index : requests) {
        aotx_shared_test_result<<<1,1>>>(index, 0, f.result); cu(cudaDeviceSynchronize()); check(f.value() == 1, "exact result bytes enter a bounded output transfer");
        aotx_shared_emit<<<1,1>>>(); cu(cudaDeviceSynchronize());
        aotx_shared_test_result<<<1,1>>>(index, 1, f.result); cu(cudaDeviceSynchronize()); check(f.value() == 1, "terminal usage enters a complete result transfer");
        aotx_shared_emit<<<1,1>>>(); cu(cudaDeviceSynchronize());
    }
    rows = f.receipts();
    for (unsigned index : requests) {
        const auto &r = rows[index];
        check(r.phase == AOTX_SHARED_DONE && r.output == 8 && r.result[1] == r.actor[0] && r.result[2] == 0xc3 && r.result[3] == 0xa9,
            "recorded output preserves distinct bytes and UTF-8 offsets");
        check(r.terminal_source > r.admission_source && !r.saved_terminal && r.prompt == r.actor[0]+8 && r.sampled == 3,
            "terminal status differs from saved terminal status");
    }
    aotx_shared_test_ack<<<1,1>>>(); cu(cudaDeviceSynchronize());
    p.clear(); for (unsigned i = 0; i < n; ++i) { auto r = command(i+1, AOTX_SHARED_RETIRE, 5); put(r, AOTX_SERVICE_HEAD+128, 4, 8); p.push_back(r); } f.batch(p, 202);
    p.clear(); for (unsigned i = 0; i < n; ++i) p.push_back(command(i+1, AOTX_SHARED_REGISTER, 1)); f.batch(p, 410);
    f.batch({command(n+1, AOTX_SHARED_REGISTER, 1)}, 202);
    auto room = command(1, AOTX_SHARED_SPACE, 6, 9000); put(room, AOTX_SERVICE_HEAD+12, 1); f.batch({room}, 202);
    f.batch({read_frame(n+1, AOTX_SHARED_SPACE_READ, 9000)}, 404);
    auto join = command(1, AOTX_SHARED_MEMBER, 7, 9000); put(join, AOTX_SERVICE_HEAD+88, n+1); put(join, AOTX_SERVICE_HEAD+116, 3); f.batch({join}, 202);
    f.batch({read_frame(n+1, AOTX_SHARED_SPACE_READ, 9000)}, 200);
    auto revoke = command(1, AOTX_SHARED_MEMBER, 8, 9000); put(revoke, AOTX_SERVICE_HEAD+88, n+1); f.batch({revoke}, 202);
    f.batch({read_frame(n+1, AOTX_SHARED_SPACE_READ, 9000)}, 404);
    auto instance = command(1, AOTX_SHARED_SPACE, 9, 9001); put(instance, AOTX_SERVICE_HEAD+12, 2); f.batch({instance}, 202);
    f.batch({read_frame(n+1, AOTX_SHARED_SPACE_READ, 9001)}, 200);
    auto reused = command(1, AOTX_SHARED_SAVE, 10); put(reused, AOTX_SERVICE_HEAD+32, 1, 8); f.batch({reused}, 202);
    auto old_read = read_frame(1, AOTX_SHARED_OPERATION_READ);
    memcpy(old_read.data()+AOTX_SERVICE_HEAD+32, old_id, 16); f.batch({old_read}, 404);
    rows = f.receipts();
    for (const auto &r : rows) if (r.phase && r.sequence == 10 && r.actor[0] == 1)
        check(memcmp(r.id, old_id, 16) != 0, "a retired key cannot alias a newer receipt handle");
    cu(cudaMemset(f.shared.participants, 0, f.shared.participant_capacity*sizeof(*f.shared.participants)));
    cu(cudaMemset(f.shared.spaces, 0, f.shared.space_capacity*sizeof(*f.shared.spaces)));
    cu(cudaMemset(f.shared.members, 0, f.shared.member_capacity*sizeof(*f.shared.members)));
    cu(cudaMemset(f.shared.conversations, 0, f.shared.conversation_capacity*sizeof(*f.shared.conversations)));
    cu(cudaMemset(f.shared.receipts, 0, f.shared.receipt_capacity*sizeof(*f.shared.receipts)));
    cu(cudaMemcpyToSymbol(aotx_shared, &f.shared, sizeof(f.shared)));
    aotx_shared_test_replay<<<1,1>>>(prefix, admission_records, f.result, 1); cu(cudaDeviceSynchronize());
    check(f.value() == 0, "malformed canonical replay bytes fail before participant mutation");
    aotx_shared_participant first; cu(cudaMemcpy(&first, f.shared.participants, sizeof(first), cudaMemcpyDeviceToHost));
    check(!first.active, "malformed replay does not publish a partial participant");
    cu(cudaMemcpyToSymbol(aotx_shared, &f.shared, sizeof(f.shared)));
    aotx_shared_test_replay<<<1,1>>>(prefix, admission_records, f.result); cu(cudaDeviceSynchronize());
    check(f.value() == 1, "complete admission prefix replays without current input execution");
    rows = f.receipts(); unsigned interrupted = 0;
    for (const auto &r : rows) if (r.operation == AOTX_SHARED_INPUT) {
        ++interrupted; check(r.phase == AOTX_SHARED_INTERRUPTED && r.gap && r.saved_admission && !r.saved_terminal,
            "saved admission without a result becomes an interrupted receipt");
    }
    check(interrupted == n, "every unfinished saved input is recovered exactly once");
    p.clear(); for (unsigned i = 0; i < n; ++i) p.push_back(command(i+1, AOTX_SHARED_INPUT, 4, i+1000, 0, "source "+std::to_string(i))); f.batch(p, 200);
    aotx_shared_state actual; cu(cudaMemcpyFromSymbol(&actual, aotx_shared, sizeof(actual)));
    check(!actual.fatal, "bounded shared state remains valid"); cudaFree(prefix);
    printf("shared-state N=%u checks=%u failures=%u\n", n, checks, failures);
}

#include "shared_selection.h"
int main(int argc, char **argv)
{
    if (argc > 2 || (argc == 2 && strcmp(argv[1], "--selection-only"))) return 2;
    if (argc == 1) { run(1); run(64); }
    shared_selection(1); shared_selection(64); printf("shared-state total checks=%u failures=%u\n", checks, failures); return failures ? 1 : 0;
}
