// Purpose: Check hostile control inputs and live binding invariants.
// Owns: Temporary journals, sockets, mappings, and child processes.
// Launch shape: One host process runs each bounded fixture in order.
// Lifetime: Every temporary resource ends before the check returns.
#include "client/client.hpp"
#include "chat/persona.hpp"
#include "instances/lifecycle.hpp"
#include "monitor/telemetry.hpp"
#include "process/child.hpp"
#include "replica/replica.hpp"

#include "cuda/ui/mirror.h"

#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <linux/memfd.h>
#include <signal.h>
#include <unistd.h>

#include <array>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace {

int failed = 0;
int applied = 0;

void check(bool value, const char *text)
{
    ++applied;
    if (value) return;
    ++failed;
    std::printf("ctrl fix: %s\n", text);
}

std::filesystem::path temp_root()
{
    std::array<char, 40> pattern{};
    const std::string base = "/tmp/aotx_ctrl_fix_XXXXXX";
    std::copy(base.begin(), base.end(), pattern.begin());
    char *made = mkdtemp(pattern.data());
    return made == nullptr ? std::filesystem::path{} : std::filesystem::path(made);
}

std::string transcript(const std::string &text, unsigned tick = 1u)
{
    return "{\"tick\":" + std::to_string(tick) +
           ",\"kind\":\"part\",\"text\":\"" + text +
           "\",\"request\":0,\"status\":\"open\",\"turn\":1}\n";
}

std::filesystem::path make_journal(const std::string &text)
{
    const std::filesystem::path root = temp_root();
    const std::filesystem::path boot = root / "0000000000000001";
    std::filesystem::create_directories(boot / "transcript");
    std::filesystem::create_directories(root / "bus");
    std::ofstream(boot / "transcript/0.jsonl") << transcript(text);
    std::ofstream(root / "phase") << "closed 1\n";
    std::ofstream(root / "settings") << "journal.dir = " << root.string() << "\n"
                                      << "models.dir = " << (root / "models").string() << "\n";
    return root;
}

void mirror_stride_case()
{
    const int descriptor = memfd_create("aotx-ctrl-fix", MFD_CLOEXEC);
    check(descriptor >= 0 && ftruncate(descriptor, 4096) == 0,
          "the mirror fixture does not open");
    void *mapping = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, descriptor, 0);
    auto *preamble = static_cast<aotx_mirror_preamble *>(mapping);
    preamble->magic = AOTX_MIRROR_MAGIC;
    preamble->layout = AOTX_MIRROR_LAYOUT;
    preamble->slots = AOTX_MIRROR_SLOTS;
    preamble->slot_bytes = 2147483648u;
    msync(mapping, 4096, MS_SYNC);
    aotx::ctrl::monitor::Telemetry telemetry;
    telemetry.tick(descriptor, 1.0);
    check(!telemetry.mirror().available &&
              telemetry.mirror().result == "The mirror preamble was refused.",
          "the overflowing mirror stride was not refused");
    munmap(mapping, 4096);
    close(descriptor);
}

void replica_identity_case()
{
    const std::filesystem::path root = make_journal("old");
    const std::filesystem::path path = root / "0000000000000001/transcript/0.jsonl";
    aotx::ctrl::replica::State state(root, root / "settings");
    check(state.open() && state.agents().size() == 1u &&
              state.agents()[0].transcript[0].text == "old",
          "the initial transcript did not read");
    const std::filesystem::path rotated = root / "rotated.jsonl";
    std::ofstream(rotated) << transcript("new");
    std::filesystem::rename(rotated, path);
    state.tick(1.0);
    check(state.agents()[0].transcript.size() == 1u &&
              state.agents()[0].transcript[0].text == "new",
          "an equal-size rotation kept old transcript data");
    std::ofstream(path, std::ios::trunc) << transcript("x", 2u);
    state.tick(2.0);
    check(state.agents()[0].transcript.size() == 1u &&
              state.agents()[0].transcript[0].text == "x",
          "a truncation kept old transcript data");
    std::filesystem::remove_all(root);
}

void replica_bound_case()
{
    const std::filesystem::path root = make_journal("ok");
    const std::filesystem::path path = root / "0000000000000001/transcript/0.jsonl";
    std::ofstream file(path, std::ios::trunc);
    file << std::string(37377u, 'x') << '\n' << transcript("kept");
    file.close();
    aotx::ctrl::replica::State state(root, root / "settings");
    check(state.open(), "the bounded replica fixture did not open");
    unsigned refusals = 0u;
    for (const std::string &line : state.take_results()) {
        if (line.find("transcript line") != std::string::npos) ++refusals;
    }
    check(refusals == 1u && state.agents().size() == 1u &&
              state.agents()[0].transcript.size() == 1u &&
              state.agents()[0].transcript[0].text == "kept",
          "an oversize replica line was not refused once and skipped");
    std::ofstream(path, std::ios::app) << std::string(37377u, 'y');
    state.tick(1.0);
    unsigned partial_refusals = 0u;
    for (const std::string &line : state.take_results()) {
        if (line.find("transcript line") != std::string::npos) ++partial_refusals;
    }
    std::ofstream(path, std::ios::app) << std::string(4096u, 'z');
    state.tick(2.0);
    check(partial_refusals == 1u && state.take_results().empty(),
          "an oversize partial line did not state exactly one refusal");
    std::ofstream(root / "0000000000000001/transcript/256.jsonl") << transcript("bad id");
    state.tick(3.0);
    check(state.agents().size() == 1u,
          "a transcript agent identifier outside the shipped bound was accepted");
    std::filesystem::remove_all(root);
}

void folding_case()
{
    const std::filesystem::path root = make_journal("first");
    const std::filesystem::path path = root / "0000000000000001/transcript/0.jsonl";
    std::ofstream file(path, std::ios::trunc);
    file << transcript("first")
         << "{\"tick\":2,\"kind\":\"bound\",\"text\":\"\",\"request\":0,"
            "\"status\":\"limit\",\"turn\":1}\n"
         << transcript("second", 3u)
         << "{\"tick\":4,\"kind\":\"reply\",\"text\":\"firstsecond\","
            "\"request\":0,\"status\":\"\",\"turn\":1}\n";
    file.close();
    std::vector<aotx::ctrl::replica::Agent> agents;
    std::string reason;
    check(aotx::ctrl::replica::read_boot_transcripts(
              root / "0000000000000001", agents, reason) && agents.size() == 1u &&
              agents[0].transcript.size() == 2u &&
              agents[0].transcript[0].kind == "reply" &&
              agents[0].transcript[0].text == "firstsecond",
          "interleaved parts duplicated the completed reply");
    std::filesystem::remove_all(root);
}

void missing_descriptor_case()
{
    const std::filesystem::path root = temp_root();
    const std::string path = (root / "aotx.sock").string();
    int ready[2];
    check(pipe(ready) == 0, "the stream fixture pipe did not open");
    std::size_t sent = 0u;
    std::thread server([&] {
        const int listen_fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        std::memcpy(address.sun_path, path.c_str(), path.size() + 1u);
        bind(listen_fd, reinterpret_cast<sockaddr *>(&address), sizeof(address));
        listen(listen_fd, 1);
        const ssize_t notified = write(ready[1], "r", 1);
        (void)notified;
        const int peer = accept4(listen_fd, nullptr, nullptr, SOCK_CLOEXEC);
        std::array<char, 4096> bytes{};
        bytes[0] = 'M';
        while (sent < 32u * 1024u * 1024u) {
            const ssize_t count = send(peer, bytes.data(), bytes.size(), MSG_NOSIGNAL);
            if (count <= 0) break;
            sent += static_cast<std::size_t>(count);
            bytes[0] = 'x';
        }
        close(peer);
        close(listen_fd);
    });
    char byte = '\0';
    check(read(ready[0], &byte, 1) == 1, "the stream fixture did not start");
    close(ready[0]);
    close(ready[1]);
    aotx::ctrl::client::Client client(root);
    for (unsigned index = 0u; index < 100u; ++index) {
        client.tick(static_cast<double>(index) / 100.0);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    bool refused = false;
    for (const std::string &line : client.take_results()) {
        if (line.find("no descriptor") != std::string::npos ||
            line.find("overflow") != std::string::npos) refused = true;
    }
    server.join();
    check(refused && sent < 32u * 1024u * 1024u,
          "the 32 MiB missing-descriptor stream was not bounded");
    std::filesystem::remove_all(root);
}

void binding_and_start_case()
{
    const std::filesystem::path root = temp_root();
    std::filesystem::create_directories(root / "build");
    std::filesystem::create_directories(root / "models");
    aotx::ctrl::instances::Lifecycle lifecycle;
    lifecycle.set_registry(root / "instances.jsonl");
    for (unsigned index = 0u; index < 2u; ++index) {
        aotx::ctrl::instances::Definition definition;
        definition.name = "Instance " + std::to_string(index);
        definition.journal = root / ("journal-" + std::to_string(index));
        definition.settings = root / ("settings-" + std::to_string(index));
        definition.build = root / "build";
        definition.models = root / "models";
        check(lifecycle.create(definition), "an instance binding did not create");
    }
    check(lifecycle.replica(0u) != lifecycle.replica(1u) &&
              lifecycle.client(0u) != lifecycle.client(1u),
          "two instances shared one replica or socket client");
    int ready[2];
    check(pipe(ready) == 0, "the live socket fixture pipe did not open");
    std::thread server([&] {
        const std::string socket_path = (root / "journal-0/aotx.sock").string();
        const int listen_fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        std::memcpy(address.sun_path, socket_path.c_str(), socket_path.size() + 1u);
        bind(listen_fd, reinterpret_cast<sockaddr *>(&address), sizeof(address));
        listen(listen_fd, 1);
        const ssize_t notified = write(ready[1], "r", 1);
        (void)notified;
        const int peer = accept4(listen_fd, nullptr, nullptr, SOCK_CLOEXEC);
        const int descriptor = memfd_create("aotx-ctrl-bind", MFD_CLOEXEC);
        char frame = 'M';
        iovec io{&frame, 1u};
        std::array<char, CMSG_SPACE(sizeof(int))> control{};
        msghdr message{};
        message.msg_iov = &io;
        message.msg_iovlen = 1u;
        message.msg_control = control.data();
        message.msg_controllen = control.size();
        cmsghdr *header = CMSG_FIRSTHDR(&message);
        header->cmsg_level = SOL_SOCKET;
        header->cmsg_type = SCM_RIGHTS;
        header->cmsg_len = CMSG_LEN(sizeof(int));
        std::memcpy(CMSG_DATA(header), &descriptor, sizeof(descriptor));
        sendmsg(peer, &message, MSG_NOSIGNAL);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        close(descriptor);
        close(peer);
        close(listen_fd);
    });
    char ready_byte = '\0';
    check(read(ready[0], &ready_byte, 1) == 1, "the live socket fixture did not start");
    close(ready[0]);
    close(ready[1]);
    for (unsigned index = 0u; index < 20u; ++index) {
        lifecycle.tick(static_cast<double>(index) / 10.0);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    check(!lifecycle.start(0u) && lifecycle.refusal().find("socket answers") != std::string::npos,
          "an answering instance socket accepted a second boot");
    server.join();
    lifecycle.select(1u);
    std::string command;
    check(lifecycle.replica(1u)->create_conversation(command) && command == "spawn worker",
          "conversation creation did not use the device spawn command");
    std::filesystem::create_directories(root / "journal-1/bus");
    std::ofstream(root / "journal-1/bus/2000-01-01-aotx.jsonl")
        << "{\"v\":1,\"run\":\"aotx\",\"agent\":\"agent-1\",\"seq\":1,"
           "\"ts\":\"2000-01-01T00:00:00.000+00:00\",\"type\":\"note\","
           "\"body\":{\"text\":\"agent 1 spawned role 2 parent 0 state 1 turn 0 ticks 1\","
           "\"tick\":1,\"boot\":\"\",\"lag_ms\":null}}\n";
    lifecycle.tick(5.0);
    check(lifecycle.replica(1u)->agents().size() == 1u &&
              lifecycle.replica(1u)->agents()[0].id == 1u,
          "the spawned worker did not become a conversation binding");
    check(lifecycle.name_conversation(1u, 1u, "Named conversation") &&
              lifecycle.replica(1u)->agents()[0].conversation == "Named conversation",
          "the conversation name did not enter the live binding");
    std::ifstream registry(root / "instances.jsonl");
    const std::string registry_text((std::istreambuf_iterator<char>(registry)),
                                    std::istreambuf_iterator<char>());
    check(registry_text.find("\"conversation_1\":\"Named conversation\"") !=
              std::string::npos,
          "the conversation name did not enter the instance registry");
    // A running word with no answering socket and no child is stale. The start passes the
    // phase check and stops at the next one, the absent boot program.
    std::ofstream(root / "journal-1/phase") << "running 1\n";
    check(!lifecycle.start(1u) && lifecycle.refusal().find("aotx_boot is not in the build") != std::string::npos,
          "a stale running phase word refused the start");
    lifecycle.tick(5.5);
    check(lifecycle.instances()[1].state == aotx::ctrl::instances::LiveState::stopped,
          "a stale running phase word was not shown as stopped");
    const bool removed = lifecycle.remove(0u);
    lifecycle.tick(6.0);
    check(removed && lifecycle.instances().size() == 1u,
          "instance removal did not free one binding");
    std::filesystem::remove_all(root);
}

void persona_storage_case()
{
    const std::filesystem::path root = temp_root();
    aotx::ctrl::chat::persona::Store store(root / "personas");
    std::string result;
    std::string voice;
    const std::filesystem::path journal = root / "journal";
    check(store.save_default(journal, "Default voice.", result) &&
              store.default_voice(journal) == "Default voice.",
          "the instance persona did not persist");
    check(store.save_override(journal, 3u, "Other voice.", result) &&
              store.override_voice(journal, 3u, voice) && voice == "Other voice.",
          "the conversation persona did not persist");
    check(store.save_override(journal, 3u, "", result) &&
              !store.override_voice(journal, 3u, voice),
          "the conversation persona did not return to the instance default");
    std::filesystem::remove_all(root);
}

void child_escalation_case()
{
    int ready[2];
    check(pipe(ready) == 0, "the child fixture pipe did not open");
    const pid_t child = fork();
    if (child == 0) {
        signal(SIGTERM, SIG_IGN);
        const ssize_t notified = write(ready[1], "r", 1);
        (void)notified;
        for (;;) pause();
    }
    char byte = '\0';
    check(read(ready[0], &byte, 1) == 1, "the child fixture did not start");
    close(ready[0]);
    close(ready[1]);
    const auto start = std::chrono::steady_clock::now();
    const aotx::ctrl::process::End ended = aotx::ctrl::process::end_child(child);
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    check(ended == aotx::ctrl::process::End::kill && seconds < 1.5,
          "a SIGTERM-ignoring child did not receive bounded SIGKILL escalation");
}

void child_last_line_case()
{
    const std::filesystem::path root = temp_root();
    std::filesystem::create_directories(root / "build");
    std::filesystem::create_directories(root / "models");
    std::filesystem::create_symlink("/bin/echo", root / "build/aotx_boot");
    aotx::ctrl::instances::Definition definition;
    definition.name = "Line fixture";
    definition.journal = root / "journal";
    definition.settings = root / "settings";
    definition.build = root / "build";
    definition.models = root / "models";
    aotx::ctrl::instances::Lifecycle lifecycle;
    check(lifecycle.create(definition) && lifecycle.start(0u),
          "the child line fixture did not start");
    for (unsigned index = 0u; index < 100u; ++index) {
        lifecycle.tick(static_cast<double>(index) / 100.0);
        if (lifecycle.instances()[0].process < 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const std::string result = lifecycle.instances()[0].result;
    check(result.find("child died with status 0: --settings") != std::string::npos,
          "the child result omitted its piped last line");
    std::filesystem::remove_all(root);
}

} // namespace

int main()
{
    mirror_stride_case();
    replica_identity_case();
    replica_bound_case();
    folding_case();
    missing_descriptor_case();
    binding_and_start_case();
    persona_storage_case();
    child_escalation_case();
    child_last_line_case();
    std::printf("ctrl fix: cases applied %d, failed %d\n", applied, failed);
    return failed == 0 ? 0 : 1;
}
