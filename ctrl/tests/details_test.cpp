// Purpose: Check that model details cannot outlive their file or build identity.
// Owns: Temporary model files, one inspector executable, and detail states.
// Threading: One host process checks batches of 1 and 64 distinct model selections.
// Lifetime: Each batch removes its fixture files after all children end.
#include "model/details.hpp"
#include "replica/store.hpp"
#include "inspect_fixture.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <thread>

namespace {
using namespace aotx::ctrl;
int applied = 0, failed = 0;
void check(bool good, const char *reason)
{
    ++applied;
    if (!good) { ++failed; std::printf("model details: %s\n", reason); }
}

bool complete(model::DetailsState &state, const std::filesystem::path &build,
              const std::filesystem::path &root, const std::vector<replica::Model> &rows)
{
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (state.action.running() && std::chrono::steady_clock::now() < until) {
        state.update(build, root, rows);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return state.action.header().has_value();
}

void batch(unsigned count)
{
    char pattern[] = "/tmp/aotx_details_XXXXXX";
    char *made = mkdtemp(pattern);
    check(made != nullptr, "the fixture directory did not open");
    if (!made) return;
    const std::filesystem::path root(made), build = root / "build";
    std::filesystem::create_directory(build);
    const auto program = build / "aotx_models";
    std::filesystem::copy_file("/proc/self/exe", program);
    std::vector<replica::Model> rows;
    for (unsigned index = 0; index < count; ++index) {
        replica::Model row;
        row.name = "model-" + std::to_string(index);
        row.file = row.name + ".gguf";
        row.role = "language";
        row.digest = std::string(64, "0123456789abcdef"[index % 16]);
        const std::string value = std::to_string(index);
        row.bytes = value.size();
        row.on_disk = row.active = true;
        std::ofstream(root / row.file) << value;
        rows.push_back(row);
    }
    for (unsigned index = 0; index < count; ++index) {
        model::DetailsState state;
        state.start(build, root, rows[index]);
        check(complete(state, build, root, rows) &&
                  state.action.header()->architecture == "family_" + std::to_string(index),
              "a selected file did not receive its own header facts");
        rows[index].active = false;
        state.update(build, root, rows);
        check(state.action.header().has_value(), "an activation change discarded unchanged file facts");
        rows[index].digest[0] = rows[index].digest[0] == 'a' ? 'b' : 'a';
        state.update(build, root, rows);
        check(!state.action.header() && !state.result.empty(), "a changed digest kept old facts");

        state.start(build, root, rows[index]);
        check(complete(state, build, root, rows), "the new identity did not inspect");
        const auto file = root / rows[index].file;
        std::filesystem::last_write_time(file, std::filesystem::last_write_time(file) +
                                        std::chrono::seconds(1));
        state.update(build, root, rows);
        check(!state.action.header(), "a changed file stamp kept old facts");

        state.start(build, root, rows[index]);
        check(complete(state, build, root, rows), "the file did not inspect before replacement");
        const auto held_time = std::filesystem::last_write_time(file);
        const auto replacement = root / "replacement";
        std::ofstream(replacement) << std::string(rows[index].bytes, '9');
        std::filesystem::last_write_time(replacement, held_time);
        std::filesystem::rename(replacement, file);
        state.update(build, root, rows);
        check(!state.action.header(), "a replacement with the same size and time kept old facts");

        state.start(build, root, rows[index]);
        check(complete(state, build, root, rows), "the replacement file did not inspect");
        std::ofstream(file) << std::string(rows[index].bytes, '7');
        std::filesystem::last_write_time(file, held_time);
        state.update(build, root, rows);
        check(!state.action.header(), "an in-place write with the same size and time kept old facts");

        state.start(build, root, rows[index]);
        check(complete(state, build, root, rows), "the changed file did not inspect");
        std::filesystem::last_write_time(program, std::filesystem::last_write_time(program) +
                                        std::chrono::seconds(1));
        state.update(build, root, rows);
        check(!state.action.header(), "a changed build kept old facts");

        state.start(build, root, rows[index]);
        check(complete(state, build, root, rows), "the build did not inspect before replacement");
        const auto program_time = std::filesystem::last_write_time(program);
        std::filesystem::copy_file(program, replacement);
        std::filesystem::last_write_time(replacement, program_time);
        std::filesystem::rename(replacement, program);
        state.update(build, root, rows);
        check(!state.action.header(), "a replaced executable with the same size and time kept old facts");

        state.start(build, root, rows[index]);
        check(complete(state, build, root, rows), "the changed build did not inspect");
        auto changed = rows;
        changed.erase(changed.begin() + index);
        state.update(build, root, changed);
        check(!state.action.header(), "an absent selected row kept old facts");

        state.start(build, root, rows[index]);
        check(complete(state, build, root, rows), "the restored row did not inspect");
        std::filesystem::remove(file);
        state.update(build, root, rows);
        check(!state.action.header(), "a removed file kept old facts");
        state.start(build, root, rows[index]);
        check(!state.action.running() && !state.result.empty(), "a missing file started an inspection");
        check(state.source == file && state.build_path == build,
              "a refused inspection displayed an earlier source or build path");
        state.clear();
        check(!state.open && state.selection.empty() && !state.action.header(),
              "a cleared selection kept its window or facts");
    }
    std::filesystem::remove_all(root);
}
} // namespace

int main(int argc, char **argv)
{
    if (argc == 3 && std::string(argv[1]) == "inspect") {
        unsigned id = 0;
        std::ifstream(argv[2]) >> id;
        return aotx::ctrl::test::write_all(1, aotx::ctrl::test::report(argv[2], id, true)) ? 0 : 1;
    }
    batch(1);
    batch(64);
    std::printf("model details: %d cases, %d failed\n", applied, failed);
    return failed ? 1 : 0;
}
