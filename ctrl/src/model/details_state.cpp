// Purpose: Invalidate header facts after file, selection, or build changes.
// Owns: File stamp comparison and inspection lifetime decisions.
// Threading: One interface thread polls the selected inspection.
// Lifetime: The binding lasts until a source changes or the selection clears.
#include "model/details.hpp"
#include "replica/store.hpp"

#include <algorithm>
#include <sys/stat.h>

namespace aotx::ctrl::model {
namespace {

std::string file_stamp(const std::filesystem::path &path)
{
    struct stat info {};
    if (::stat(path.c_str(), &info) != 0 || !S_ISREG(info.st_mode)) return {};
    return path.lexically_normal().string() + "\n" + std::to_string(info.st_dev) + "\n" +
           std::to_string(info.st_ino) + "\n" + std::to_string(info.st_size) + "\n" +
           std::to_string(info.st_mtim.tv_sec) + "\n" + std::to_string(info.st_mtim.tv_nsec) +
           "\n" + std::to_string(info.st_ctim.tv_sec) + "\n" +
           std::to_string(info.st_ctim.tv_nsec);
}

std::string binding(const std::filesystem::path &build, const std::filesystem::path &directory,
                    const replica::Model &row)
{
    const std::string file = file_stamp(directory / row.file);
    const std::string program = file_stamp(build / "aotx_models");
    if (file.empty() || program.empty()) return {};
    return file + "\n" + program;
}

} // namespace

void DetailsState::start(const std::filesystem::path &build,
                         const std::filesystem::path &directory, const replica::Model &row)
{
    clear();
    open = true;
    build_path = build;
    source = directory / row.file;
    selection = replica::store::key(row);
    stamp = binding(build, directory, row);
    if (!row.on_disk || stamp.empty()) {
        result = "Inspection needs the model file and aotx_models in the selected build.";
        return;
    }
    action.start(build, source.string());
}

void DetailsState::update(const std::filesystem::path &build,
                          const std::filesystem::path &directory,
                          const std::vector<replica::Model> &rows)
{
    if (!selection.empty() && !stamp.empty()) {
        const auto row = std::find_if(rows.begin(), rows.end(), [this](const auto &item) {
            return replica::store::key(item) == selection;
        });
        if (row == rows.end() || !row->on_disk || binding(build, directory, *row) != stamp) {
            action.cancel();
            selection.clear();
            stamp.clear();
            result = "The model file, model entry, or build changed. Inspect the file again.";
        }
    }
    action.tick();
}

void DetailsState::clear()
{
    action.cancel();
    build_path.clear();
    source.clear();
    selection.clear();
    stamp.clear();
    result.clear();
    open = false;
}

} // namespace aotx::ctrl::model
