// Purpose: Tail token and page statistics as typed replica records.
// Owns: Two bounded cursors and one latest page row per agent and page.
// Launch shape: One interface thread reads new complete lines in file order.
// Lifetime: Rows reset when the selected boot changes or a file rotates.
#include "replica/stats.hpp"

#include "replica/schema.hpp"

#include <sys/stat.h>

#include <fstream>
#include <map>

namespace aotx::ctrl::replica::stats {
namespace {

constexpr std::size_t line_bound = 512u;

struct Cursor {
    std::uintmax_t offset = 0u;
    dev_t device = 0;
    ino_t inode = 0;
    std::string partial;
    std::uint64_t line = 0u;
    bool dropping = false;
};

template <class Take, class Reset>
void tail(const std::filesystem::path &path, Cursor &cursor, const char *name,
          std::vector<std::string> &results, Take take, Reset reset)
{
    struct stat info{};
    if (::stat(path.c_str(), &info) != 0 || info.st_size < 0) return;
    if (cursor.device != 0 && (cursor.device != info.st_dev || cursor.inode != info.st_ino ||
        static_cast<std::uintmax_t>(info.st_size) < cursor.offset)) {
        cursor = Cursor{};
        reset();
    }
    cursor.device = info.st_dev;
    cursor.inode = info.st_ino;
    if (static_cast<std::uintmax_t>(info.st_size) == cursor.offset) return;
    std::ifstream file(path, std::ios::binary);
    if (!file) return;
    file.seekg(static_cast<std::streamoff>(cursor.offset));
    char byte = '\0';
    while (file.get(byte)) {
        ++cursor.offset;
        if (byte == '\n') {
            ++cursor.line;
            if (!cursor.dropping && !take(cursor.partial)) {
                results.push_back("The " + std::string(name) + " line " +
                                  std::to_string(cursor.line) + " was refused.");
            }
            cursor.partial.clear();
            cursor.dropping = false;
        } else if (!cursor.dropping && cursor.partial.size() < line_bound) {
            cursor.partial.push_back(byte);
        } else if (!cursor.dropping) {
            cursor.partial.clear();
            cursor.dropping = true;
            results.push_back("The " + std::string(name) + " line " +
                              std::to_string(cursor.line + 1u) + " was refused.");
        }
    }
}

} // namespace

struct Reader::Impl {
    std::filesystem::path boot;
    Cursor token_cursor;
    Cursor page_cursor;
    std::vector<TokenStat> tokens;
    std::vector<PageStat> pages;
    std::uint64_t page_tick = 0u;
    double rate = 0.0;
    double rate_time = 0.0;
    std::size_t rate_count = 0u;

    void reset()
    {
        token_cursor = Cursor{};
        page_cursor = Cursor{};
        tokens.clear();
        pages.clear();
        page_tick = 0u;
        rate = 0.0;
        rate_time = 0.0;
        rate_count = 0u;
    }
};

Reader::Reader() : impl_(std::make_unique<Impl>()) {}
Reader::~Reader() = default;

void Reader::read(const std::filesystem::path &boot, double now,
                  std::vector<std::string> &results)
{
    if (boot != impl_->boot) {
        impl_->boot = boot;
        impl_->reset();
    }
    if (boot.empty()) return;
    tail(boot / "tokens.jsonl", impl_->token_cursor, "token statistics", results,
         [this](const std::string &line) {
             TokenStat row;
             if (!schema::token_stat(line, row)) return false;
             impl_->tokens.push_back(row);
             return true;
         }, [this] {
             impl_->tokens.clear();
             impl_->rate = 0.0;
             impl_->rate_time = 0.0;
             impl_->rate_count = 0u;
         });
    tail(boot / "pages.jsonl", impl_->page_cursor, "page statistics", results,
         [this](const std::string &line) {
             PageStat row;
             if (!schema::page_stat(line, row)) return false;
             if (row.tick > impl_->page_tick) {
                 impl_->pages.clear();
                 impl_->page_tick = row.tick;
             }
             for (PageStat &held : impl_->pages) {
                 if (held.agent == row.agent && held.page == row.page) {
                     held = row;
                     return true;
                 }
             }
             impl_->pages.push_back(row);
             return true;
         }, [this] { impl_->pages.clear(); impl_->page_tick = 0u; });
    if (impl_->rate_time == 0.0) {
        impl_->rate_time = now;
        impl_->rate_count = impl_->tokens.size();
    } else if (now > impl_->rate_time && impl_->tokens.size() >= impl_->rate_count &&
               impl_->tokens.size() != impl_->rate_count) {
        impl_->rate = static_cast<double>(impl_->tokens.size() - impl_->rate_count) /
                      (now - impl_->rate_time);
        impl_->rate_time = now;
        impl_->rate_count = impl_->tokens.size();
    }
}

const std::vector<TokenStat> &Reader::tokens() const { return impl_->tokens; }
const std::vector<PageStat> &Reader::pages() const { return impl_->pages; }
double Reader::token_rate() const { return impl_->rate; }

} // namespace aotx::ctrl::replica::stats
