// Purpose: Tail statistics and optional measurement records.
// Owns: Bounded cursors, statistics rows, and measurement rings.
// Launch shape: One interface thread reads new complete lines in file order.
// Lifetime: Rows reset when the selected boot changes or a file rotates.
#include "replica/stats.hpp"

#include "replica/schema.hpp"

#include <sys/stat.h>

#include <algorithm>
#include <fstream>
#include <map>
#ifdef AOTX_AFFECT
#include <utility>
#endif

namespace aotx::ctrl::replica::stats {
namespace {

constexpr std::size_t statistics_line_bound = 512u;
#ifdef AOTX_AFFECT
constexpr std::size_t affect_line_bound = 1024u;
constexpr std::size_t calibration_line_bound = 4096u;
#endif

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
          std::size_t bound, std::vector<std::string> &results, Take take, Reset reset)
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
        } else if (!cursor.dropping && cursor.partial.size() < bound) {
            cursor.partial.push_back(byte);
        } else if (!cursor.dropping) {
            cursor.partial.clear();
            cursor.dropping = true;
            results.push_back("The " + std::string(name) + " line " +
                              std::to_string(cursor.line + 1u) + " was refused.");
        }
    }
}

#ifdef AOTX_AFFECT
template <class Row>
void keep_turn(std::vector<Row> &rows, Row made)
{
    constexpr std::size_t held_turns = 32u;
    std::size_t count = 0u;
    auto oldest = rows.end();
    for (auto row = rows.begin(); row != rows.end(); ++row) {
        if (row->agent != made.agent) continue;
        if (oldest == rows.end()) oldest = row;
        ++count;
    }
    if (count == held_turns) rows.erase(oldest);
    rows.push_back(std::move(made));
}
#endif

} // namespace

struct Reader::Impl {
    std::filesystem::path boot;
    std::filesystem::path store;
    Cursor token_cursor;
    Cursor page_cursor;
    std::vector<TokenStat> tokens;
    std::vector<PageStat> pages;
    std::uint64_t page_tick = 0u;
    double rate = 0.0;
    double rate_time = 0.0;
    std::size_t rate_count = 0u;
#ifdef AOTX_AFFECT
    Cursor affect_cursor;
    Cursor quality_cursor;
    Cursor calibration_cursor;
    Cursor probe_cursor;
    std::vector<AffectTrace> affect;
    std::vector<QualityLine> quality;
    std::optional<Calibration> calibration;
    std::vector<ProbeAccuracy> probes;
#endif

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
#ifdef AOTX_AFFECT
        affect_cursor = Cursor{};
        quality_cursor = Cursor{};
        affect.clear();
        quality.clear();
#endif
    }
};

Reader::Reader() : impl_(std::make_unique<Impl>()) {}
Reader::~Reader() = default;

void Reader::read(const std::filesystem::path &boot, const std::filesystem::path &store,
                  double now,
                  std::vector<std::string> &results)
{
#ifndef AOTX_AFFECT
    (void)store;
#endif
    if (boot != impl_->boot) {
        impl_->boot = boot;
        impl_->reset();
    }
#ifdef AOTX_AFFECT
    if (store != impl_->store) {
        impl_->store = store;
        impl_->calibration_cursor = Cursor{};
        impl_->probe_cursor = Cursor{};
        impl_->calibration.reset();
        impl_->probes.clear();
    }
    tail(store / "affect/calibration.jsonl", impl_->calibration_cursor, "calibration",
         calibration_line_bound, results,
         [this](const std::string &line) {
             Calibration made;
             impl_->calibration.reset();
             if (!schema::calibration(line, made)) return false;
             impl_->calibration = std::move(made);
             return true;
         }, [this] { impl_->calibration.reset(); });
    tail(store / "probes.jsonl", impl_->probe_cursor, "probe catalog",
         statistics_line_bound, results,
         [this](const std::string &line) {
             ProbeAccuracy made;
             if (!schema::probe_accuracy(line, made)) return false;
             const auto same = [&made](const ProbeAccuracy &held) {
                 return held.name == made.name || held.axis == made.axis;
             };
             const auto found = std::find_if(impl_->probes.begin(), impl_->probes.end(), same);
             if (found == impl_->probes.end()) impl_->probes.push_back(std::move(made));
             else *found = std::move(made);
             return true;
         }, [this] { impl_->probes.clear(); });
#endif
    if (boot.empty()) return;
    tail(boot / "tokens.jsonl", impl_->token_cursor, "token statistics",
         statistics_line_bound, results,
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
    tail(boot / "pages.jsonl", impl_->page_cursor, "page statistics",
         statistics_line_bound, results,
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
#ifdef AOTX_AFFECT
    tail(boot / "affect.jsonl", impl_->affect_cursor, "affect trace",
         affect_line_bound, results,
         [this](const std::string &line) {
             AffectTrace row;
             if (!schema::affect_trace(line, row)) return false;
             if (row.trace) keep_turn(impl_->affect, std::move(row));
             return true;
         }, [this] { impl_->affect.clear(); });
    tail(boot / "quality.jsonl", impl_->quality_cursor, "quality",
         statistics_line_bound, results,
         [this](const std::string &line) {
             QualityLine row;
             if (!schema::quality_line(line, row)) return false;
             keep_turn(impl_->quality, std::move(row));
             return true;
         }, [this] { impl_->quality.clear(); });
#endif
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
#ifdef AOTX_AFFECT
const std::vector<AffectTrace> &Reader::affect_traces() const { return impl_->affect; }
const std::vector<QualityLine> &Reader::quality_lines() const { return impl_->quality; }
const std::optional<Calibration> &Reader::calibration() const { return impl_->calibration; }
const std::vector<ProbeAccuracy> &Reader::probe_accuracies() const { return impl_->probes; }
#endif
double Reader::token_rate() const { return impl_->rate; }

} // namespace aotx::ctrl::replica::stats
