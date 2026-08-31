// Purpose: Read typed mirror snapshots and card memory without a CUDA link.
// Owns: One read-only mirror mapping and one optional NVML session.
// Launch shape: One interface thread samples bounded structures.
// Lifetime: Mappings and the library close with the reader.
#include "monitor/telemetry.hpp"

#include "cuda/ui/mirror.h"

#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <utility>

namespace aotx::ctrl::monitor {
namespace {

using Device = void *;
struct Memory {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
};

} // namespace

struct Telemetry::Impl {
    void *mapping = MAP_FAILED;
    std::size_t mapping_bytes = 0u;
    dev_t device = 0;
    ino_t inode = 0;
    MirrorSample mirror;
    std::uint64_t last_tick = 0u;
    double last_time = 0.0;

    void *nvml = nullptr;
    int (*init)() = nullptr;
    int (*shutdown)() = nullptr;
    int (*count)(unsigned *) = nullptr;
    int (*handle)(unsigned, Device *) = nullptr;
    int (*memory)(Device, Memory *) = nullptr;
    int (*name)(Device, char *, unsigned) = nullptr;
    std::vector<CardMemory> cards;
    std::string card_result;
    double next_card_sample = 0.0;

    Impl()
    {
        nvml = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
        if (nvml == nullptr) {
            card_result = "Card memory is absent because libnvidia-ml.so.1 did not load.";
            return;
        }
        init = reinterpret_cast<int (*)()>(dlsym(nvml, "nvmlInit_v2"));
        shutdown = reinterpret_cast<int (*)()>(dlsym(nvml, "nvmlShutdown"));
        count = reinterpret_cast<int (*)(unsigned *)>(dlsym(nvml, "nvmlDeviceGetCount_v2"));
        handle = reinterpret_cast<int (*)(unsigned, Device *)>(
            dlsym(nvml, "nvmlDeviceGetHandleByIndex_v2"));
        memory = reinterpret_cast<int (*)(Device, Memory *)>(
            dlsym(nvml, "nvmlDeviceGetMemoryInfo"));
        name = reinterpret_cast<int (*)(Device, char *, unsigned)>(
            dlsym(nvml, "nvmlDeviceGetName"));
        if (init == nullptr || shutdown == nullptr || count == nullptr || handle == nullptr ||
            memory == nullptr || name == nullptr || init() != 0) {
            card_result = "Card memory is absent because NVML did not start.";
            dlclose(nvml);
            nvml = nullptr;
            return;
        }
        card_result = "Card memory is available through NVML.";
    }

    ~Impl()
    {
        if (mapping != MAP_FAILED) munmap(mapping, mapping_bytes);
        if (nvml != nullptr) {
            shutdown();
            dlclose(nvml);
        }
    }

    void close_mapping()
    {
        if (mapping != MAP_FAILED) munmap(mapping, mapping_bytes);
        mapping = MAP_FAILED;
        mapping_bytes = 0u;
        device = 0;
        inode = 0;
        mirror.available = false;
    }

    bool map_descriptor(int descriptor)
    {
        struct stat info{};
        if (descriptor < 0 || fstat(descriptor, &info) != 0 || info.st_size <= 0) {
            close_mapping();
            mirror.result = "The mirror is not attached.";
            return false;
        }
        if (mapping != MAP_FAILED && info.st_dev == device && info.st_ino == inode) return true;
        close_mapping();
        mapping_bytes = static_cast<std::size_t>(info.st_size);
        mapping = mmap(nullptr, mapping_bytes, PROT_READ, MAP_SHARED, descriptor, 0);
        if (mapping == MAP_FAILED) {
            mapping_bytes = 0u;
            mirror.result = "The mirror descriptor does not map.";
            return false;
        }
        device = info.st_dev;
        inode = info.st_ino;
        return true;
    }

    void read_mirror(int descriptor, double now)
    {
        if (!map_descriptor(descriptor)) return;
        if (mapping_bytes < sizeof(aotx_mirror_preamble)) {
            mirror.result = "The mirror preamble is too short.";
            return;
        }
        const auto *preamble = static_cast<const aotx_mirror_preamble *>(mapping);
        const std::size_t slots = preamble->slots;
        const std::size_t stride = preamble->slot_bytes;
        const bool product_overflows = stride != 0u &&
            slots > (std::numeric_limits<std::size_t>::max() - sizeof(*preamble)) / stride;
        const std::size_t required = product_overflows ? 0u : sizeof(*preamble) + slots * stride;
        if (preamble->magic != AOTX_MIRROR_MAGIC || preamble->layout != AOTX_MIRROR_LAYOUT ||
            preamble->slots != AOTX_MIRROR_SLOTS ||
            preamble->slot_bytes < sizeof(aotx_mirror_snapshot) || product_overflows ||
            required > mapping_bytes) {
            mirror.available = false;
            mirror.result = "The mirror preamble was refused.";
            return;
        }
        aotx_mirror_head newest{};
        for (std::uint32_t index = 0u; index < preamble->slots; ++index) {
            const auto *slot = reinterpret_cast<const aotx_mirror_snapshot *>(
                static_cast<const unsigned char *>(mapping) + sizeof(*preamble) +
                static_cast<std::size_t>(index) * preamble->slot_bytes);
            const std::uint64_t before = __atomic_load_n(&slot->head.sequence, __ATOMIC_ACQUIRE);
            if (before == 0u) continue;
            aotx_mirror_head copy{};
            std::memcpy(&copy, &slot->head, sizeof(copy));
            const std::uint64_t after = __atomic_load_n(&slot->head.sequence, __ATOMIC_ACQUIRE);
            if (before == after && copy.sequence == before && before > newest.sequence) newest = copy;
        }
        if (newest.sequence == 0u) {
            mirror.result = "The mirror has no complete snapshot.";
            return;
        }
        if (last_time != 0.0 && now > last_time && newest.tick >= last_tick) {
            mirror.tick_rate = static_cast<double>(newest.tick - last_tick) / (now - last_time);
        }
        if (newest.tick != last_tick) {
            last_tick = newest.tick;
            last_time = now;
        }
        mirror.sequence = newest.sequence;
        mirror.tick = newest.tick;
        mirror.available = true;
        mirror.result = "The mirror snapshot is current.";
    }

    void read_cards(double now)
    {
        if (nvml == nullptr || now < next_card_sample) return;
        next_card_sample = now + 1.0;
        unsigned held = 0u;
        if (count(&held) != 0) {
            cards.clear();
            card_result = "Card memory is absent because NVML did not list the cards.";
            return;
        }
        std::vector<CardMemory> found;
        for (unsigned index = 0u; index < held; ++index) {
            Device card = nullptr;
            Memory bytes{};
            std::array<char, 96> card_name{};
            if (handle(index, &card) != 0 || memory(card, &bytes) != 0 ||
                name(card, card_name.data(), card_name.size()) != 0) continue;
            found.push_back({index, card_name.data(), bytes.used / (1024u * 1024u),
                             bytes.total / (1024u * 1024u)});
        }
        cards = std::move(found);
        card_result = cards.empty() ? "Card memory is absent because NVML returned no card."
                                    : "Card memory is available through NVML.";
    }
};

Telemetry::Telemetry() : impl_(std::make_unique<Impl>()) {}
Telemetry::~Telemetry() = default;
void Telemetry::tick(int mirror_descriptor, double now)
{
    impl_->read_mirror(mirror_descriptor, now);
    impl_->read_cards(now);
}
const MirrorSample &Telemetry::mirror() const { return impl_->mirror; }
const std::vector<CardMemory> &Telemetry::cards() const { return impl_->cards; }
const std::string &Telemetry::card_result() const { return impl_->card_result; }

} // namespace aotx::ctrl::monitor
