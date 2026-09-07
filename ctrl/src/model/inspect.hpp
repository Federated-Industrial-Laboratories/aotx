// Purpose: Read model header facts through the model store program.
// Owns: One child, its bounded output, and the selected source identity.
// Threading: One interface thread starts and polls each action.
// Lifetime: The action cancels and reaps its child at destruction.
#ifndef AOTX_CTRL_MODEL_INSPECT_HPP
#define AOTX_CTRL_MODEL_INSPECT_HPP

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace aotx::ctrl::model {

enum class InspectStatus { idle, running, complete, unsupported, failed, cancelled };

struct InspectBlockType {
    std::string name;
    std::uint32_t id = 0;
    std::uint64_t count = 0;
    bool supported = false;
};

struct InspectLayerType {
    std::string name;
    std::uint32_t count = 0;
};

struct InspectHeader {
    // Text fields retain the inspector's escaped bytes.
    std::string file, architecture, pre_tokenizer, tokenizer_model, rotary_pairs;
    bool pre_tokenizer_supported = false, tokenizer_model_supported = false;
    bool rotary_pairs_supported = false, layer_sets_supported = false;
    bool build_support = false;
    std::uint64_t tensors = 0, vocabulary = 0, unknown_tensors = 0;
    std::uint32_t layers = 0, hidden = 0, layer_limit = 0;
    std::uint64_t file_bytes = 0, header_bytes = 0, chat_template_bytes = 0;
    std::optional<std::uint64_t> received_bytes;
    std::string chat_template_sha256;
    std::vector<InspectBlockType> block_types;
    std::vector<InspectLayerType> layer_types;
};

struct InspectLimits {
    std::size_t output_bytes = 128u * 1024u;
    std::chrono::milliseconds duration{30000};
};

class InspectAction {
  public:
    explicit InspectAction(InspectLimits limits = {});
    ~InspectAction();
    InspectAction(const InspectAction &) = delete;
    InspectAction &operator=(const InspectAction &) = delete;

    // A replacement clears all old facts before its child starts.
    bool start(const std::filesystem::path &build, const std::string &input);
    void tick();
    void cancel();
    bool running() const;
    InspectStatus status() const;
    const std::filesystem::path &build() const;
    const std::string &input() const;
    const std::string &message() const;
    const std::string &output() const;
    const std::optional<InspectHeader> &header() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace aotx::ctrl::model
#endif
