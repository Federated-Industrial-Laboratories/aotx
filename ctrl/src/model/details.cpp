// Purpose: Display the disk inspector's model header facts.
// Owns: Model details window and bounded report presentation.
// Threading: One interface thread draws one selected inspection.
// Lifetime: The details state owns the result between frames.
#include "model/details.hpp"
#include "imgui.h"

namespace aotx::ctrl::model {
namespace {
const char *support(bool value) { return value ? "yes" : "no"; }

void header(const InspectHeader &row)
{
    ImGui::SeparatorText("Header support");
    ImGui::Text("Selected build: %s", support(row.build_support));
    ImGui::TextWrapped("This checks header fields and tensor sets. Runtime use is not verified.");
    ImGui::TextWrapped("Weight integrity, memory fit, prompt wrap, prefill, and restore need separate checks.");
    ImGui::Text("Architecture: %s", row.architecture.c_str());
    ImGui::Text("Layers: %u  Hidden width: %u  Vocabulary: %llu", row.layers, row.hidden,
                static_cast<unsigned long long>(row.vocabulary));
    for (const auto &layer : row.layer_types)
        ImGui::Text("%s layers: %u", layer.name.c_str(), layer.count);
    ImGui::Text("Tensor sets: %s  Unknown tensors: %llu  Layer limit: %u",
                support(row.layer_sets_supported),
                static_cast<unsigned long long>(row.unknown_tensors), row.layer_limit);
    ImGui::SeparatorText("Token and weight formats");
    ImGui::Text("Pre-tokenizer: %s  Supported: %s", row.pre_tokenizer.c_str(),
                support(row.pre_tokenizer_supported));
    ImGui::Text("Tokenizer model: %s  Supported: %s", row.tokenizer_model.c_str(),
                support(row.tokenizer_model_supported));
    ImGui::Text("Rotary pairs: %s  Supported: %s", row.rotary_pairs.c_str(),
                support(row.rotary_pairs_supported));
    for (const auto &block : row.block_types)
        ImGui::Text("%s (ID %u): %llu tensors  Supported: %s", block.name.c_str(), block.id,
                    static_cast<unsigned long long>(block.count), support(block.supported));
    ImGui::Text("File: %llu bytes  Header: %llu bytes",
                static_cast<unsigned long long>(row.file_bytes),
                static_cast<unsigned long long>(row.header_bytes));
    ImGui::Text("Chat template: %llu bytes",
                static_cast<unsigned long long>(row.chat_template_bytes));
    ImGui::TextWrapped("Template SHA-256: %s", row.chat_template_sha256.c_str());
}
} // namespace

void draw_details(DetailsState &state)
{
    if (!state.open) return;
    ImGui::SetNextWindowSize(ImVec2(640, 640), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Model details", &state.open)) {
        ImGui::TextWrapped("File: %s", state.source.c_str());
        ImGui::TextWrapped("Build: %s", state.build_path.c_str());
        ImGui::TextWrapped("%s", state.result.empty() ? state.action.message().c_str()
                                                    : state.result.c_str());
        if (state.action.running() && ImGui::Button("Cancel")) state.action.cancel();
        if (state.action.header()) header(*state.action.header());
        if (!state.action.output().empty() && ImGui::CollapsingHeader("Inspector output")) {
            ImGui::BeginChild("Output", ImVec2(0, 180), ImGuiChildFlags_Borders,
                              ImGuiWindowFlags_HorizontalScrollbar);
            ImGui::TextUnformatted(state.action.output().c_str());
            ImGui::EndChild();
        }
    }
    ImGui::End();
}
} // namespace aotx::ctrl::model
