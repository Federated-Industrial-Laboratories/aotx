// Purpose: Present the bounded reply formatting document.
// Owns: Temporary ImGui colors and child regions for code.
// Launch shape: One interface thread draws one parsed document.
// Lifetime: No presentation state remains after the call returns.
#include "chat/render.hpp"

#include "chat/format.hpp"
#include "imgui.h"

#include <algorithm>

namespace aotx::ctrl::chat {
namespace {

void span(const format::Span &item, bool first)
{
    if (!first) ImGui::SameLine(0.0f, 0.0f);
    if (item.kind == format::SpanKind::bold) {
        const ImVec4 color = ImGui::GetStyleColorVec4(ImGuiCol_Text);
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(
            std::min(1.0f, color.x * 1.25f), std::min(1.0f, color.y * 1.25f),
            std::min(1.0f, color.z * 1.25f), color.w));
        ImGui::TextUnformatted(item.text.c_str());
        ImGui::PopStyleColor();
    } else if (item.kind == format::SpanKind::italic) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
        ImGui::TextUnformatted(item.text.c_str());
        ImGui::PopStyleColor();
    } else if (item.kind == format::SpanKind::code) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.45f, 0.85f, 0.95f, 1.0f));
        ImGui::TextUnformatted(item.text.c_str());
        ImGui::PopStyleColor();
    } else {
        ImGui::TextUnformatted(item.text.c_str());
    }
}

void spans(const std::vector<format::Span> &items)
{
    for (std::size_t index = 0u; index < items.size(); ++index) span(items[index], index == 0u);
}

void code_block(const format::Block &block)
{
    if (ImGui::Button("Copy")) ImGui::SetClipboardText(block.text.c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("Copy this code block.");
    const std::size_t lines = 1u + static_cast<std::size_t>(
        std::count(block.text.begin(), block.text.end(), '\n'));
    const float height = ImGui::GetTextLineHeightWithSpacing() *
                         static_cast<float>(std::min<std::size_t>(lines, 14u) + 1u);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.025f, 0.030f, 0.038f, 1.0f));
    ImGui::BeginChild("Code", ImVec2(0.0f, height), ImGuiChildFlags_Borders,
                      ImGuiWindowFlags_HorizontalScrollbar);
    ImGui::TextUnformatted(block.text.c_str());
    ImGui::EndChild();
    ImGui::PopStyleColor();
}

} // namespace

void draw_formatted(const std::string &reply)
{
    const format::Document document = format::render(reply);
    for (std::size_t index = 0u; index < document.size(); ++index) {
        const format::Block &block = document[index];
        ImGui::PushID(static_cast<int>(index));
        if (block.kind == format::BlockKind::code) {
            code_block(block);
        } else if (block.kind == format::BlockKind::heading) {
            ImGui::SetWindowFontScale(block.level == 1u ? 1.35f : 1.18f);
            spans(block.spans);
            ImGui::SetWindowFontScale(1.0f);
        } else if (block.kind == format::BlockKind::bullet) {
            ImGui::Bullet();
            ImGui::SameLine();
            spans(block.spans);
        } else if (block.kind == format::BlockKind::number) {
            ImGui::Text("%u.", block.level);
            ImGui::SameLine();
            spans(block.spans);
        } else {
            spans(block.spans);
        }
        ImGui::PopID();
    }
}

} // namespace aotx::ctrl::chat
