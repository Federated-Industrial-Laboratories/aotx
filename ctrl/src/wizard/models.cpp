// Purpose: Derive startup roles from the exact selected store row.
// Owns: Selection refresh rules and active role validation.
// Threading: One host caller checks complete model row sets.
// Lifetime: No model file or process is owned by these checks.
#include "wizard/models.hpp"
#include "replica/store.hpp"

#include <utility>

namespace aotx::ctrl::wizard {

bool language_role(const std::string &role)
{
    return role == "language" || role == "language-q4";
}

ModelSelection::ModelSelection(std::string preferred_role)
    : preferred_role_(std::move(preferred_role)) {}

void ModelSelection::refresh(bool readable, std::vector<replica::Model> rows, std::string reason)
{
    readable_ = readable;
    read_reason_ = std::move(reason);
    rows_.clear();
    if (!readable) return;
    rows_ = std::move(rows);
    if (!identity_.empty()) return;
    unsigned best = 6;
    for (const auto &row : rows_) {
        if (!language_role(row.role)) continue;
        const unsigned rank = (row.active && row.on_disk ? 0u : row.on_disk ? 2u : 4u) +
                              (row.role == preferred_role_ ? 0u : 1u);
        if (rank >= best) continue;
        best = rank;
        identity_ = replica::store::key(row);
    }
}

bool ModelSelection::select(const std::string &identity)
{
    identity_ = identity;
    return selected() != nullptr;
}

const replica::Model *ModelSelection::selected() const
{
    if (!readable_ || identity_.empty()) return nullptr;
    const replica::Model *found = nullptr;
    for (const auto &row : rows_) if (replica::store::key(row) == identity_) {
        if (found != nullptr || !language_role(row.role)) return nullptr;
        found = &row;
    }
    return found;
}

bool ModelSelection::request(const std::string &name_or_identity, const std::string &role,
                             std::string &reason)
{
    reason.clear();
    if (!role.empty() && !language_role(role)) {
        reason = "The selected role must be language or language-q4.";
        return false;
    }
    if (!readable_) { reason = refusal(); return false; }
    if (name_or_identity.empty() && role.empty()) {
        if (selected() != nullptr) return true;
        reason = refusal();
        return false;
    }
    const replica::Model *found = nullptr;
    for (const auto &row : rows_) {
        if (!language_role(row.role) || (!role.empty() && row.role != role)) continue;
        if (!name_or_identity.empty()) {
            if (row.name != name_or_identity && replica::store::key(row) != name_or_identity) continue;
        } else if (!row.active || !row.on_disk) continue;
        if (found != nullptr) {
            reason = "More than one model matches the requested name or role. Select an exact model identity.";
            return false;
        }
        found = &row;
    }
    if (found == nullptr) {
        reason = "The requested language model is not in the readable store.";
        return false;
    }
    return select(replica::store::key(*found));
}

const std::vector<replica::Model> &ModelSelection::rows() const { return rows_; }
const std::string &ModelSelection::identity() const { return identity_; }
bool ModelSelection::readable() const { return readable_; }

std::string ModelSelection::refusal() const
{
    if (!readable_) return read_reason_.empty() ? "The model store does not read." : read_reason_;
    if (identity_.empty()) return "The store lists no language model.";
    if (selected() == nullptr)
        return "The selected model identity is absent or repeated. Select a model again.";
    return {};
}

bool can_fetch(const replica::Model *model)
{
    return model != nullptr && model->catalogued && !model->on_disk;
}

bool can_activate(const replica::Model *model)
{
    return model != nullptr && model->catalogued && model->on_disk && !model->active;
}

ModelRoles model_roles(const ModelSelection &selection)
{
    ModelRoles result;
    const replica::Model *selected = selection.selected();
    if (selected == nullptr) { result.refusal = selection.refusal(); return result; }
    if (!selected->on_disk) {
        result.refusal = "The selected language model file is not on disk.";
        return result;
    }
    if (!selected->active) {
        result.refusal = "The selected language model is not active in the manifest.";
        return result;
    }
    for (const std::string &role : {std::string("embedding"), std::string("reranker"), selected->role}) {
        const replica::Model *active = nullptr;
        for (const auto &row : selection.rows()) if (row.active && row.role == role) {
            if (active != nullptr) {
                result.roles.clear();
                result.embedding_available = false;
                result.refusal = "The manifest has more than one active model for " + role + ".";
                return result;
            }
            active = &row;
        }
        if (active == nullptr || !active->on_disk) continue;
        if (!result.roles.empty()) result.roles += ',';
        result.roles += role;
        if (role == "embedding") result.embedding_available = true;
    }
    if (!result.embedding_available)
        result.notice = "Memory tools are unavailable because no active embedding model file is on disk.";
    return result;
}

} // namespace aotx::ctrl::wizard
