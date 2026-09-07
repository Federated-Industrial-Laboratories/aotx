// Purpose: Keep first-run model selection and startup roles consistent.
// Owns: Store rows, a full selection identity, and startup role facts.
// Threading: One caller updates a complete row set at each refresh.
// Lifetime: Selection identity remains until the caller selects another model.
#ifndef AOTX_CTRL_WIZARD_MODELS_HPP
#define AOTX_CTRL_WIZARD_MODELS_HPP

#include "instances/lifecycle.hpp"
#include "replica/replica.hpp"

#include <string>
#include <vector>

namespace aotx::ctrl::wizard {

class ModelSelection {
  public:
    explicit ModelSelection(std::string preferred_role = AOTX_CTRL_LANGUAGE_ROLE);
    void refresh(bool readable, std::vector<replica::Model> rows, std::string reason = {});
    bool select(const std::string &identity);
    bool request(const std::string &name_or_identity, const std::string &role,
                 std::string &reason);
    const replica::Model *selected() const;
    const std::vector<replica::Model> &rows() const;
    const std::string &identity() const;
    bool readable() const;
    std::string refusal() const;

  private:
    std::vector<replica::Model> rows_;
    std::string identity_, preferred_role_, read_reason_;
    bool readable_ = false;
};

struct ModelRoles {
    std::string roles;
    std::string refusal;
    std::string notice;
    bool embedding_available = false;
    bool ready() const { return !roles.empty() && refusal.empty(); }
};

bool language_role(const std::string &role);
bool can_fetch(const replica::Model *model);
bool can_activate(const replica::Model *model);
ModelRoles model_roles(const ModelSelection &selection);

} // namespace aotx::ctrl::wizard
#endif
