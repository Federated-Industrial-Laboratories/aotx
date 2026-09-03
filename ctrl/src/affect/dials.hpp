// Purpose: Define affect setting dials and their live figure state.
// Owns: Editable values for the thirteen affect and quality settings.
// Launch shape: One interface thread draws one settings window.
// Lifetime: Values remain until the selected instance changes.
#ifndef AOTX_CTRL_AFFECT_DIALS_HPP
#define AOTX_CTRL_AFFECT_DIALS_HPP

#include <array>

namespace aotx::ctrl::client { class Client; }
namespace aotx::ctrl::replica { class State; }
namespace aotx::ctrl::toast { class Lane; }

namespace aotx::ctrl::affect::dials {

struct State {
    std::array<float, 13> values{};
    bool initialized = false;
};

void draw(State &view, replica::State &state, client::Client &client,
          toast::Lane &toasts, double now, bool *open);

} // namespace aotx::ctrl::affect::dials

#endif
