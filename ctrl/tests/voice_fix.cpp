// Purpose: Check spoken voice coupling gains and caps.
// Owns: No state outside one focused check.
// Launch shape: One host thread maps three bounded states.
// Lifetime: All mapped values end before the check returns.
#include "voice_fix.hpp"

#include "voice/coupling.hpp"

#include <cmath>
#include <cstdio>

namespace {

bool near(double left, double right)
{
    return std::fabs(left - right) < 0.0000001;
}

} // namespace

void aotx_ctrl_voice_fix(int &applied, int &failed)
{
    using aotx::ctrl::voice::CouplingValues;
    using aotx::ctrl::voice::spoken_voice_values;
    const CouplingValues zero = spoken_voice_values(0.0, 0.0);
    const CouplingValues high = spoken_voice_values(2.0, 2.0);
    const CouplingValues low = spoken_voice_values(-2.0, -2.0);
    ++applied;
    if (near(zero.length, 1.0) && near(zero.noise, 0.667) && near(zero.width, 0.8) &&
        near(high.length, 0.75) && near(high.noise, 0.834) && near(high.width, 0.9) &&
        near(low.length, 1.25) && near(low.noise, 0.5) && near(low.width, 0.7)) return;
    ++failed;
    std::printf("ctrl fix: the spoken voice coupling did not hold its neutral values and caps\n");
}
