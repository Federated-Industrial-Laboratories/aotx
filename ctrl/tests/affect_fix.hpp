// Purpose: Declare focused checks for affect replica data.
// Owns: No test state outside one call.
// Launch shape: One host thread applies each fixture in order.
// Lifetime: All fixture data ends before the call returns.
#ifndef AOTX_CTRL_TESTS_AFFECT_FIX_HPP
#define AOTX_CTRL_TESTS_AFFECT_FIX_HPP

void aotx_ctrl_affect_fix(int &applied, int &failed);

#endif
