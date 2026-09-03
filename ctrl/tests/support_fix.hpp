// Purpose: Declare storage and child-process support checks.
// Owns: No test state outside one call.
// Launch shape: One host thread applies each fixture in order.
// Lifetime: All fixture data ends before the call returns.
#ifndef AOTX_CTRL_TESTS_SUPPORT_FIX_HPP
#define AOTX_CTRL_TESTS_SUPPORT_FIX_HPP

void aotx_ctrl_support_fix(int &applied, int &failed);

#endif
