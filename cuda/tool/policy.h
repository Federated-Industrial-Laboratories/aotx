/* Purpose: Define the system tool choices shared by the device and its clients.
 * Owns: Constants for ten tool groups and three conversation choices.
 * Launch shape: Not applicable; constants only.
 * Lifetime: The settings and command formats. */
#ifndef AOTX_TOOL_POLICY_H
#define AOTX_TOOL_POLICY_H

#define AOTX_TOOL_POLICY_GROUPS 10u
#define AOTX_TOOL_POLICY_ALL 1023u
#define AOTX_TOOL_POLICY_CHOICES 1048575u
#define AOTX_TOOL_POLICY_INHERIT 0u
#define AOTX_TOOL_POLICY_OFF 1u
#define AOTX_TOOL_POLICY_ON 2u

#define AOTX_TOOL_POLICY_NAMES(X) \
    X("memory_recall") X("memory_write") X("fs_read") X("fs_list") \
    X("fs_write") X("fs_update") X("run") X("skill_use") X("fs_stat") X("imported")

#endif
