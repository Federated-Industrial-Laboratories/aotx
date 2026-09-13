/* Purpose: Set the bounded shared service table sizes.
 * Owns: Compile-time capacities for persistent device state.
 * Threading: The host allocates the tables before replay.
 * Lifetime: One compatible complete runtime file. */
#ifndef AOTX_SHARED_PROFILE_H
#define AOTX_SHARED_PROFILE_H
#ifndef AOTX_SHARED_PARTICIPANTS
#define AOTX_SHARED_PARTICIPANTS 1024u
#endif
#ifndef AOTX_SHARED_SPACES
#define AOTX_SHARED_SPACES 1024u
#endif
#ifndef AOTX_SHARED_CONVERSATIONS
#define AOTX_SHARED_CONVERSATIONS 4096u
#endif
#ifndef AOTX_SHARED_MEMBERS
#define AOTX_SHARED_MEMBERS 4096u
#endif
#ifndef AOTX_SHARED_RECEIPTS
#define AOTX_SHARED_RECEIPTS 1024u
#endif
#ifndef AOTX_SHARED_COMMAND_BYTES
#define AOTX_SHARED_COMMAND_BYTES 8192u
#endif
#ifndef AOTX_SHARED_RESULT_BYTES
#define AOTX_SHARED_RESULT_BYTES 65536u
#endif
#define AOTX_SHARED_EMIT 16u
#define AOTX_SHARED_TRANSFER (AOTX_SHARED_COMMAND_BYTES + 512u)
#endif
