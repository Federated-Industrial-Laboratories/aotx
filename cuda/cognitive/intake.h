/* Purpose: Define recorded source interpretations and their bounded response format.
 * Owns: Byte sizes and kind values; no address or credential is stored.
 * Launch shape: Batches of 1 to 64 source requests.
 * Lifetime: Recorded input, typed memory and file recovery. */
#ifndef AOTX_COGNITIVE_INTAKE_H
#define AOTX_COGNITIVE_INTAKE_H
#define AOTX_INTAKE_CHOICE 14u
#define AOTX_INTAKE_REPLY 4096u
#define AOTX_INTAKE_META 128u
#define AOTX_INTAKE_EXTRA (AOTX_INTAKE_META + AOTX_INTAKE_REPLY + AOTX_RECALL_SELECTION)
/* A valid JSON item uses at least ten bytes. This allocation covers every possible item. */
#define AOTX_INTAKE_ITEMS (AOTX_INTAKE_REPLY / 8u)
#define AOTX_INTAKE_PAYLOAD 96u
#define AOTX_INTAKE_PARTICIPANT 1u
#define AOTX_INTAKE_TASK 2u
#define AOTX_INTAKE_ASSERTION 3u
#define AOTX_INTAKE_CORRECTION 4u
#define AOTX_INTAKE_TICKS 4096u
#endif
