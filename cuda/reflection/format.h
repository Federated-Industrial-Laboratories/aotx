/* Purpose: Define source-bound task review records and limits.
 * Owns: Portable constants; no state or device address.
 * Launch shape: Batches of up to 64 independent source reviews.
 * Lifetime: Version 1 work records and version 4 memory cues. */
#ifndef AOTX_REFLECTION_FORMAT_H
#define AOTX_REFLECTION_FORMAT_H
#define AOTX_REVIEW_REQUEST 20u
#define AOTX_REVIEW_RESULT 21u
#define AOTX_REVIEW_BATCH 64u
#define AOTX_REVIEW_REFERENCES 5u
#define AOTX_REVIEW_SELECTION_BYTES (16u + 32u * AOTX_REVIEW_REFERENCES)
#define AOTX_REVIEW_TEXT "Review the supported outcome before repeating this task."
#define AOTX_REVIEW_TEXT_BYTES (sizeof(AOTX_REVIEW_TEXT) - 1u)
#define AOTX_REVIEW_CUE_BYTES (64u + AOTX_REVIEW_TEXT_BYTES)
#define AOTX_REVIEW_PAYLOAD (AOTX_REVIEW_SELECTION_BYTES + AOTX_REVIEW_CUE_BYTES)
#endif
