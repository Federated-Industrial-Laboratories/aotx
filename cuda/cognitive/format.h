/* Purpose: Define portable typed state, media and replay byte layouts.
 * Owns: Schema constants; no process address or deployment credential is stored.
 * Launch shape: Shared constants for batched device work and disk framing.
 * Lifetime: Schema 1 files and state images. */
#ifndef AOTX_COGNITIVE_FORMAT_H
#define AOTX_COGNITIVE_FORMAT_H
#include <stdint.h>

#define AOTX_COG_SCHEMA 1u
#define AOTX_COG_HEADER 128u
#define AOTX_COG_OBJECT 256u
#define AOTX_COG_OBJECTS 256u
#define AOTX_COG_PAYLOAD (1024u * 1024u)
#define AOTX_COG_IMAGE (AOTX_COG_HEADER + AOTX_COG_OBJECTS * AOTX_COG_OBJECT + AOTX_COG_PAYLOAD)
#define AOTX_COG_UNKNOWN UINT32_MAX
#define AOTX_COG_SCALE 1000000u

#define AOTX_COG_EVENT 1u
#define AOTX_COG_ASSERTION 2u
#define AOTX_COG_APPRAISAL 3u
#define AOTX_COG_RELATIONSHIP 4u
#define AOTX_COG_CUE 5u
#define AOTX_COG_INTENTION 6u
#define AOTX_COG_WORKING 7u
#define AOTX_COG_MEDIA 8u
#define AOTX_COG_COMPONENT 9u
#define AOTX_COG_SELECTION 10u
#define AOTX_COG_POLICY 11u
#define AOTX_COG_IDENTITY 12u
#define AOTX_COG_TOMBSTONE 1u
#define AOTX_COG_PROTECTED 2u
#define AOTX_COG_PRIVATE 0u
#define AOTX_COG_ROOM 1u
#define AOTX_COG_INSTANCE 2u
#define AOTX_COG_AUTHORED 1u
#define AOTX_COG_OBSERVED 2u
#define AOTX_COG_REPORTED 3u
#define AOTX_COG_INFERRED 4u

/* Object offsets. All integer fields are unsigned and little endian. */
#define AOTX_CO_SCHEMA 0u      /* uint16 */
#define AOTX_CO_KIND 2u        /* uint16 */
#define AOTX_CO_FLAGS 4u       /* uint32 */
#define AOTX_CO_ID 8u          /* 16 bytes */
#define AOTX_CO_LINEAGE 24u    /* 16 bytes */
#define AOTX_CO_VERSION 40u    /* uint64 */
#define AOTX_CO_CREATED 48u    /* uint64 */
#define AOTX_CO_UPDATED 56u    /* uint64 */
#define AOTX_CO_OWNER 64u      /* 16 bytes */
#define AOTX_CO_ROOM 80u       /* 16 bytes */
#define AOTX_CO_SOURCE 96u     /* 16 bytes */
#define AOTX_CO_SOURCE_VERSION 112u /* uint64 */
#define AOTX_CO_SUBJECT 120u   /* 16 bytes */
#define AOTX_CO_SUPERSEDES 136u /* 16 bytes */
#define AOTX_CO_SUPER_VERSION 152u /* uint64 */
#define AOTX_CO_OFFSET 160u    /* uint64, relative to the payload arena */
#define AOTX_CO_BYTES 168u     /* uint64 */
#define AOTX_CO_SCOPE 176u     /* uint32 */
#define AOTX_CO_SOURCE_KIND 180u /* uint32 */
#define AOTX_CO_EVIDENCE 184u  /* uint32: unknown, supported, disputed, or withdrawn */
#define AOTX_CO_RETENTION 188u /* uint32: ordinary, retained, or pending */
#define AOTX_CO_IMPORTANCE 192u /* uint32, scale or unknown */
#define AOTX_CO_EXPIRY 200u    /* uint64 sequence, zero has no expiry */
#define AOTX_CO_EMBEDDING 208u /* 16 bytes */
#define AOTX_CO_EMBED_VERSION 224u /* uint64 */
#define AOTX_CO_POLICY 232u    /* uint64, nonzero schema policy revision */
/* Bytes 196..199 and 240..255 are reserved zero. */

/* Media payload: 192-byte descriptor, sample/feature bytes, then uint64 positions.
 * Offsets: schema 0, modality 4, representation 8, dtype 12, four uint64 dimensions 16.
 * Further offsets: data bytes 48, position count 56, layout 64, rank 68, source SHA-256 72.
 * Remaining offsets: model SHA-256 104, processor SHA-256 136, audio sample rate uint32 168,
 * reserved zero 172..191. The sample rate is zero for non-audio representations. */
#define AOTX_COG_MEDIA_HEADER 192u
#define AOTX_COG_IMAGE_MEDIA 1u
#define AOTX_COG_AUDIO_MEDIA 2u
#define AOTX_COG_FEATURE_MEDIA 3u
#define AOTX_COG_SOURCE_BYTES 1u
#define AOTX_COG_EXACT_FEATURES 2u
#define AOTX_COG_U8 1u
#define AOTX_COG_I16 2u
#define AOTX_COG_F16 3u
#define AOTX_COG_F32 4u
#define AOTX_COG_LINEAR 1u
#define AOTX_COG_SPATIAL 2u
#define AOTX_COG_TEMPORAL 3u

/* Appraisal: eight uint32 values: schema, benefit, harm, arousal, consequence category,
 * confidence, units revision, reserved zero. Category 0 is unknown; 1..4 are ordered.
 * Intensities and confidence use AOTX_COG_SCALE; unknown is distinct from zero.
 * Confidence has no implied calibration or probability meaning. */
#define AOTX_COG_APPRAISAL_BYTES 32u
/* Selection: uint32 schema/count, eight reserved zero bytes, then 32-byte rows:
 * object ID, uint64 version, uint32 representation (1 text, 2 media), reserved zero. */
#define AOTX_COG_SELECTION_MAX 64u

#define AOTX_COG_OK 0u
#define AOTX_COG_FORMAT 1u
#define AOTX_COG_CAPACITY 2u
#define AOTX_COG_REFERENCE 3u
#define AOTX_COG_SCOPE 4u
#define AOTX_COG_VERSION 5u
#define AOTX_COG_SEQUENCE 6u
#define AOTX_COG_SOURCE 7u
#define AOTX_COG_LAYOUT 8u
#define AOTX_COG_MISSING 9u
#define AOTX_COG_STALE 10u
#define AOTX_COG_DENIED 11u
#endif
