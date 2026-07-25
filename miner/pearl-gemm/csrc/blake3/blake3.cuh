#pragma once

#include <cstdint>
#include "blake3_constants.hpp"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"

#include <cutlass/arch/memory.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/detail/layout.hpp>

using namespace cute;

// A single BLAKE3 round (i.e. 8 G operations -- see section 2.2 of BLAKE3 spec)
// laid out in a primitive form, to allow the compiler to do everything in registers.
// state0-15 are equivalent to v0-15 in the spec.
#define BLAKE3_ROUND()                                          \
  do {                                                          \
    rState(0) = add32(rState(0), add32(rState(4), rBlock(0)));  \
    rState(12) = rightrotate32(rState(12) ^ rState(0), 16);     \
    rState(8) = add32(rState(8), rState(12));                   \
    rState(4) = rightrotate32(rState(4) ^ rState(8), 12);       \
    rState(0) = add32(rState(0), add32(rState(4), rBlock(1)));  \
    rState(12) = rightrotate32(rState(12) ^ rState(0), 8);      \
    rState(8) = add32(rState(8), rState(12));                   \
    rState(4) = rightrotate32(rState(4) ^ rState(8), 7);        \
    rState(1) = add32(rState(1), add32(rState(5), rBlock(2)));  \
    rState(13) = rightrotate32(rState(13) ^ rState(1), 16);     \
    rState(9) = add32(rState(9), rState(13));                   \
    rState(5) = rightrotate32(rState(5) ^ rState(9), 12);       \
    rState(1) = add32(rState(1), add32(rState(5), rBlock(3)));  \
    rState(13) = rightrotate32(rState(13) ^ rState(1), 8);      \
    rState(9) = add32(rState(9), rState(13));                   \
    rState(5) = rightrotate32(rState(5) ^ rState(9), 7);        \
    rState(2) = add32(rState(2), add32(rState(6), rBlock(4)));  \
    rState(14) = rightrotate32(rState(14) ^ rState(2), 16);     \
    rState(10) = add32(rState(10), rState(14));                 \
    rState(6) = rightrotate32(rState(6) ^ rState(10), 12);      \
    rState(2) = add32(rState(2), add32(rState(6), rBlock(5)));  \
    rState(14) = rightrotate32(rState(14) ^ rState(2), 8);      \
    rState(10) = add32(rState(10), rState(14));                 \
    rState(6) = rightrotate32(rState(6) ^ rState(10), 7);       \
    rState(3) = add32(rState(3), add32(rState(7), rBlock(6)));  \
    rState(15) = rightrotate32(rState(15) ^ rState(3), 16);     \
    rState(11) = add32(rState(11), rState(15));                 \
    rState(7) = rightrotate32(rState(7) ^ rState(11), 12);      \
    rState(3) = add32(rState(3), add32(rState(7), rBlock(7)));  \
    rState(15) = rightrotate32(rState(15) ^ rState(3), 8);      \
    rState(11) = add32(rState(11), rState(15));                 \
    rState(7) = rightrotate32(rState(7) ^ rState(11), 7);       \
    rState(0) = add32(rState(0), add32(rState(5), rBlock(8)));  \
    rState(15) = rightrotate32(rState(15) ^ rState(0), 16);     \
    rState(10) = add32(rState(10), rState(15));                 \
    rState(5) = rightrotate32(rState(5) ^ rState(10), 12);      \
    rState(0) = add32(rState(0), add32(rState(5), rBlock(9)));  \
    rState(15) = rightrotate32(rState(15) ^ rState(0), 8);      \
    rState(10) = add32(rState(10), rState(15));                 \
    rState(5) = rightrotate32(rState(5) ^ rState(10), 7);       \
    rState(1) = add32(rState(1), add32(rState(6), rBlock(10))); \
    rState(12) = rightrotate32(rState(12) ^ rState(1), 16);     \
    rState(11) = add32(rState(11), rState(12));                 \
    rState(6) = rightrotate32(rState(6) ^ rState(11), 12);      \
    rState(1) = add32(rState(1), add32(rState(6), rBlock(11))); \
    rState(12) = rightrotate32(rState(12) ^ rState(1), 8);      \
    rState(11) = add32(rState(11), rState(12));                 \
    rState(6) = rightrotate32(rState(6) ^ rState(11), 7);       \
    rState(2) = add32(rState(2), add32(rState(7), rBlock(12))); \
    rState(13) = rightrotate32(rState(13) ^ rState(2), 16);     \
    rState(8) = add32(rState(8), rState(13));                   \
    rState(7) = rightrotate32(rState(7) ^ rState(8), 12);       \
    rState(2) = add32(rState(2), add32(rState(7), rBlock(13))); \
    rState(13) = rightrotate32(rState(13) ^ rState(2), 8);      \
    rState(8) = add32(rState(8), rState(13));                   \
    rState(7) = rightrotate32(rState(7) ^ rState(8), 7);        \
    rState(3) = add32(rState(3), add32(rState(4), rBlock(14))); \
    rState(14) = rightrotate32(rState(14) ^ rState(3), 16);     \
    rState(9) = add32(rState(9), rState(14));                   \
    rState(4) = rightrotate32(rState(4) ^ rState(9), 12);       \
    rState(3) = add32(rState(3), add32(rState(4), rBlock(15))); \
    rState(14) = rightrotate32(rState(14) ^ rState(3), 8);      \
    rState(9) = add32(rState(9), rState(14));                   \
    rState(4) = rightrotate32(rState(4) ^ rState(9), 7);        \
  } while (0)

// BN-3: In-place cycle decomposition — saves 14 register moves vs. the
// rOrigBlock copy approach. The permutation decomposes into two disjoint
// 8-cycles each requiring exactly 1 temporary register.
// PROTOCOL SAFE: the resulting rBlock values are identical to the original.
#define BLAKE3_PERMUTE()                           \
  do {                                             \
    u32 _tmp;                                      \
    /* Cycle A: 0→2→3→10→12→9→11→5→0 */           \
    _tmp = rBlock(0);                              \
    rBlock(0)  = rBlock(2);                        \
    rBlock(2)  = rBlock(3);                        \
    rBlock(3)  = rBlock(10);                       \
    rBlock(10) = rBlock(12);                       \
    rBlock(12) = rBlock(9);                        \
    rBlock(9)  = rBlock(11);                       \
    rBlock(11) = rBlock(5);                        \
    rBlock(5)  = _tmp;                             \
    /* Cycle B: 1→6→4→7→13→14→15→8→1 */           \
    _tmp = rBlock(1);                              \
    rBlock(1)  = rBlock(6);                        \
    rBlock(6)  = rBlock(4);                        \
    rBlock(4)  = rBlock(7);                        \
    rBlock(7)  = rBlock(13);                       \
    rBlock(13) = rBlock(14);                       \
    rBlock(14) = rBlock(15);                       \
    rBlock(15) = rBlock(8);                        \
    rBlock(8)  = _tmp;                             \
  } while (0)

using u32 = uint32_t;
using u64 = uint64_t;

namespace blake3 {
// BLAKE3 compress parameters
// counter: A 64-bit counter, t = t0, t1, with t0 the lower order word and t1 the higher order word.
// block_len: The number of input bytes in the block (32 bits).
// flags: A set of domain separation bit flags (32 bits).
struct CompressParams {
  u64 counter;
  u32 block_len;
  u32 flags;
};

__device__ __constant__ u32 IV[8] = {IV0, IV1, IV2, IV3, IV4, IV5, IV6, IV7};

// Some constexpr variants of CompressParams
// Regular inner node compression in the Merkle Tree stage
__device__ __constant__ constexpr CompressParams COMPRESS_PARAMS_INNER_NODE = {
    .counter = 0,
    .block_len = MSG_BLOCK_SIZE,
    .flags = KEYED_HASH | PARENT};
// Compression of the root of the Merkle Tree
__device__ __constant__ constexpr CompressParams COMPRESS_PARAMS_ROOT = {
    .counter = 0,
    .block_len = MSG_BLOCK_SIZE,
    .flags = KEYED_HASH | ROOT | PARENT};
// Compression for noise generation (single 64-byte message block with key)
__device__ __constant__ constexpr CompressParams
    COMPRESS_PARAMS_SINGLE_BLOCK_KEYED = {
        .counter = 0,
        .block_len = MSG_BLOCK_SIZE,
        .flags = KEYED_HASH | CHUNK_START | CHUNK_END | ROOT};

// BN-9: __forceinline__ ensures no call overhead on any inlining boundary.
CUTLASS_DEVICE
u32 add32(u32 x, u32 y) {
  return x + y;
}

CUTLASS_DEVICE
u32 rightrotate32(u32 x, u32 n) {
  // BN-2: shf.r.wrap.b32 is a single-cycle barrel rotate on SM90.
  // Semantically identical to (x >> n) | (x << (32-n)) for n in [1,31].
  // BLAKE3 rotation constants (7, 8, 12, 16) are all compile-time immediates,
  // so NVCC will emit shf.r.wrap.b32 with immediate operand.
  u32 result;
  asm("shf.r.wrap.b32 %0, %1, %1, %2;" : "=r"(result) : "r"(x), "r"(n));
  return result;
}

// Compress a 64-byte message block
// For keyed hashes (params.flags & KEYED_HASH), state[8..11] must be 0, not IV.
// For non-keyed hashes, state[8..11] = IV0..IV3 per the BLAKE3 spec.
template <class RmemTensorBlock, class RmemTensorChainingValue>
CUTLASS_DEVICE void compress_msg_block_u32(
    RmemTensorBlock const& block, RmemTensorChainingValue& chaining_value,
    const CompressParams& params) {  // BN-6: const ref avoids 4-register copy
  Tensor rState = make_tensor_like<uint32_t>(block);
  Tensor rBlock = make_tensor_like<uint32_t>(block);

  copy(block, rBlock);
  // Initialize state
  copy(chaining_value, rState);
  if (params.flags & KEYED_HASH) {
    rState(8) = 0;
    rState(9) = 0;
    rState(10) = 0;
    rState(11) = 0;
  } else {
    rState(8) = IV0;
    rState(9) = IV1;
    rState(10) = IV2;
    rState(11) = IV3;
  }
  rState(12) = params.counter;
  rState(13) = params.counter >> 32;
  rState(14) = params.block_len;
  rState(15) = params.flags;

  // 6 rounds and permutations
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 6; ++i) {
    BLAKE3_ROUND();
    BLAKE3_PERMUTE();
  }
  // Final round w/o permutation
  BLAKE3_ROUND();
  // Real BLAKE3 has some operations here on state8-15, but we don't care about these
  // so we can only change state0-7. Copy the result to the chaining value tensor.
  chaining_value(0) = rState(0) ^ rState(8);
  chaining_value(1) = rState(1) ^ rState(9);
  chaining_value(2) = rState(2) ^ rState(10);
  chaining_value(3) = rState(3) ^ rState(11);
  chaining_value(4) = rState(4) ^ rState(12);
  chaining_value(5) = rState(5) ^ rState(13);
  chaining_value(6) = rState(6) ^ rState(14);
  chaining_value(7) = rState(7) ^ rState(15);
}

// Single-block keyed BLAKE3 hash: blake3(msg, key) for a 64-byte message.
// Output is 32 bytes. This matches blake3_digest(&msg, Some(key)) from the CPU.
CUTLASS_DEVICE void blake3_keyed_hash(const uint32_t* msg, const uint32_t* key, uint32_t* output) {
  Tensor data = make_tensor(make_gmem_ptr(msg), Int<blake3::MSG_BLOCK_SIZE_U32>{});
  Tensor chaining_value = make_tensor(make_gmem_ptr(key), Int<blake3::CHAINING_VALUE_SIZE_U32>{});
  Tensor out = make_tensor(make_gmem_ptr(output), Int<blake3::CHAINING_VALUE_SIZE_U32>{});
  copy(chaining_value, out);
  blake3::compress_msg_block_u32(data, out, blake3::COMPRESS_PARAMS_SINGLE_BLOCK_KEYED);
}

}  // namespace blake3
