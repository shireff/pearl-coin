#pragma once

#include <cstdint>
#include <cstddef>

namespace pearl {
namespace mining {

constexpr uint32_t JOB_STATUS_PENDING = 0;
constexpr uint32_t JOB_STATUS_RUNNING = 1;
constexpr uint32_t JOB_STATUS_DONE = 2;
constexpr uint32_t JOB_STATUS_STOP = 0xFFFFFFFF;

constexpr int JACKPOT_SIZE = 16;
constexpr int LROT_PER_TILE = 1;

struct alignas(16) MiningJob {
    uint32_t status;
    uint32_t candidate_id;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t rank;
    uint32_t tile_h;
    uint32_t tile_w;
    uint32_t num_tiles;
    uint32_t a_offset;
    uint32_t b_offset;
    uint32_t a_rows_offset;
    uint32_t b_cols_offset;
    uint32_t result_offset;
    uint32_t pow_target[8];
    uint32_t pow_key[8];
    uint32_t a_noise_seed;
    uint32_t b_noise_seed;
};

struct alignas(16) MiningResult {
    uint32_t status;
    uint32_t candidate_id;
    uint32_t tile_idx;
    uint32_t a_rows_offset;
    uint32_t b_cols_offset;
};

void launch_gpu_mining(
    const int32_t* a_noised,
    const int32_t* b_noised_t,
    const int32_t* a_rows_data,
    const int32_t* b_cols_data,
    const MiningJob* jobs,
    uint32_t* jackpots,
    uint32_t* hashes,
    uint8_t* winner_flags,
    uint32_t num_candidates,
    uint32_t P,
    uint32_t Q,
    uint32_t tile_h,
    uint32_t tile_w,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    uint32_t rank,
    cudaStream_t stream);

__device__ __forceinline__ bool blake3_check_difficulty(
    const uint32_t* hash,
    const uint32_t* target)
{
    for (int i = 7; i >= 0; i--) {
        const uint32_t h = hash[i];
        const uint32_t t = target[i];
        if (h != t) return h < t;
    }
    return true;
}

__device__ __forceinline__ void blake3_keyed_hash_inline(
    const uint32_t* msg,
    const uint32_t* key,
    uint32_t* output);

#define BLAKE3_ROUND_INLINE(state, block)                    \
    do {                                                      \
      state[0]  = blake3_add32(state[0],  blake3_add32(state[4],  block[0]));  \
      state[12] = blake3_rightrotate32(state[12] ^ state[0],  16);             \
      state[8]  = blake3_add32(state[8],  state[12]);                        \
      state[4]  = blake3_rightrotate32(state[4]  ^ state[8],  12);            \
      state[0]  = blake3_add32(state[0],  blake3_add32(state[4],  block[1]));  \
      state[12] = blake3_rightrotate32(state[12] ^ state[0],   8);             \
      state[8]  = blake3_add32(state[8],  state[12]);                        \
      state[4]  = blake3_rightrotate32(state[4]  ^ state[8],   7);            \
      state[1]  = blake3_add32(state[1],  blake3_add32(state[5],  block[2]));  \
      state[13] = blake3_rightrotate32(state[13] ^ state[1],  16);             \
      state[9]  = blake3_add32(state[9],  state[13]);                        \
      state[5]  = blake3_rightrotate32(state[5]  ^ state[9],  12);            \
      state[1]  = blake3_add32(state[1],  blake3_add32(state[5],  block[3]));  \
      state[13] = blake3_rightrotate32(state[13] ^ state[1],   8);             \
      state[9]  = blake3_add32(state[9],  state[13]);                        \
      state[5]  = blake3_rightrotate32(state[5]  ^ state[9],   7);            \
      state[2]  = blake3_add32(state[2],  blake3_add32(state[6],  block[4]));  \
      state[14] = blake3_rightrotate32(state[14] ^ state[2],  16);             \
      state[10] = blake3_add32(state[10], state[14]);                        \
      state[6]  = blake3_rightrotate32(state[6]  ^ state[10], 12);            \
      state[2]  = blake3_add32(state[2],  blake3_add32(state[6],  block[5]));  \
      state[14] = blake3_rightrotate32(state[14] ^ state[2],   8);             \
      state[10] = blake3_add32(state[10], state[14]);                        \
      state[6]  = blake3_rightrotate32(state[6]  ^ state[10],  7);            \
      state[3]  = blake3_add32(state[3],  blake3_add32(state[7],  block[6]));  \
      state[15] = blake3_rightrotate32(state[15] ^ state[3],  16);             \
      state[11] = blake3_add32(state[11], state[15]);                        \
      state[7]  = blake3_rightrotate32(state[7]  ^ state[11], 12);            \
      state[3]  = blake3_add32(state[3],  blake3_add32(state[7],  block[7]));  \
      state[15] = blake3_rightrotate32(state[15] ^ state[3],   8);             \
      state[11] = blake3_add32(state[11], state[15]);                        \
      state[7]  = blake3_rightrotate32(state[7]  ^ state[11],  7);            \
      state[0]  = blake3_add32(state[0],  blake3_add32(state[5],  block[8]));  \
      state[15] = blake3_rightrotate32(state[15] ^ state[0],  16);             \
      state[10] = blake3_add32(state[10], state[15]);                        \
      state[5]  = blake3_rightrotate32(state[5]  ^ state[10], 12);            \
      state[0]  = blake3_add32(state[0],  blake3_add32(state[5],  block[9]));  \
      state[15] = blake3_rightrotate32(state[15] ^ state[0],   8);             \
      state[10] = blake3_add32(state[10], state[15]);                        \
      state[5]  = blake3_rightrotate32(state[5]  ^ state[10],  7);            \
      state[1]  = blake3_add32(state[1],  blake3_add32(state[6],  block[10])); \
      state[12] = blake3_rightrotate32(state[12] ^ state[1],  16);             \
      state[11] = blake3_add32(state[11], state[12]);                        \
      state[6]  = blake3_rightrotate32(state[6]  ^ state[11], 12);            \
      state[1]  = blake3_add32(state[1],  blake3_add32(state[6],  block[11])); \
      state[12] = blake3_rightrotate32(state[12] ^ state[1],   8);             \
      state[11] = blake3_add32(state[11], state[12]);                        \
      state[6]  = blake3_rightrotate32(state[6]  ^ state[11],  7);            \
      state[2]  = blake3_add32(state[2],  blake3_add32(state[7],  block[12])); \
      state[13] = blake3_rightrotate32(state[13] ^ state[2],  16);             \
      state[8]  = blake3_add32(state[8],  state[13]);                        \
      state[7]  = blake3_rightrotate32(state[7]  ^ state[8],  12);            \
      state[2]  = blake3_add32(state[2],  blake3_add32(state[7],  block[13])); \
      state[13] = blake3_rightrotate32(state[13] ^ state[2],   8);             \
      state[8]  = blake3_add32(state[8],  state[13]);                        \
      state[7]  = blake3_rightrotate32(state[7]  ^ state[8],   7);            \
      state[3]  = blake3_add32(state[3],  blake3_add32(state[4],  block[14])); \
      state[14] = blake3_rightrotate32(state[14] ^ state[3],  16);             \
      state[9]  = blake3_add32(state[9],  state[14]);                        \
      state[4]  = blake3_rightrotate32(state[4]  ^ state[9],  12);            \
      state[3]  = blake3_add32(state[3],  blake3_add32(state[4],  block[15])); \
      state[14] = blake3_rightrotate32(state[14] ^ state[3],   8);             \
      state[9]  = blake3_add32(state[9],  state[14]);                        \
      state[4]  = blake3_rightrotate32(state[4]  ^ state[9],   7);            \
    } while (0)

#define BLAKE3_PERMUTE_INLINE(block)                          \
    do {                                                       \
      uint32_t _tmp;                                           \
      _tmp = block[0];  block[0]  = block[2];  block[2]  = block[3];  block[3]  = block[10]; block[10] = block[12]; block[12] = block[9];  block[9]  = block[11]; block[11] = block[5];  block[5]  = _tmp; \
      _tmp = block[1];  block[1]  = block[6];  block[6]  = block[4];  block[4]  = block[7];  block[7]  = block[13]; block[13] = block[14]; block[14] = block[15]; block[15] = block[8];  block[8]  = _tmp; \
    } while (0)

__device__ __forceinline__ uint32_t blake3_add32(uint32_t x, uint32_t y) {
    return x + y;
}

__device__ __forceinline__ uint32_t blake3_rightrotate32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ void blake3_compress_msg_block_u32(
    const uint32_t* msg,
    uint32_t* chaining_value,
    uint64_t counter,
    uint32_t block_len,
    uint32_t flags) {
  uint32_t state[16];
  uint32_t block[16];

  for (int i = 0; i < 16; i++) block[i] = msg[i];

  state[0]  = chaining_value[0];
  state[1]  = chaining_value[1];
  state[2]  = chaining_value[2];
  state[3]  = chaining_value[3];
  state[4]  = chaining_value[4];
  state[5]  = chaining_value[5];
  state[6]  = chaining_value[6];
  state[7]  = chaining_value[7];
  // Blake3 spec §2.3: state[8..11] = IV[0..3], not 0
  state[8]  = 0x6A09E667u;
  state[9]  = 0xBB67AE85u;
  state[10] = 0x3C6EF372u;
  state[11] = 0xA54FF53Au;
  state[12] = static_cast<uint32_t>(counter);
  state[13] = static_cast<uint32_t>(counter >> 32);
  state[14] = block_len;
  state[15] = flags;

  BLAKE3_ROUND_INLINE(state, block);
  BLAKE3_PERMUTE_INLINE(block);
  BLAKE3_ROUND_INLINE(state, block);
  BLAKE3_PERMUTE_INLINE(block);
  BLAKE3_ROUND_INLINE(state, block);
  BLAKE3_PERMUTE_INLINE(block);
  BLAKE3_ROUND_INLINE(state, block);
  BLAKE3_PERMUTE_INLINE(block);
  BLAKE3_ROUND_INLINE(state, block);
  BLAKE3_PERMUTE_INLINE(block);
  BLAKE3_ROUND_INLINE(state, block);
  BLAKE3_ROUND_INLINE(state, block);

  chaining_value[0] = state[0] ^ state[8];
  chaining_value[1] = state[1] ^ state[9];
  chaining_value[2] = state[2] ^ state[10];
  chaining_value[3] = state[3] ^ state[11];
  chaining_value[4] = state[4] ^ state[12];
  chaining_value[5] = state[5] ^ state[13];
  chaining_value[6] = state[6] ^ state[14];
  chaining_value[7] = state[7] ^ state[15];
}

__device__ __forceinline__ void blake3_keyed_hash_inline(
    const uint32_t* msg,
    const uint32_t* key,
    uint32_t* output) {
  uint32_t cv[8];
  for (int i = 0; i < 8; i++) cv[i] = key[i];
  // CHUNK_START | CHUNK_END | ROOT | KEYED_HASH = 1 | 2 | 8 | 16 = 0x1B
  blake3_compress_msg_block_u32(msg, cv, 0, 64, 0x1u | 0x2u | 0x8u | 0x10u);
  for (int i = 0; i < 8; i++) output[i] = cv[i];
}

// Standard (non-keyed) BLAKE3 hash — uses BLAKE3 IV as chaining value.
// Used for Alephium PoW: Blake3(blob_with_nonce) with no custom key.
// BLAKE3 IV = first 8 words of SHA-256 IV (fractional parts of square roots).
__device__ __forceinline__ void blake3_standard_hash_inline(
    const uint32_t* msg,   // 16 x uint32 = 64 bytes input block
    uint32_t* output) {    // 8 x uint32 = 32 bytes output hash
  uint32_t cv[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
  };
  // CHUNK_START | CHUNK_END | ROOT flags = single-chunk, single-block hash
  // CHUNK_START | CHUNK_END | ROOT = 1 | 2 | 8 = 0xB
  blake3_compress_msg_block_u32(msg, cv, 0, 64, 0x1u | 0x2u | 0x8u);
  for (int i = 0; i < 8; i++) output[i] = cv[i];
}

__global__ void gpu_mining_kernel(
    const int32_t* __restrict__ d_a_noised,
    const int32_t* __restrict__ d_b_noised_t,
    const int32_t* __restrict__ d_a_rows_data,
    const int32_t* __restrict__ d_b_cols_data,
    const MiningJob* __restrict__ d_jobs,
    uint32_t* __restrict__ d_jackpots,
    uint32_t* __restrict__ d_hashes,
    uint8_t* __restrict__ d_winner_flags,
    uint32_t num_candidates,
    uint32_t P,
    uint32_t Q,
    uint32_t tile_h,
    uint32_t tile_w,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    uint32_t rank
);

__global__ void gpu_jackpot_hash_kernel(
    const uint32_t* __restrict__ d_jackpots,
    const uint32_t* __restrict__ d_keys,
    uint32_t* __restrict__ d_hashes,
    uint32_t num_candidates,
    uint32_t num_combos
);

void launch_gpu_jackpot_hash(
    const uint32_t* jackpots,
    const uint32_t* keys,
    uint32_t* hashes,
    uint32_t num_candidates,
    uint32_t num_combos,
    cudaStream_t stream);

cudaError_t launch_gpu_candidate_generation(
    int8_t* d_a_raw,
    int8_t* d_b_raw_t,
    uint32_t batch_size,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    uint64_t base_seed,
    cudaStream_t stream);

} // namespace mining
} // namespace pearl
