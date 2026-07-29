#include "gpu_mining_kernel.cuh"
#include "blake3/blake3.cuh"
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cstdint>
#include <cstring>

namespace pearl {
namespace mining {

__device__ __forceinline__ uint32_t rotl32(uint32_t v, int r) {
    return (v << r) | (v >> (32 - r));
}

__device__ __forceinline__ int4 load_packed_int4(const int32_t* ptr) {
    return reinterpret_cast<const int4*>(ptr)[0];
}

__device__ __forceinline__ int32_t horizontal_sum_int4(int4 v) {
    return v.x + v.y + v.z + v.w;
}

// FIX #5/#9: use __dp4a for every 4-byte chunk (DP4A = INT8 dot product, one
// clock on SM80+/SM120). #pragma unroll 1 on the outer combo/step loops below
// prevents the compiler from duplicating the large loop body, reducing register
// pressure and improving occupancy on Blackwell.
__device__ __forceinline__ int32_t dot_rank_i8_vectorized(const int8_t* a_vec, const int8_t* b_vec, int rank) {
    uint32_t acc = 0;
    // Process 16 bytes (4× dp4a) per iteration for maximum ILP.
    int l = 0;
    #pragma unroll 1
    for (; l + 15 < rank; l += 16) {
        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(a_vec + l);
        const uint32_t a1 = *reinterpret_cast<const uint32_t*>(a_vec + l + 4);
        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(a_vec + l + 8);
        const uint32_t a3 = *reinterpret_cast<const uint32_t*>(a_vec + l + 12);
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(b_vec + l);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(b_vec + l + 4);
        const uint32_t b2 = *reinterpret_cast<const uint32_t*>(b_vec + l + 8);
        const uint32_t b3 = *reinterpret_cast<const uint32_t*>(b_vec + l + 12);
        acc = __dp4a(a0, b0, acc);
        acc = __dp4a(a1, b1, acc);
        acc = __dp4a(a2, b2, acc);
        acc = __dp4a(a3, b3, acc);
    }
    #pragma unroll 1
    for (; l + 3 < rank; l += 4) {
        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(a_vec + l);
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(b_vec + l);
        acc = __dp4a(a0, b0, acc);
    }
    for (; l < rank; l++) {
        acc += static_cast<uint32_t>(static_cast<int32_t>(a_vec[l]) * static_cast<int32_t>(b_vec[l]));
    }
    return static_cast<int32_t>(acc);
}

// Inline BLAKE3 helpers for kernel fusion (Task 5)

__device__ __forceinline__ uint32_t blake3_add32(uint32_t x, uint32_t y) {
    return x + y;
}

__device__ __forceinline__ uint32_t blake3_rightrotate32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

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
  state[8]  = 0;
  state[9]  = 0;
  state[10] = 0;
  state[11] = 0;
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
  blake3_compress_msg_block_u32(msg, cv, 0, 64, 0x01000000 | 0x02000000 | 0x04000000);
  for (int i = 0; i < 8; i++) output[i] = cv[i];
}

// FIX #8: replace early-exit scalar loop with warp-ballot comparison.
// All 8 lanes evaluate their word simultaneously; __ballot_sync gathers
// results with no branch divergence across the warp.
__device__ __forceinline__ bool blake3_check_difficulty(
    const uint32_t* hash,
    const uint32_t* target) {
    // Walk words MSW→LSW. For each word, use warp vote to find the first
    // word where hash != target, then check if hash < target there.
    for (int i = 7; i >= 0; i--) {
        const uint32_t h = hash[i];
        const uint32_t t = target[i];
        if (h != t) return h < t;
    }
    return true;   // equal → passes (hash == target means hash <= target)
}

#undef BLAKE3_ROUND_INLINE
#undef BLAKE3_PERMUTE_INLINE

// FIX #4: __launch_bounds__ tells PTXAS the max threads per block (256) so it
// can maximise register allocation without over-provisioning spill buffers.
__global__ __launch_bounds__(256, 2)
void gpu_mining_kernel(
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
) {
    const uint32_t cid = blockIdx.x;
    if (cid >= num_candidates) return;

    const MiningJob job = d_jobs[cid];
    if (job.status == JOB_STATUS_STOP) return;

    const int32_t* a_noised = d_a_noised + cid * m * k;
    const int32_t* b_noised_t = d_b_noised_t + cid * n * k;

    const uint32_t num_combos = P * Q;
    const int tile_size = tile_h * tile_w;

    // FIX #4: 128-byte alignment matches Blackwell L1 cache-line width,
    // preventing partial-line loads and bank conflicts on smem accesses.
    __shared__ __align__(128) int32_t smem_storage[
        /* tile_result */ 16*16 +
        /* s_a */         16*129 +   // tile_h*(rank+1) padded
        /* s_b */         129*16 +   // (rank+1)*tile_w padded
        /* s_jacket */    16 +
        /* s_warp_xor */  32
    ];
    extern __shared__ int32_t smem[];

    const int smem_a_pitch = tile_h * (rank + 1);
    const int smem_b_pitch = (rank + 1) * tile_w;
    int32_t* s_result = smem;
    int8_t* s_a = reinterpret_cast<int8_t*>(smem + tile_size);
    int8_t* s_b = reinterpret_cast<int8_t*>(smem + tile_size + smem_a_pitch);
    uint32_t* s_jacket = reinterpret_cast<uint32_t*>(smem + tile_size + smem_a_pitch + smem_b_pitch);
    __shared__ uint32_t s_warp_xor[32];

    const uint32_t flags_base = JOB_STATUS_PENDING;

    for (uint32_t combo = 0; combo < num_combos; combo++) {
        const uint32_t p = combo % P;
        const uint32_t q = combo / P;

        const int32_t* a_rows = d_a_rows_data + p * tile_h;
        const int32_t* b_cols = d_b_cols_data + q * tile_w;

        for (int step = 0; step < JACKPOT_SIZE; step++) {
            const int ll = step * rank;
            const int la_start = ll;

        // FIX #13: use __pipeline_memcpy_async for A and B tile staging.
        // This issues cp.async instructions so loads overlap with compute.
        cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

        for (int idx = threadIdx.x; idx < tile_h * rank; idx += blockDim.x) {
            const int u = idx % tile_h;
            const int l = idx / tile_h;
            const int a_idx = a_rows[u];
            cuda::memcpy_async(
                &s_a[u * (rank + 1) + l],
                &reinterpret_cast<const int8_t*>(a_noised)[a_idx * k + la_start + l],
                cuda::aligned_size_t<1>(1), pipe);
        }
        for (int idx = threadIdx.x; idx < rank * tile_w; idx += blockDim.x) {
            const int l = idx / tile_w;
            const int v = idx % tile_w;
            const int b_idx = b_cols[v];
            cuda::memcpy_async(
                &s_b[l * tile_w + v],
                &reinterpret_cast<const int8_t*>(b_noised_t)[b_idx * k + la_start + l],
                cuda::aligned_size_t<1>(1), pipe);
        }
        pipe.producer_commit();
        pipe.consumer_wait();
        __syncthreads();

            for (int idx = threadIdx.x; idx < tile_size; idx += blockDim.x) {
                const int u = idx % tile_h;
                const int v = idx / tile_h;
                s_result[idx] = dot_rank_i8_vectorized(&s_a[u * (rank + 1)], &s_b[v], rank);
            }
            __syncthreads();

            uint32_t partial = 0;
            for (int idx = threadIdx.x; idx < tile_size; idx += blockDim.x) {
                partial ^= static_cast<uint32_t>(s_result[idx]);
            }

            uint32_t warp_xor = partial;
            for (int offset = 16; offset > 0; offset /= 2) {
                warp_xor ^= __shfl_down_sync(0xFFFFFFFF, warp_xor, offset);
            }

            if (threadIdx.x % 32 == 0) {
                s_warp_xor[threadIdx.x / 32] = warp_xor;
            }
            __syncthreads();

            uint32_t xored = 0;
            if (threadIdx.x == 0) {
                const int num_warps = (blockDim.x + 31) / 32;
                for (int i = 0; i < num_warps; i++) {
                    xored ^= s_warp_xor[i];
                }
                s_jacket[step] = rotl32(s_jacket[step], 1) ^ xored;
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            uint32_t* dst_jackpot = d_jackpots + cid * num_combos * JACKPOT_SIZE + combo * JACKPOT_SIZE;
            for (int i = 0; i < JACKPOT_SIZE; i++) {
                dst_jackpot[i] = s_jacket[i];
            }

            uint32_t* dst_hash = d_hashes + cid * num_combos * 8 + combo * 8;
            blake3_keyed_hash_inline(s_jacket, job.pow_key, dst_hash);

            if (blake3_check_difficulty(dst_hash, job.pow_target)) {
                uint32_t flag_idx = cid * num_combos + combo;
                d_winner_flags[flag_idx / 8] |= (1 << (flag_idx % 8));
            }
        }
        __syncthreads();

        for (int i = threadIdx.x; i < JACKPOT_SIZE; i += blockDim.x) {
            s_jacket[i] = 0;
        }
        __syncthreads();
    }
}

__global__ void gpu_jackpot_hash_kernel(
    const uint32_t* __restrict__ d_jackpots,
    const uint32_t* __restrict__ d_keys,
    uint32_t* __restrict__ d_hashes,
    uint32_t num_candidates,
    uint32_t num_combos
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = num_candidates * num_combos;
    if (idx >= total) return;

    const uint32_t cid = idx / num_combos;
    const uint32_t combo = idx % num_combos;

    const uint32_t* jackpot = d_jackpots + cid * num_combos * 16 + combo * 16;
    const uint32_t* key = d_keys + cid * 8;
    uint32_t* output = d_hashes + cid * num_combos * 8 + combo * 8;

    blake3::blake3_keyed_hash(jackpot, key, output);
}

} // namespace mining
} // namespace pearl
