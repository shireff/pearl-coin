#include "alph_mining_kernel.cuh"
#include "gpu_mining_kernel.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <algorithm>

namespace pearl {
namespace mining {

// ── Alephium Blake3 PoW mining kernel ────────────────────────────────────────
//
// Alephium PoW contract for the pool is:
//   hash2 = BLAKE3(BLAKE3(nonce24 || headerBlob))
//   chain index gate is derived from hash2[31] % 16
// Queued share validity is then: hash2 <= target and chain-group matches.
//
// This kernel now follows the canonical nonce representation instead of the
// old in-blob uint64 overwrite scheme.
// ──────────────────────────────────────────────────────────────────────────────

static constexpr int BLOB_LEN      = 302;
static constexpr int NOUNCE_LEN    = 24;
static constexpr int HASH_LEN      = 32;
static constexpr int HASH_BLOCKS   = 6;            // ceil((24 + 302)/64) = 6
static constexpr int HASH_BYTES    = HASH_BLOCKS * 64;  // 384 bytes, padded
static constexpr int HASH_WORDS    = HASH_BYTES / 4;  // 96 words
static constexpr int PADDED_BYTES  = 320;
static constexpr int PADDED_WORDS  = PADDED_BYTES / 4;

__global__
__launch_bounds__(256, 4)
void alph_mine_kernel(
    const uint32_t* __restrict__ d_blob_words,
    uint64_t base_nonce,
    uint32_t num_nonces,
    const uint32_t* __restrict__ d_target,
    int64_t*  __restrict__ d_found_nonce,
    uint8_t*  __restrict__ d_found_flag) {

    __shared__ uint32_t s_blob[PADDED_WORDS];
    __shared__ uint32_t s_target[8];

    for (int i = threadIdx.x; i < PADDED_WORDS; i += blockDim.x)
        s_blob[i] = d_blob_words[i];
    for (int i = threadIdx.x; i < 8; i += blockDim.x)
        s_target[i] = __byte_perm(d_target[i], 0, 0x0123);
    __syncthreads();

    if (*d_found_flag) return;

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nonces) return;
    uint64_t nonce = base_nonce + tid;

    // Build the canonical 24-byte nonce in big-endian form.
    uint8_t nonce24[NOUNCE_LEN];
    #pragma unroll
    for (int i = 0; i < NOUNCE_LEN; ++i)
        nonce24[i] = static_cast<uint8_t>((nonce >> (8 * (NOUNCE_LEN - 1 - i))) & 0xFFu);

    // Copy the 302-byte header blob into local 32-bit registers.
    uint32_t r[PADDED_WORDS];
    #pragma unroll 4
    for (int i = 0; i < PADDED_WORDS; i++)
        r[i] = s_blob[i];

    const uint8_t* blob_bytes = reinterpret_cast<const uint8_t*>(r);

    // First BLAKE3 pass over nonce24 || headerBlob.
    uint32_t cv[8] = {
        0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
        0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
    };

    for (int block = 0; block < HASH_BLOCKS; ++block) {
        uint32_t msg[16] = {0};
        const int block_offset = block * 64;
        int block_end = block_offset + 64;
        if (block_end > NOUNCE_LEN + BLOB_LEN)
            block_end = NOUNCE_LEN + BLOB_LEN;

        #pragma unroll
        for (int w = 0; w < 16; ++w) {
            uint32_t word = 0;
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int idx = block_offset + (w * 4) + b;
                uint8_t byte_value = 0;
                if (idx < NOUNCE_LEN) {
                    byte_value = nonce24[idx];
                } else if (idx - NOUNCE_LEN < BLOB_LEN) {
                    byte_value = blob_bytes[idx - NOUNCE_LEN];
                }
                word |= static_cast<uint32_t>(byte_value) << (8 * b);
            }
            msg[w] = word;
        }

        uint32_t block_len = 64;
        uint32_t flags = 0u;
        if (block == 0) {
            flags = 0x1u;
        }
        if (block == HASH_BLOCKS - 1) {
            block_len = block_end - block_offset;
            flags |= 0x2u | 0x8u;
        }

        blake3_compress_msg_block_u32(msg, cv, 0, block_len, flags);
    }

    // Second BLAKE3 pass over the 32-byte first-pass digest:
    uint32_t outer_msg[16] = {0};
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        outer_msg[i] = cv[i];

    uint32_t outer_cv[8] = {
        0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
        0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
    };
    blake3_compress_msg_block_u32(outer_msg, outer_cv, 0, HASH_LEN, 0x1u | 0x2u | 0x8u);

    // Compare the final 32-byte hash against the target in big-endian order.
    bool winner = true;
    for (int i = 0; i < 8; ++i) {
        uint32_t hash_be = __byte_perm(outer_cv[i], 0, 0x0123);
        if (hash_be < s_target[i]) break;
        if (hash_be > s_target[i]) {
            winner = false;
            break;
        }
    }

    if (winner) {
        unsigned long long int expected = 0xFFFFFFFFFFFFFFFFULL;
        unsigned long long int desired  = static_cast<unsigned long long int>(nonce);
        atomicCAS(reinterpret_cast<unsigned long long int*>(d_found_nonce), expected, desired);
        *d_found_flag = 1;
    }
}

void launch_alph_mine_kernel(
    const uint8_t* d_blob,       // raw blob bytes (302 bytes, device ptr)
    uint64_t base_nonce,
    uint32_t num_nonces,
    const uint8_t* d_target_u8,  // 32 bytes big-endian, device ptr
    int64_t* d_found_nonce,
    uint8_t* d_found_flag,
    cudaStream_t stream) {

    // Target is provided as uint8[32]. Reinterpret as uint32[8] directly.
    // The kernel compares byte-swapped hash words against these uint32 words,
    // so we reinterpret in place (no copy needed — memory layout is identical).
    const uint32_t* d_target = reinterpret_cast<const uint32_t*>(d_target_u8);

    // Blob must be padded to PADDED_BYTES (320) with zeros, as uint32 words.
    // We reinterpret the blob pointer — the caller guarantees 320 bytes allocated.
    const uint32_t* d_blob_words = reinterpret_cast<const uint32_t*>(d_blob);

    constexpr int THREADS = 256;
    int blocks = (num_nonces + THREADS - 1) / THREADS;


    alph_mine_kernel<<<blocks, THREADS, 0, stream>>>(
        d_blob_words, base_nonce, num_nonces,
        d_target, d_found_nonce, d_found_flag);
}

} // namespace mining
} // namespace pearl
