#pragma once

#include <cstdint>
#include <cstddef>

namespace pearl {
namespace mining {

// Alephium Blake3 PoW mining kernel.
// blob_words: zero-padded blob as uint32[80] (320 bytes, only first 302 are real data)
// base_nonce: starting nonce for this batch
// num_nonces: number of nonces this launch covers
// d_target: 32-byte target as uint8[32] (big-endian)
// d_found_nonce: output int64 — winning nonce or -1
// d_found_flag: output uint8 — 1 if winner found
void launch_alph_mine_kernel(
    const uint8_t* d_blob,
    uint64_t base_nonce,
    uint32_t num_nonces,
    const uint8_t* d_target_u8,
    int64_t* d_found_nonce,
    uint8_t* d_found_flag,
    cudaStream_t stream);

} // namespace mining
} // namespace pearl
