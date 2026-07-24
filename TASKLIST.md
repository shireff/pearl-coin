# Pearl Project - Optimization Task List

## Critical

| Task ID | Title | File(s) | Function(s) | Reason | Expected Improvement | Est. Difficulty | Dependencies | Status |
|---------|-------|---------|-------------|--------|---------------------|-----------------|--------------|--------|
| C-01 | Fix CUDA version mismatch in Colab build | `install_colab.sh`, `miner/pearl-gemm/build_inplace.py` | N/A | Colab nvcc 12.8 vs torch CUDA 13.0 causes build failure | Enables build on Colab | Low | None | [x] Completed |

## High Priority

| Task ID | Title | File(s) | Function(s) | Reason | Expected Improvement | Est. Difficulty | Dependencies | Status |
|---------|-------|---------|-------------|--------|---------------------|-----------------|--------------|--------|
| H-01 | Flatten matrix storage in mining hot path | `zk-pow/src/ffi/mine.rs` | `try_mine_one`, `try_mine_one_moe`, `build_matrix_proof`, `flatten_matrix` | `Vec<Vec<i8>>` causes heap fragmentation and poor cache locality; flat arrays improve L1/L2 hit rate | 2-3× cache efficiency | Low | None | [x] Completed |
| H-02 | Parallel CPU jackpot search with rayon | `zk-pow/src/ffi/mine.rs` | `try_mine_one`, `try_mine_one_moe` | Single-threaded search underutilizes modern CPUs; rayon parallel iter scales with core count | N× cores | Low | H-01 | [x] Completed |
| H-03 | Pre-allocate reusable buffers in hot loop | `zk-pow/src/ffi/mine.rs` | `try_mine_one`, `try_mine_one_moe` | Repeated Vec allocation in hot loop causes allocator pressure and cache thrashing | 1.5-2× | Low | H-01 | [x] Completed |
| H-04 | SIMD-optimized BLAKE3 batching | `pearl-blake3/src/` | `blake3_digest`, `MerkleTree` | 7+ sequential BLAKE3 calls per attempt; SIMD + batching reduces hash overhead | 3-5× hash throughput | Medium | H-01 | [ ] Not Started |

## Medium Priority

| Task ID | Title | File(s) | Function(s) | Reason | Expected Improvement | Est. Difficulty | Dependencies | Status |
|---------|-------|---------|-------------|--------|---------------------|-----------------|--------------|--------|
| M-01 | GPU offload of tile evaluation | `zk-pow/src/ffi/mine.rs`, `miner/pearl-gemm/csrc/` | `try_mine_one` jackpot loop, new CUDA kernels | Core mining loop runs on CPU; GPU can evaluate tiles 100-1000× faster | 100-1000× | High | H-01, H-02 | [ ] Not Started |
| M-02 | Fused GEMM+hash+accumulation kernel | `miner/pearl-gemm/csrc/` | New CUDA kernel | Separate kernel launches waste cycles; fused pipeline reduces global memory traffic | 1.5-2× | High | M-01 | [ ] Not Started |
| M-03 | Persistent GPU kernel for mining | `miner/vllm-miner/src/` | New persistent kernel | Per-job launch overhead dominates at small job sizes; persistent kernel eliminates it | 2-10× GPU utilization | High | M-01 | [ ] Not Started |
| M-04 | Shared memory tiling for GEMM | `miner/pearl-gemm/csrc/gemm/` | `pearl_gemm_api.cpp`, CUTLASS instantiations | Current GEMM may have suboptimal shared memory usage; tiling reduces global memory transactions | 2-4× GPU FLOP/s | Medium | None | [ ] Not Started |
| M-05 | Adaptive STARK degree selection | `zk-pow/src/circuit/` | `pearl_circuit.rs`, `pearl_preprocess.rs` | Fixed STARK degree wastes trace space; adaptive degree reduces proving time | 2-4× prove time | Medium | None | [ ] Not Started |
| M-06 | Incremental Merkle tree for proofs | `pearl-blake3/src/` | `MerkleTree`, `get_multileaf_proof` | Full Merkle tree rebuild per proof; incremental update for single-leaf proofs is faster | 2-3× proof generation | Medium | H-01 | [ ] Not Started |

## Low Priority

| Task ID | Title | File(s) | Function(s) | Reason | Expected Improvement | Est. Difficulty | Dependencies | Status |
|---------|-------|---------|-------------|--------|---------------------|-----------------|--------------|--------|
| L-01 | Double buffering CPU/GPU pipeline | `miner/vllm-miner/src/` | Pipeline orchestration | CPU prepares next job while GPU processes current; overlaps latency | 1.5-2× pipeline | Medium | M-03 | [ ] Not Started |
| L-02 | CUDA Graph capture | `miner/vllm-miner/src/` | Kernel launch sequence | Captures GPU pipeline as graph; eliminates CPU launch overhead for repeated jobs | 1.2-1.5× | Medium | M-03 | [ ] Not Started |
| L-03 | Zero-copy proof buffer sharing | `zk-pow/bindings/go/src/` | FFI copy routines | CFFI copy overhead between Rust and Go; zero-copy reduces latency | 1.1-1.3× | Low | None | [ ] Not Started |
| L-04 | int7 GEMM kernel optimization | `miner/pearl-gemm/csrc/gemm/` | `noise_generation.cu`, `denoise_converter.cu` | Better tensor core utilization for int7→int32 MMA on Blackwell | 1.2-1.5× | Medium | None | [ ] Not Started |
| L-05 | GPU-accelerated STARK proving | `zk-pow/src/api/prove.rs` | `prove_block` | STARK proving is the ultimate bottleneck; GPU acceleration is research-level | 10-100× | Very High | M-01 | [ ] Not Started |

## Notes

- **H-01 (Flatten matrix storage)** is the highest ROI, lowest risk optimization. It is the foundation for H-02, H-03, and M-01.
- **M-01 (GPU offload)** is the highest-impact optimization but has the highest risk and complexity.
- **L-05 (GPU STARK proving)** is the ultimate bottleneck solver but is research-grade.
- All optimizations must preserve correctness, consensus behavior, deterministic outputs, and cryptographic correctness.
