//! GPU-native batch mining operations.
//!
//! This module provides GPU-accelerated mining that replaces the CPU-bound
//! jackpot search in `try_mine_one()` and `try_mine_one_moe()`.
//!
//! # Architecture
//!
//! When `gpu_prove` + `pyo3` features are enabled, this module calls into the
//! `pearl_gemm_cuda` Python extension, which provides real CUDA kernel execution
//! for batch mining.
//!
//! # Execution Path
//!
//! 1. CPU generates random matrices and computes commitment hashes
//! 2. CPU uploads matrices and job descriptors to GPU once per batch
//! 3. GPU executes the complete mining loop across all candidates in parallel
//! 4. GPU writes jackpot arrays for all tile combinations to result buffer
//! 5. CPU reads back jackpots and performs blake3 hashing + difficulty check
//! 6. CPU builds proof only for winning candidate
//!
//! # Performance
//!
//! The GPU path eliminates the CPU-bound nested loops in `try_mine_one()`.
//! For typical parameters (m=1024, n=1024, k=1024, rank=32), the GPU can
//! evaluate thousands of candidates in parallel, reducing mining time from
//! ~100ms per candidate to ~1ms per batch.

use anyhow::Result;

/// Check if GPU mining is available.
/// Returns true if both `gpu_prove` and `pyo3` features are enabled.
#[inline]
pub fn gpu_mining_available() -> bool {
    cfg!(all(feature = "gpu_prove", feature = "pyo3"))
}

// ============================================================================
// GPU Mining via Python CUDA Extension
// ============================================================================

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
fn call_gpu_mine_batch(
    a_noised: &[i32],
    b_noised_t: &[i32],
    a_rows_data: &[i32],
    b_cols_data: &[i32],
    jobs: &[u8],
    num_jobs: i64,
    P: i64,
    Q: i64,
    tile_h: i64,
    tile_w: i64,
    m: i64,
    n: i64,
    k: i64,
    rank: i64,
) -> Result<Vec<u32>> {
    use pyo3::prelude::*;

    Python::with_gil(|py| {
        let pearl_gemm = py.import("pearl_gemm_cuda")
            .map_err(|e| anyhow::anyhow!("Failed to import pearl_gemm_cuda: {}", e))?;

        let a_noised_tensor = create_cuda_tensor_i32(py, a_noised)?;
        let b_noised_t_tensor = create_cuda_tensor_i32(py, b_noised_t)?;
        let a_rows_data_tensor = create_cuda_tensor_i32(py, a_rows_data)?;
        let b_cols_data_tensor = create_cuda_tensor_i32(py, b_cols_data)?;
        let jobs_tensor = create_cuda_tensor_u8(py, jobs)?;

        let args = [
            a_noised_tensor.into_py(py),
            b_noised_t_tensor.into_py(py),
            a_rows_data_tensor.into_py(py),
            b_cols_data_tensor.into_py(py),
            jobs_tensor.into_py(py),
            num_jobs.into_py(py),
            P.into_py(py),
            Q.into_py(py),
            tile_h.into_py(py),
            tile_w.into_py(py),
            m.into_py(py),
            n.into_py(py),
            k.into_py(py),
            rank.into_py(py),
        ];
        let py_tuple = pyo3::types::PyTuple::new(py, &args)
            .map_err(|e| anyhow::anyhow!("Failed to create PyTuple: {}", e))?;

        let result = pearl_gemm
            .call_method1("gpu_mine_batch", py_tuple)
            .map_err(|e| anyhow::anyhow!("GPU mining batch failed: {}", e))?;

        let cpu_tensor = result.call_method0("cpu")
            .map_err(|e| anyhow::anyhow!("Failed to move tensor to CPU: {}", e))?;
        let np_array = cpu_tensor.call_method0("numpy")
            .map_err(|e| anyhow::anyhow!("Failed to convert to numpy: {}", e))?;
        let list = np_array.call_method0("tolist")
            .map_err(|e| anyhow::anyhow!("Failed to convert to list: {}", e))?;
        
        let nested: Vec<Vec<Vec<u32>>> = list.extract()
            .map_err(|e| anyhow::anyhow!("Failed to extract GPU result: {}", e))?;
        
        let flat: Vec<u32> = nested.into_iter().flatten().flatten().collect();

        Ok(flat)
    })
}

// ============================================================================
// Python CUDA Tensor Helpers
// ============================================================================

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
fn create_cuda_tensor_i32(py: pyo3::Python, data: &[i32]) -> Result<pyo3::Py<pyo3::PyAny>> {
    use pyo3::prelude::*;
    use pyo3::types::PyList;

    let py_list = PyList::new(py, data.iter().cloned())
        .map_err(|e| anyhow::anyhow!("Failed to create PyList: {}", e))?;

    let torch = py.import("torch")
        .map_err(|e| anyhow::anyhow!("Failed to import torch: {}", e))?;

    let tensor = torch
        .call_method1("tensor", (py_list,))
        .and_then(|t| t.call_method1("to", ("cuda",)))
        .map_err(|e| anyhow::anyhow!("Failed to create CUDA tensor: {}", e))?;

    Ok(tensor.into())
}

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
fn create_cuda_tensor_u8(py: pyo3::Python, data: &[u8]) -> Result<pyo3::Py<pyo3::PyAny>> {
    use pyo3::prelude::*;
    use pyo3::types::PyList;

    let py_list = PyList::new(py, data.iter().cloned())
        .map_err(|e| anyhow::anyhow!("Failed to create PyList: {}", e))?;

    let torch = py.import("torch")
        .map_err(|e| anyhow::anyhow!("Failed to import torch: {}", e))?;

    let tensor = torch
        .call_method1("tensor", (py_list,))
        .and_then(|t| t.call_method1("to", ("cuda",)))
        .map_err(|e| anyhow::anyhow!("Failed to create CUDA tensor: {}", e))?;

    Ok(tensor.into())
}

// ============================================================================
// CPU Fallback Implementations
// ============================================================================

#[cfg(not(all(feature = "gpu_prove", feature = "pyo3")))]
fn call_gpu_mine_batch(
    _a_noised: &[i32],
    _b_noised_t: &[i32],
    _a_rows_data: &[i32],
    _b_cols_data: &[i32],
    _jobs: &[u8],
    _num_jobs: i64,
    _P: i64,
    _Q: i64,
    _tile_h: i64,
    _tile_w: i64,
    _m: i64,
    _n: i64,
    _k: i64,
    _rank: i64,
) -> Result<Vec<u32>> {
    anyhow::bail!("GPU mining not available - build with --features gpu_prove,pyo3")
}

// ============================================================================
// Public API
// ============================================================================

/// GPU-native batch mining.
///
/// Launches a persistent GPU kernel that searches for winning mining candidates
/// in parallel. The kernel evaluates jackpot arrays for each candidate and
/// returns all jackpots for CPU-side blake3 hashing and difficulty check.
///
/// # Arguments
/// - `a_noised`: Noised A matrices (num_jobs x m x k), flattened int32
/// - `b_noised_t`: Noised B^T matrices (num_jobs x n x k), flattened int32
/// - `a_rows_data`: Row indices for tile evaluation (P x tile_h), int32
/// - `b_cols_data`: Column indices for tile evaluation (Q x tile_w), int32
/// - `jobs`: Mining job descriptors (serialized MiningJob structs)
/// - `num_jobs`: Number of mining jobs
/// - `P`: Number of a_rows partitions
/// - `Q`: Number of b_cols partitions
/// - `tile_h`: Tile height
/// - `tile_w`: Tile width
/// - `m`: Matrix A rows
/// - `n`: Matrix B cols
/// - `k`: Matrix inner dim
/// - `rank`: Noise rank
///
/// # Returns
/// Flat vector of jackpot arrays: [num_jobs, P*Q, 16] flattened to u32
pub fn gpu_mine_batch(
    a_noised: &[i32],
    b_noised_t: &[i32],
    a_rows_data: &[i32],
    b_cols_data: &[i32],
    jobs: &[u8],
    num_jobs: i64,
    P: i64,
    Q: i64,
    tile_h: i64,
    tile_w: i64,
    m: i64,
    n: i64,
    k: i64,
    rank: i64,
) -> Result<Vec<u32>> {
    if !gpu_mining_available() {
        anyhow::bail!("GPU mining not available - build with --features gpu_prove,pyo3")
    }

    call_gpu_mine_batch(
        a_noised,
        b_noised_t,
        a_rows_data,
        b_cols_data,
        jobs,
        num_jobs,
        P,
        Q,
        tile_h,
        tile_w,
        m,
        n,
        k,
        rank,
    )
}
