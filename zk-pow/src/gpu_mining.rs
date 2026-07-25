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

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
use pyo3::prelude::*;

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
    Python::with_gil(|py| {
        let pearl_gemm = py.import("pearl_gemm_cuda")
            .map_err(|e| anyhow::anyhow!("Failed to import pearl_gemm_cuda: {}", e))?;

        let a_noised_tensor = create_cuda_tensor_i32(a_noised)?;
        let b_noised_t_tensor = create_cuda_tensor_i32(b_noised_t)?;
        let a_rows_data_tensor = create_cuda_tensor_i32(a_rows_data)?;
        let b_cols_data_tensor = create_cuda_tensor_i32(b_cols_data)?;
        let jobs_tensor = create_cuda_tensor_u8(jobs)?;

        let args = [
            a_noised_tensor, b_noised_t_tensor,
            a_rows_data_tensor, b_cols_data_tensor,
            jobs_tensor,
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

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
fn call_gpu_jackpot_hash(
    jackpots: &[u32],
    keys: &[u32],
    num_jobs: i64,
    num_combos: i64,
) -> Result<Vec<u32>> {
    use pyo3::prelude::*;

    Python::with_gil(|py| {
        let pearl_gemm = py.import("pearl_gemm_cuda")
            .map_err(|e| anyhow::anyhow!("Failed to import pearl_gemm_cuda: {}", e))?;

        let jackpots_tensor = create_cuda_tensor_u32(jackpots)?;
        let keys_tensor = create_cuda_tensor_u32(keys)?;

        let result = pearl_gemm
            .call_method1("gpu_jackpot_hash", (
                jackpots_tensor, keys_tensor,
                num_jobs, num_combos,
            ))
            .map_err(|e| anyhow::anyhow!("GPU jackpot hash failed: {}", e))?;

        let cpu_tensor = result.call_method0("cpu")
            .map_err(|e| anyhow::anyhow!("Failed to move tensor to CPU: {}", e))?;
        let np_array = cpu_tensor.call_method0("numpy")
            .map_err(|e| anyhow::anyhow!("Failed to convert to numpy: {}", e))?;
        let list = np_array.call_method0("tolist")
            .map_err(|e| anyhow::anyhow!("Failed to convert to list: {}", e))?;
        
        let nested: Vec<Vec<Vec<u32>>> = list.extract()
            .map_err(|e| anyhow::anyhow!("Failed to extract GPU hash result: {}", e))?;
        
        let flat: Vec<u32> = nested.into_iter().flatten().flatten().collect();

        Ok(flat)
    })
}

// ============================================================================
// Public CUDA Tensor Helpers
// ============================================================================

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
pub fn create_cuda_tensor_i32(data: &[i32]) -> Result<pyo3::Py<pyo3::PyAny>> {
    create_cuda_tensor_i32_impl(data)
}

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
pub fn create_cuda_tensor_u32(data: &[u32]) -> Result<pyo3::Py<pyo3::PyAny>> {
    create_cuda_tensor_u32_impl(data)
}

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
pub fn create_cuda_tensor_u8(data: &[u8]) -> Result<pyo3::Py<pyo3::PyAny>> {
    create_cuda_tensor_u8_impl(data)
}

// ============================================================================
// Python CUDA Tensor Helpers
// ============================================================================

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
fn create_cuda_tensor_i32_impl(data: &[i32]) -> Result<pyo3::Py<pyo3::PyAny>> {
    Python::with_gil(|py| {
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
    })
}

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
fn create_cuda_tensor_u32_impl(data: &[u32]) -> Result<pyo3::Py<pyo3::PyAny>> {
    Python::with_gil(|py| {
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
    })
}

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
fn create_cuda_tensor_u8_impl(data: &[u8]) -> Result<pyo3::Py<pyo3::PyAny>> {
    Python::with_gil(|py| {
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
    })
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

#[cfg(not(all(feature = "gpu_prove", feature = "pyo3")))]
fn call_gpu_jackpot_hash(
    _jackpots: &[u32],
    _keys: &[u32],
    _num_jobs: i64,
    _num_combos: i64,
) -> Result<Vec<u32>> {
    anyhow::bail!("GPU jackpot hash not available - build with --features gpu_prove,pyo3")
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

/// GPU-native BLAKE3 jackpot hash.
///
/// Computes BLAKE3 keyed hashes for jackpot arrays on GPU using the
/// fixed CUDA BLAKE3 implementation. This is bit-identical to the
/// CPU blake3 crate's keyed hash mode.
///
/// # Arguments
/// - `jackpots`: Flat vector of jackpot arrays [num_jobs, num_combos, 16] u32
/// - `keys`: Flat vector of BLAKE3 keys [num_jobs, 8] u32
/// - `num_jobs`: Number of mining jobs
/// - `num_combos`: Number of tile combinations (P * Q)
///
/// # Returns
/// Flat vector of BLAKE3 hashes [num_jobs, num_combos, 8] u32
pub fn gpu_jackpot_hash(
    jackpots: &[u32],
    keys: &[u32],
    num_jobs: i64,
    num_combos: i64,
) -> Result<Vec<u32>> {
    if !gpu_mining_available() {
        anyhow::bail!("GPU mining not available - build with --features gpu_prove,pyo3")
    }

    call_gpu_jackpot_hash(jackpots, keys, num_jobs, num_combos)
}

// ============================================================================
// CUDA Graph API
// ============================================================================

/// Opaque handle for CUDA Graph state
pub struct MiningGraphState {
    #[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
    inner: pyo3::Py<pyo3::PyAny>,
}

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
fn call_create_mining_graph(
    num_candidates: i64,
    num_combos: i64,
    tile_h: i64,
    tile_w: i64,
    m: i64,
    n: i64,
    k: i64,
    rank: i64,
) -> Result<pyo3::Py<pyo3::PyAny>> {
    Python::with_gil(|py| {
        let pearl_gemm = py.import("pearl_gemm_cuda")
            .map_err(|e| anyhow::anyhow!("Failed to import pearl_gemm_cuda: {}", e))?;

        let result = pearl_gemm
            .call_method1("create_mining_graph", (
                num_candidates, num_combos,
                tile_h, tile_w,
                m, n, k, rank,
            ))
            .map_err(|e| anyhow::anyhow!("Failed to create mining graph: {}", e))?;

        Ok(result.into())
    })
}

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
fn call_destroy_mining_graph(state: &pyo3::Py<pyo3::PyAny>) -> Result<()> {
    Python::with_gil(|py| {
        state.call_method0(py, "destroy")
            .map_err(|e| anyhow::anyhow!("Failed to destroy mining graph: {}", e))?;
        Ok(())
    })
}

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
fn call_update_mining_graph(
    state: &pyo3::Py<pyo3::PyAny>,
    num_candidates: i64,
    num_combos: i64,
    tile_h: i64,
    tile_w: i64,
    m: i64,
    n: i64,
    k: i64,
    rank: i64,
) -> Result<()> {
    Python::with_gil(|py| {
        state.call_method1(py, "update", (
            num_candidates, num_combos,
            tile_h, tile_w,
            m, n, k, rank,
        ))
            .map_err(|e| anyhow::anyhow!("Failed to update mining graph: {}", e))?;
        Ok(())
    })
}

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
fn call_launch_mining_graph(
    state: &pyo3::Py<pyo3::PyAny>,
    a_noised: &pyo3::Py<pyo3::PyAny>,
    b_noised_t: &pyo3::Py<pyo3::PyAny>,
    a_rows: &pyo3::Py<pyo3::PyAny>,
    b_cols: &pyo3::Py<pyo3::PyAny>,
    jobs: &pyo3::Py<pyo3::PyAny>,
    jackpots: &pyo3::Py<pyo3::PyAny>,
    keys: &pyo3::Py<pyo3::PyAny>,
    hashes: &pyo3::Py<pyo3::PyAny>,
) -> Result<()> {
    Python::with_gil(|py| {
        state.call_method1(py, "launch", (
            a_noised, b_noised_t, a_rows, b_cols,
            jobs, jackpots, keys, hashes,
        ))
            .map_err(|e| anyhow::anyhow!("Failed to launch mining graph: {}", e))?;
        Ok(())
    })
}

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
fn call_synchronize_mining_graph(state: &pyo3::Py<pyo3::PyAny>) -> Result<()> {
    Python::with_gil(|py| {
        state.call_method0(py, "synchronize")
            .map_err(|e| anyhow::anyhow!("Failed to synchronize mining graph: {}", e))?;
        Ok(())
    })
}

impl MiningGraphState {
    /// Create a new CUDA Graph for batch mining
    pub fn new(
        num_candidates: i64,
        num_combos: i64,
        tile_h: i64,
        tile_w: i64,
        m: i64,
        n: i64,
        k: i64,
        rank: i64,
    ) -> Result<Self> {
        if !gpu_mining_available() {
            anyhow::bail!("GPU mining not available - build with --features gpu_prove,pyo3")
        }

        let inner = call_create_mining_graph(
            num_candidates, num_combos,
            tile_h, tile_w,
            m, n, k, rank,
        )?;

        Ok(Self { inner })
    }

    /// Update graph with new parameters
    pub fn update(
        &self,
        num_candidates: i64,
        num_combos: i64,
        tile_h: i64,
        tile_w: i64,
        m: i64,
        n: i64,
        k: i64,
        rank: i64,
    ) -> Result<()> {
        call_update_mining_graph(
            &self.inner,
            num_candidates, num_combos,
            tile_h, tile_w,
            m, n, k, rank,
        )
    }

    /// Launch the graph with current data
    pub fn launch(
        &self,
        a_noised: &pyo3::Py<pyo3::PyAny>,
        b_noised_t: &pyo3::Py<pyo3::PyAny>,
        a_rows: &pyo3::Py<pyo3::PyAny>,
        b_cols: &pyo3::Py<pyo3::PyAny>,
        jobs: &pyo3::Py<pyo3::PyAny>,
        jackpots: &pyo3::Py<pyo3::PyAny>,
        keys: &pyo3::Py<pyo3::PyAny>,
        hashes: &pyo3::Py<pyo3::PyAny>,
    ) -> Result<()> {
        call_launch_mining_graph(
            &self.inner,
            a_noised, b_noised_t, a_rows, b_cols,
            jobs, jackpots, keys, hashes,
        )
    }

    /// Wait for graph completion
    pub fn synchronize(&self) -> Result<()> {
        call_synchronize_mining_graph(&self.inner)
    }
}

impl Drop for MiningGraphState {
    fn drop(&mut self) {
        #[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
        {
            let _ = call_destroy_mining_graph(&self.inner);
        }
    }
}

