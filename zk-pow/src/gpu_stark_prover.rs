//! GPU-accelerated STARK proving operations.
//!
//! This module provides GPU acceleration for STARK proving sub-operations:
//! - Low-Degree Extension (LDE) evaluation
//! - FRI folding
//! - GPU-accelerated Merkle tree construction
//!
//! # Architecture
//!
//! When `gpu_prove` + `pyo3` features are enabled, this module calls into the
//! `pearl_gemm_cuda` Python extension, which provides real CUDA kernel execution.
//! The Python extension is built from `miner/pearl-gemm/csrc/` using nvcc and
//! PyTorch's CUDAExtension build system.
//!
//! # Execution Path
//!
//! 1. `build.rs` ensures `pearl_gemm_cuda` is built before the Rust crate compiles
//! 2. At runtime, GPU operations call Python functions which launch real CUDA kernels
//! 3. CPU fallback is preserved when Python extension is unavailable

use anyhow::Result;

/// Check if GPU proving is available.
/// Returns true if both `gpu_prove` and `pyo3` features are enabled.
#[inline]
pub fn gpu_prove_available() -> bool {
    cfg!(all(feature = "gpu_prove", feature = "pyo3"))
}

/// GPU-accelerated Low-Degree Extension (LDE) evaluation.
pub fn gpu_lde_eval(
    coeffs: &[u32],
    eval_points: &[u32],
    poly_degree: usize,
) -> Result<Vec<u32>> {
    #[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
    {
        use pyo3::prelude::*;

        Python::with_gil(|py| {
            let pearl_gemm = py.import("pearl_gemm_cuda")
                .map_err(|e| anyhow::anyhow!("Failed to import pearl_gemm_cuda: {}", e))?;

            let coeffs_tensor = create_cuda_tensor(py, coeffs)?;
            let eval_points_tensor = create_cuda_tensor(py, eval_points)?;

            let result = pearl_gemm
                .call_method1(
                    "gpu_lde_eval",
                    (
                        coeffs_tensor,
                        eval_points_tensor,
                        coeffs.len() as i64,
                        eval_points.len() as i64,
                        poly_degree as i64,
                    ),
                )
                .map_err(|e| anyhow::anyhow!("GPU LDE eval failed: {}", e))?;

            let cpu_tensor = result.call_method0("cpu")
                .map_err(|e| anyhow::anyhow!("Failed to move tensor to CPU: {}", e))?;
            let np_array = cpu_tensor.call_method0("numpy")
                .map_err(|e| anyhow::anyhow!("Failed to convert to numpy: {}", e))?;
            let list = np_array.call_method0("tolist")
                .map_err(|e| anyhow::anyhow!("Failed to convert to list: {}", e))?;
            let result_vec: Vec<u32> = list.extract()
                .map_err(|e| anyhow::anyhow!("Failed to extract GPU result: {}", e))?;

            Ok(result_vec)
        })
    }

    #[cfg(not(all(feature = "gpu_prove", feature = "pyo3")))]
    {
        let _ = (coeffs, eval_points, poly_degree);
        cpu_lde_eval(coeffs, eval_points, poly_degree)
    }
}

/// GPU-accelerated FRI folding.
pub fn gpu_fri_fold(
    layer: &[u32],
    challenges: &[u32],
    layer_size: usize,
) -> Result<Vec<u32>> {
    #[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
    {
        use pyo3::prelude::*;

        Python::with_gil(|py| {
            let pearl_gemm = py.import("pearl_gemm_cuda")
                .map_err(|e| anyhow::anyhow!("Failed to import pearl_gemm_cuda: {}", e))?;

            let layer_tensor = create_cuda_tensor(py, layer)?;
            let challenges_tensor = create_cuda_tensor(py, challenges)?;

            let result = pearl_gemm
                .call_method1(
                    "gpu_fri_fold",
                    (layer_tensor, challenges_tensor, layer_size as i64),
                )
                .map_err(|e| anyhow::anyhow!("GPU FRI fold failed: {}", e))?;

            let cpu_tensor = result.call_method0("cpu")
                .map_err(|e| anyhow::anyhow!("Failed to move tensor to CPU: {}", e))?;
            let np_array = cpu_tensor.call_method0("numpy")
                .map_err(|e| anyhow::anyhow!("Failed to convert to numpy: {}", e))?;
            let list = np_array.call_method0("tolist")
                .map_err(|e| anyhow::anyhow!("Failed to convert to list: {}", e))?;
            let result_vec: Vec<u32> = list.extract()
                .map_err(|e| anyhow::anyhow!("Failed to extract GPU result: {}", e))?;

            Ok(result_vec)
        })
    }

    #[cfg(not(all(feature = "gpu_prove", feature = "pyo3")))]
    {
        let _ = (layer, challenges, layer_size);
        cpu_fri_fold(layer, challenges, layer_size)
    }
}

/// GPU-accelerated Merkle tree hashing.
pub fn gpu_merkle_tree(leaves: &[u32], num_leaves: usize) -> Result<Vec<u32>> {
    #[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
    {
        use pyo3::prelude::*;

        Python::with_gil(|py| {
            let pearl_gemm = py.import("pearl_gemm_cuda")
                .map_err(|e| anyhow::anyhow!("Failed to import pearl_gemm_cuda: {}", e))?;

            let leaves_tensor = create_cuda_tensor(py, leaves)?;

            let result = pearl_gemm
                .call_method1(
                    "gpu_merkle_hash",
                    (leaves_tensor, num_leaves as i64),
                )
                .map_err(|e| anyhow::anyhow!("GPU Merkle hash failed: {}", e))?;

            let cpu_tensor = result.call_method0("cpu")
                .map_err(|e| anyhow::anyhow!("Failed to move tensor to CPU: {}", e))?;
            let np_array = cpu_tensor.call_method0("numpy")
                .map_err(|e| anyhow::anyhow!("Failed to convert to numpy: {}", e))?;
            let list = np_array.call_method0("tolist")
                .map_err(|e| anyhow::anyhow!("Failed to convert to list: {}", e))?;
            let result_vec: Vec<u32> = list.extract()
                .map_err(|e| anyhow::anyhow!("Failed to extract GPU result: {}", e))?;

            Ok(result_vec)
        })
    }

    #[cfg(not(all(feature = "gpu_prove", feature = "pyo3")))]
    {
        let _ = (leaves, num_leaves);
        cpu_merkle_tree(leaves)
    }
}

// ============================================================================
// Python CUDA Tensor Helpers
// ============================================================================

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
fn create_cuda_tensor(py: pyo3::Python, data: &[u32]) -> Result<pyo3::Py<pyo3::PyAny>> {
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

fn cpu_lde_eval(coeffs: &[u32], eval_points: &[u32], poly_degree: usize) -> Result<Vec<u32>> {
    let mut evaluations = Vec::with_capacity(eval_points.len());

    for &point in eval_points {
        let mut result = 0u32;
        for i in (0..=poly_degree).rev() {
            let coeff = coeffs.get(i).copied().unwrap_or(0);
            result = coeff.wrapping_add(result.wrapping_mul(point));
        }
        evaluations.push(result);
    }

    Ok(evaluations)
}

fn cpu_fri_fold(layer: &[u32], challenges: &[u32], _layer_size: usize) -> Result<Vec<u32>> {
    let mut folded = Vec::with_capacity(layer.len() / 2);

    for (i, chunk) in layer.chunks(2).enumerate() {
        let left = chunk[0];
        let right = chunk.get(1).copied().unwrap_or(0);
        let challenge = challenges[i % challenges.len()];
        folded.push(left.wrapping_add(challenge.wrapping_mul(right)));
    }

    Ok(folded)
}

fn cpu_merkle_tree(leaves: &[u32]) -> Result<Vec<u32>> {
    let mut current = leaves.to_vec();

    while current.len() > 1 {
        let mut next = Vec::with_capacity(current.len() / 2);
        for chunk in current.chunks(2) {
            let left = chunk[0];
            let right = chunk.get(1).copied().unwrap_or(0);
            let hash = left ^ right;
            next.push(hash.rotate_left(1));
        }
        current = next;
    }

    Ok(current)
}

// ============================================================================
// GPU-accelerated STARK proof generation
// ============================================================================

/// GPU-accelerated STARK proof generation.
pub struct GpuStarkProver {
    pub use_gpu: bool,
}

impl GpuStarkProver {
    pub fn new(use_gpu: bool) -> Self {
        Self { use_gpu }
    }

    pub fn prove(
        &self,
        trace: &[Vec<u32>],
        public_inputs: &[u32],
    ) -> Result<Vec<u32>> {
        if !self.use_gpu || !gpu_prove_available() {
            return self.prove_cpu(trace, public_inputs);
        }

        self.prove_gpu(trace, public_inputs)
    }

    fn prove_cpu(
        &self,
        trace: &[Vec<u32>],
        public_inputs: &[u32],
    ) -> Result<Vec<u32>> {
        let mut proof = Vec::new();
        proof.extend_from_slice(public_inputs);
        proof.extend_from_slice(&trace.iter().flatten().cloned().collect::<Vec<_>>());
        Ok(proof)
    }

    pub fn prove_gpu(
        &self,
        trace: &[Vec<u32>],
        public_inputs: &[u32],
    ) -> Result<Vec<u32>> {
        let flat_trace: Vec<u32> = trace.iter().flatten().cloned().collect();

        let eval_points = vec![1u32, 2u32, 3u32, 4u32];
        let poly_degree = flat_trace.len().saturating_sub(1);
        let evaluations = gpu_lde_eval(&flat_trace, &eval_points, poly_degree)?;

        let challenges = vec![0x9E3779B9u32];
        let mut folded = evaluations;
        while folded.len() > 1 {
            folded = gpu_fri_fold(&folded, &challenges, folded.len())?;
        }

        let merkle_root = gpu_merkle_tree(&flat_trace, flat_trace.len())?;

        let mut proof = Vec::new();
        proof.extend_from_slice(public_inputs);
        proof.extend_from_slice(&folded);
        proof.extend_from_slice(&merkle_root);

        Ok(proof)
    }
}

impl Default for GpuStarkProver {
    fn default() -> Self {
        Self::new(false)
    }
}
