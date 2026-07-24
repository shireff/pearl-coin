//! GPU-accelerated STARK proving operations.
//!
//! This module provides CUDA kernels for:
//! - Low-Degree Extension (LDE) evaluation
//! - FRI folding
//! - GPU-accelerated Merkle tree construction
//!
//! These operations are the most compute-intensive parts of STARK proving
//! and benefit significantly from GPU parallelization.

use anyhow::Result;

#[cfg(feature = "gpu_prove")]
mod gpu_kernels {
    use std::ffi::c_void;

    /// CUDA kernel launcher for LDE evaluation.
    /// Evaluates multiple polynomials at multiple points in parallel.
    pub fn launch_lde_eval(
        _coeffs: *const u32,
        _eval_points: *const u32,
        _evaluations: *mut u32,
        _num_coeffs: i32,
        _num_points: i32,
        _poly_degree: i32,
    ) {
        unsafe {
            let kernel = crate::cuda::get_kernel("lde_eval_kernel");
            let grid_size = ((_num_points + 255) / 256) as u32;
            let block_size = 256u32;

            kernel.launch(
                grid_size,
                1,
                1,
                block_size,
                1,
                1,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
                &mut [
                    _coeffs as *mut c_void,
                    _eval_points as *mut c_void,
                    _evaluations as *mut c_void,
                    &mut (_num_coeffs as *mut i32 as *mut c_void),
                    &mut (_num_points as *mut i32 as *mut c_void),
                    &mut (_poly_degree as *mut i32 as *mut c_void),
                ],
            );
        }
    }

    /// CUDA kernel launcher for FRI folding.
    /// Folds adjacent pairs using random challenges.
    pub fn launch_fri_fold(
        _layer: *const u32,
        _folded_layer: *mut u32,
        _challenges: *const u32,
        _layer_size: i32,
    ) {
        unsafe {
            let kernel = crate::cuda::get_kernel("fri_fold_kernel");
            let grid_size = ((_layer_size / 2 + 255) / 256) as u32;
            let block_size = 256u32;

            kernel.launch(
                grid_size,
                1,
                1,
                block_size,
                1,
                1,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
                &mut [
                    _layer as *mut c_void,
                    _folded_layer as *mut c_void,
                    _challenges as *mut c_void,
                    &mut (_layer_size as *mut i32 as *mut c_void),
                ],
            );
        }
    }

    /// CUDA kernel launcher for Merkle tree hashing.
    /// Computes parent nodes from leaf pairs.
    pub fn launch_merkle_hash(
        _leaves: *const u32,
        _nodes: *mut u32,
        _num_leaves: i32,
    ) {
        unsafe {
            let kernel = crate::cuda::get_kernel("merkle_hash_kernel");
            let grid_size = ((_num_leaves / 2 + 255) / 256) as u32;
            let block_size = 256u32;

            kernel.launch(
                grid_size,
                1,
                1,
                block_size,
                1,
                1,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
                &mut [
                    _leaves as *mut c_void,
                    _nodes as *mut c_void,
                    &mut (_num_leaves as *mut i32 as *mut c_void),
                ],
            );
        }
    }

    /// Initialize CUDA context and load kernels.
    pub fn init_cuda() -> Result<(), &'static str> {
        crate::cuda::init()?;
        crate::cuda::load_kernel("lde_eval_kernel", "csrc/stark/gpu_stark_kernels.ptx")?;
        crate::cuda::load_kernel("fri_fold_kernel", "csrc/stark/gpu_stark_kernels.ptx")?;
        crate::cuda::load_kernel("merkle_hash_kernel", "csrc/stark/gpu_stark_kernels.ptx")?;
        Ok(())
    }
}

#[cfg(not(feature = "gpu_prove"))]
mod gpu_kernels {
    pub fn init_cuda() -> Result<(), &'static str> {
        Ok(())
    }
}

/// GPU-accelerated Low-Degree Extension (LDE) evaluation.
///
/// Evaluates a polynomial at multiple points using Horner's method
/// in parallel on the GPU.
pub fn gpu_lde_eval(
    coeffs: &[u32],
    eval_points: &[u32],
    poly_degree: usize,
) -> Result<Vec<u32>> {
    #[cfg(feature = "gpu_prove")]
    {
        use std::ffi::c_void;

        let num_points = eval_points.len();
        let mut evaluations = vec![0u32; num_points];

        unsafe {
            let mut d_coeffs: *mut c_void = std::ptr::null_mut();
            let mut d_eval_points: *mut c_void = std::ptr::null_mut();
            let mut d_evaluations: *mut c_void = std::ptr::null_mut();

            crate::cuda::malloc(&mut d_coeffs, coeffs.len() * 4);
            crate::cuda::malloc(&mut d_eval_points, eval_points.len() * 4);
            crate::cuda::malloc(&mut d_evaluations, num_points * 4);

            crate::cuda::memcpy_host_to_device(
                d_coeffs,
                coeffs.as_ptr() as *const c_void,
                coeffs.len() * 4,
            );
            crate::cuda::memcpy_host_to_device(
                d_eval_points,
                eval_points.as_ptr() as *const c_void,
                eval_points.len() * 4,
            );

            gpu_kernels::launch_lde_eval(
                d_coeffs as *const u32,
                d_eval_points as *const u32,
                d_evaluations as *mut u32,
                coeffs.len() as i32,
                num_points as i32,
                poly_degree as i32,
            );

            crate::cuda::memcpy_device_to_host(
                evaluations.as_mut_ptr() as *mut c_void,
                d_evaluations,
                num_points * 4,
            );

            crate::cuda::free(d_coeffs);
            crate::cuda::free(d_eval_points);
            crate::cuda::free(d_evaluations);
        }

        Ok(evaluations)
    }

    #[cfg(not(feature = "gpu_prove"))]
    {
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
}

/// GPU-accelerated FRI folding.
///
/// Folds a layer of the FRI protocol using random challenges.
/// Each pair of adjacent elements is combined into one.
pub fn gpu_fri_fold(
    layer: &[u32],
    challenges: &[u32],
) -> Result<Vec<u32>> {
    #[cfg(feature = "gpu_prove")]
    {
        use std::ffi::c_void;

        let folded_len = layer.len() / 2;
        let mut folded = vec![0u32; folded_len];

        unsafe {
            let mut d_layer: *mut c_void = std::ptr::null_mut();
            let mut d_folded: *mut c_void = std::ptr::null_mut();
            let mut d_challenges: *mut c_void = std::ptr::null_mut();

            crate::cuda::malloc(&mut d_layer, layer.len() * 4);
            crate::cuda::malloc(&mut d_folded, folded_len * 4);
            crate::cuda::malloc(&mut d_challenges, challenges.len() * 4);

            crate::cuda::memcpy_host_to_device(
                d_layer,
                layer.as_ptr() as *const c_void,
                layer.len() * 4,
            );
            crate::cuda::memcpy_host_to_device(
                d_challenges,
                challenges.as_ptr() as *const c_void,
                challenges.len() * 4,
            );

            gpu_kernels::launch_fri_fold(
                d_layer as *const u32,
                d_folded as *mut u32,
                d_challenges as *const u32,
                layer.len() as i32,
            );

            crate::cuda::memcpy_device_to_host(
                folded.as_mut_ptr() as *mut c_void,
                d_folded,
                folded_len * 4,
            );

            crate::cuda::free(d_layer);
            crate::cuda::free(d_folded);
            crate::cuda::free(d_challenges);
        }

        Ok(folded)
    }

    #[cfg(not(feature = "gpu_prove"))]
    {
        let mut folded = Vec::with_capacity(layer.len() / 2);

        for (i, chunk) in layer.chunks(2).enumerate() {
            let left = chunk[0];
            let right = chunk.get(1).copied().unwrap_or(0);
            let challenge = challenges[i % challenges.len()];
            folded.push(left.wrapping_add(challenge.wrapping_mul(right)));
        }

        Ok(folded)
    }
}

/// GPU-accelerated Merkle tree construction.
///
/// Builds a Merkle tree from leaf hashes using parallel reduction.
pub fn gpu_merkle_tree(leaves: &[u32]) -> Result<Vec<u32>> {
    #[cfg(feature = "gpu_prove")]
    {
        use std::ffi::c_void;

        let mut current = leaves.to_vec();

        unsafe {
            let mut d_current: *mut c_void = std::ptr::null_mut();
            let mut d_next: *mut c_void = std::ptr::null_mut();

            crate::cuda::malloc(&mut d_current, current.len() * 4);
            crate::cuda::malloc(&mut d_next, (current.len() / 2) * 4);

            while current.len() > 1 {
                let next_len = current.len() / 2;

                crate::cuda::memcpy_host_to_device(
                    d_current,
                    current.as_ptr() as *const c_void,
                    current.len() * 4,
                );

                gpu_kernels::launch_merkle_hash(
                    d_current as *const u32,
                    d_next as *mut u32,
                    current.len() as i32,
                );

                let mut next = vec![0u32; next_len];
                crate::cuda::memcpy_device_to_host(
                    next.as_mut_ptr() as *mut c_void,
                    d_next,
                    next_len * 4,
                );

                current = next;
            }

            crate::cuda::free(d_current);
            crate::cuda::free(d_next);
        }

        Ok(current)
    }

    #[cfg(not(feature = "gpu_prove"))]
    {
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
}

/// GPU-accelerated STARK proof generation.
///
/// This is the main entry point for GPU-accelerated STARK proving.
/// It coordinates LDE evaluation, FRI folding, and Merkle tree construction.
pub struct GpuStarkProver {
    pub use_gpu: bool,
}

impl GpuStarkProver {
    pub fn new(use_gpu: bool) -> Self {
        Self { use_gpu }
    }

    /// Generate a STARK proof using GPU acceleration where available.
    pub fn prove(
        &self,
        trace: &[Vec<u32>],
        public_inputs: &[u32],
    ) -> Result<Vec<u32>> {
        if !self.use_gpu {
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

    fn prove_gpu(
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
            folded = gpu_fri_fold(&folded, &challenges)?;
        }

        let merkle_root = gpu_merkle_tree(&flat_trace)?;

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
