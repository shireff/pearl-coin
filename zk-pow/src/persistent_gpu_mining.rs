//! Persistent GPU mining pipeline.
//!
//! This module provides a persistent GPU mining pipeline that keeps the GPU worker
//! resident across multiple mining batches, eliminating kernel launch overhead.

use anyhow::Result;

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
use pyo3::prelude::*;

/// Check if persistent GPU mining is available.
#[inline]
pub fn persistent_gpu_mining_available() -> bool {
    cfg!(all(feature = "gpu_prove", feature = "pyo3"))
}

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
fn call_create_persistent_pipeline(max_jackpots: i64) -> Result<pyo3::Py<pyo3::PyAny>> {
    Python::with_gil(|py| {
        let pearl_gemm = py.import("pearl_gemm_cuda")
            .map_err(|e| anyhow::anyhow!("Failed to import pearl_gemm_cuda: {}", e))?;

        let result = pearl_gemm
            .call_method1("create_persistent_pipeline", (max_jackpots,))
            .map_err(|e| anyhow::anyhow!("Failed to create persistent pipeline: {}", e))?;

        Ok(result.into())
    })
}

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
fn call_destroy_persistent_pipeline(state: &pyo3::Py<pyo3::PyAny>) -> Result<()> {
    Python::with_gil(|py| {
        state.call_method0(py, "destroy")
            .map_err(|e| anyhow::anyhow!("Failed to destroy persistent pipeline: {}", e))?;
        Ok(())
    })
}

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
fn call_pipeline_set_buffers(
    state: &pyo3::Py<pyo3::PyAny>,
    a_noised: &pyo3::Py<pyo3::PyAny>,
    b_noised_t: &pyo3::Py<pyo3::PyAny>,
    a_rows: &pyo3::Py<pyo3::PyAny>,
    b_cols: &pyo3::Py<pyo3::PyAny>,
) -> Result<()> {
    Python::with_gil(|py| {
        state.call_method1(py, "set_buffers", (a_noised, b_noised_t, a_rows, b_cols))
            .map_err(|e| anyhow::anyhow!("Failed to set pipeline buffers: {}", e))?;
        Ok(())
    })
}

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
fn call_pipeline_enqueue_jobs(
    state: &pyo3::Py<pyo3::PyAny>,
    jobs: &pyo3::Py<pyo3::PyAny>,
    num_jobs: i64,
) -> Result<()> {
    Python::with_gil(|py| {
        state.call_method1(py, "enqueue_jobs", (jobs, num_jobs))
            .map_err(|e| anyhow::anyhow!("Failed to enqueue jobs: {}", e))?;
        Ok(())
    })
}

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
fn call_pipeline_wait_for_completion(state: &pyo3::Py<pyo3::PyAny>) -> Result<()> {
    Python::with_gil(|py| {
        state.call_method0(py, "wait_for_completion")
            .map_err(|e| anyhow::anyhow!("Failed to wait for completion: {}", e))?;
        Ok(())
    })
}

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
fn call_launch_persistent_worker(state: &pyo3::Py<pyo3::PyAny>) -> Result<()> {
    Python::with_gil(|py| {
        state.call_method0(py, "launch_worker")
            .map_err(|e| anyhow::anyhow!("Failed to launch persistent worker: {}", e))?;
        Ok(())
    })
}

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
fn call_stop_persistent_worker(state: &pyo3::Py<pyo3::PyAny>) -> Result<()> {
    Python::with_gil(|py| {
        state.call_method0(py, "stop_worker")
            .map_err(|e| anyhow::anyhow!("Failed to stop persistent worker: {}", e))?;
        Ok(())
    })
}

// ============================================================================
// Public API
// ============================================================================

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
pub struct PersistentGpuMiningSession {
    state: Option<pyo3::Py<pyo3::PyAny>>,
    max_jackpots: i64,
}

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
impl PersistentGpuMiningSession {
    /// Create a new persistent GPU mining session.
    pub fn new(max_jackpots: i64) -> Result<Self> {
        let state = call_create_persistent_pipeline(max_jackpots)?;
        Ok(Self {
            state: Some(state),
            max_jackpots,
        })
    }

    /// Set global data buffers for the pipeline.
    pub fn set_buffers(
        &self,
        a_noised: &pyo3::Py<pyo3::PyAny>,
        b_noised_t: &pyo3::Py<pyo3::PyAny>,
        a_rows: &pyo3::Py<pyo3::PyAny>,
        b_cols: &pyo3::Py<pyo3::PyAny>,
    ) -> Result<()> {
        let state = self.state.as_ref().ok_or_else(|| anyhow::anyhow!("Pipeline not initialized"))?;
        call_pipeline_set_buffers(state, a_noised, b_noised_t, a_rows, b_cols)
    }

    /// Launch the persistent GPU worker kernel.
    pub fn launch_worker(&self) -> Result<()> {
        let state = self.state.as_ref().ok_or_else(|| anyhow::anyhow!("Pipeline not initialized"))?;
        call_launch_persistent_worker(state)
    }

    /// Stop the persistent GPU worker kernel.
    pub fn stop_worker(&self) -> Result<()> {
        let state = self.state.as_ref().ok_or_else(|| anyhow::anyhow!("Pipeline not initialized"))?;
        call_stop_persistent_worker(state)
    }

    /// Enqueue jobs for GPU processing.
    pub fn enqueue_jobs(&self, jobs: &pyo3::Py<pyo3::PyAny>, num_jobs: i64) -> Result<()> {
        let state = self.state.as_ref().ok_or_else(|| anyhow::anyhow!("Pipeline not initialized"))?;
        call_pipeline_enqueue_jobs(state, jobs, num_jobs)
    }

    /// Wait for all queued jobs to complete.
    pub fn wait_for_completion(&self) -> Result<()> {
        let state = self.state.as_ref().ok_or_else(|| anyhow::anyhow!("Pipeline not initialized"))?;
        call_pipeline_wait_for_completion(state)
    }
}

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
impl Drop for PersistentGpuMiningSession {
    fn drop(&mut self) {
        if let Some(state) = self.state.take() {
            let _ = call_destroy_persistent_pipeline(&state);
        }
    }
}
