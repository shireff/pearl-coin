//! GPU-native batch mining operations.
//!
//! This module provides GPU-accelerated mining via direct C-ABI FFI when
//! `gpu_prove` is enabled, eliminating Python and PyTorch overhead.

use anyhow::Result;

// ============================================================================
// C-ABI FFI (Task 2 - primary path, gpu_prove without pyo3)
// ============================================================================

#[cfg(feature = "gpu_prove")]
mod c_abi {
    use super::*;
    use std::os::raw::c_int;

    #[repr(C, align(16))]
    #[derive(Clone, Copy)]
    pub struct MiningJob {
        pub status: u32,
        pub candidate_id: u32,
        pub m: u32,
        pub n: u32,
        pub k: u32,
        pub rank: u32,
        pub tile_h: u32,
        pub tile_w: u32,
        pub num_tiles: u32,
        pub a_offset: u32,
        pub b_offset: u32,
        pub a_rows_offset: u32,
        pub b_cols_offset: u32,
        pub result_offset: u32,
        pub pow_target: [u32; 8],
        pub pow_key: [u32; 8],
        pub a_noise_seed: u32,
        pub b_noise_seed: u32,
    }

    unsafe extern "C" {
        fn pearl_gpu_mine_batch(
            d_a_noised: *const i32,
            d_b_noised_t: *const i32,
            d_a_rows: *const i32,
            d_b_cols: *const i32,
            d_jobs: *const u8,
            d_jackpots: *mut u32,
            d_hashes: *mut u32,
            d_winner_flags: *mut u8,
            num_jobs: u32,
            p: u32,
            q: u32,
            tile_h: u32,
            tile_w: u32,
            m: u32,
            n: u32,
            k: u32,
            rank: u32,
        ) -> c_int;

        fn pearl_gpu_jackpot_hash(
            d_jackpots: *const u32,
            d_keys: *const u32,
            d_hashes: *mut u32,
            num_jobs: u32,
            num_combos: u32,
        ) -> c_int;
    }

    pub fn gpu_mine_batch(
        a_noised: &[i32],
        b_noised_t: &[i32],
        a_rows_data: &[i32],
        b_cols_data: &[i32],
        jobs: &[u8],
        num_jobs: u32,
        p: u32,
        q: u32,
        tile_h: u32,
        tile_w: u32,
        m: u32,
        n: u32,
        k: u32,
        rank: u32,
        jackpots: &mut [u32],
        hashes: &mut [u32],
        winner_flags: &mut [u8],
    ) -> Result<()> {
        unsafe {
            let err = pearl_gpu_mine_batch(
                a_noised.as_ptr(),
                b_noised_t.as_ptr(),
                a_rows_data.as_ptr(),
                b_cols_data.as_ptr(),
                jobs.as_ptr(),
                jackpots.as_mut_ptr(),
                hashes.as_mut_ptr(),
                winner_flags.as_mut_ptr(),
                num_jobs,
                p,
                q,
                tile_h,
                tile_w,
                m,
                n,
                k,
                rank,
            );
            if err != 0 {
                anyhow::bail!("GPU mining failed with CUDA error {}", err);
            }
        }
        Ok(())
    }

    pub fn gpu_jackpot_hash(
        jackpots: &[u32],
        keys: &[u32],
        num_jobs: u32,
        num_combos: u32,
        hashes: &mut [u32],
    ) -> Result<()> {
        unsafe {
            let err = pearl_gpu_jackpot_hash(
                jackpots.as_ptr(),
                keys.as_ptr(),
                hashes.as_mut_ptr(),
                num_jobs,
                num_combos,
            );
            if err != 0 {
                anyhow::bail!("GPU jackpot hash failed with CUDA error {}", err);
            }
        }
        Ok(())
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Check if GPU mining is available.
#[inline]
pub fn gpu_mining_available() -> bool {
    cfg!(feature = "gpu_prove")
}

/// GPU-native batch mining (fused: jackpots + BLAKE3 hashes).
pub fn gpu_mine_batch(
    a_noised: &[i32],
    b_noised_t: &[i32],
    a_rows_data: &[i32],
    b_cols_data: &[i32],
    jobs: &[u8],
    num_jobs: i64,
    p: i64,
    q: i64,
    tile_h: i64,
    tile_w: i64,
    m: i64,
    n: i64,
    k: i64,
    rank: i64,
) -> Result<(Vec<u32>, Vec<u32>)> {
    #[cfg(feature = "gpu_prove")]
    {
        let num_jobs_u = num_jobs as u32;
        let p_u = p as u32;
        let q_u = q as u32;
        let tile_h_u = tile_h as u32;
        let tile_w_u = tile_w as u32;
        let m_u = m as u32;
        let n_u = n as u32;
        let k_u = k as u32;
        let rank_u = rank as u32;

        let num_combos = (p_u as usize) * (q_u as usize);
        let jackpots_size = (num_jobs_u as usize) * num_combos * 16;
        let hashes_size = (num_jobs_u as usize) * num_combos * 8;
        let flags_size = ((num_jobs_u as usize) * num_combos + 7) / 8;

        let mut jackpots = vec![0u32; jackpots_size];
        let mut hashes = vec![0u32; hashes_size];
        let mut winner_flags = vec![0u8; flags_size];

        c_abi::gpu_mine_batch(
            a_noised,
            b_noised_t,
            a_rows_data,
            b_cols_data,
            jobs,
            num_jobs_u,
            p_u,
            q_u,
            tile_h_u,
            tile_w_u,
            m_u,
            n_u,
            k_u,
            rank_u,
            &mut jackpots,
            &mut hashes,
            &mut winner_flags,
        )?;

        Ok((jackpots, hashes))
    }

    #[cfg(not(feature = "gpu_prove"))]
    {
        let _ = (a_noised, b_noised_t, a_rows_data, b_cols_data, jobs, num_jobs, p, q, tile_h, tile_w, m, n, k, rank);
        anyhow::bail!("GPU mining not available - build with --features gpu_prove")
    }
}

/// GPU-native BLAKE3 jackpot hash (legacy, retained for backward compatibility).
pub fn gpu_jackpot_hash(
    jackpots: &[u32],
    keys: &[u32],
    num_jobs: i64,
    num_combos: i64,
) -> Result<Vec<u32>> {
    #[cfg(feature = "gpu_prove")]
    {
        let num_jobs_u = num_jobs as u32;
        let num_combos_u = num_combos as u32;
        let hashes_size = (num_jobs_u as usize) * (num_combos_u as usize) * 8;
        let mut hashes = vec![0u32; hashes_size];

        c_abi::gpu_jackpot_hash(
            jackpots,
            keys,
            num_jobs_u,
            num_combos_u,
            &mut hashes,
        )?;

        Ok(hashes)
    }

    #[cfg(not(feature = "gpu_prove"))]
    {
        let _ = (jackpots, keys, num_jobs, num_combos);
        anyhow::bail!("GPU jackpot hash not available - build with --features gpu_prove")
    }
}

// ============================================================================
// CUDA Graph FFI (Task 5 - real C-ABI bindings, no stubs)
// ============================================================================

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
mod graph_ffi {
    use super::*;
    use std::os::raw::{c_int, c_void};

    #[repr(C)]
    pub struct MiningGraphState {
        pub state: *mut c_void,
    }

    unsafe extern "C" {
        fn pearl_create_mining_graph_state(
            num_candidates: u32,
            num_combos: u32,
            tile_h: u32,
            tile_w: u32,
            m: u32,
            n: u32,
            k: u32,
            rank: u32,
        ) -> *mut c_void;

        fn pearl_destroy_mining_graph_state(state: *mut c_void);

        fn pearl_update_mining_graph_state(
            state: *mut c_void,
            num_candidates: u32,
            num_combos: u32,
            tile_h: u32,
            tile_w: u32,
            m: u32,
            n: u32,
            k: u32,
            rank: u32,
        ) -> c_int;

        fn pearl_launch_mining_graph_state(
            state: *mut c_void,
            a_noised: *const i32,
            b_noised_t: *const i32,
            a_rows: *const i32,
            b_cols: *const i32,
            jobs: *const u8,
            jackpots: *mut u32,
            hashes: *mut u32,
            winner_flags: *mut u8,
        ) -> c_int;

        fn pearl_synchronize_mining_graph_state(state: *mut c_void) -> c_int;

        fn pearl_pipeline_get_jackpots(
            state: *mut c_void,
            jackpots: *mut u32,
            size: *mut usize,
        ) -> c_int;

        fn pearl_pipeline_get_hashes(
            state: *mut c_void,
            hashes: *mut u32,
            size: *mut usize,
        ) -> c_int;
    }

    impl MiningGraphState {
        #[allow(clippy::too_many_arguments)]
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
            unsafe {
                let state = pearl_create_mining_graph_state(
                    num_candidates as u32,
                    num_combos as u32,
                    tile_h as u32,
                    tile_w as u32,
                    m as u32,
                    n as u32,
                    k as u32,
                    rank as u32,
                );
                if state.is_null() {
                    anyhow::bail!("Failed to create mining graph state");
                }
                Ok(Self { state })
            }
        }

        #[allow(clippy::too_many_arguments)]
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
            unsafe {
                let err = pearl_update_mining_graph_state(
                    self.state,
                    num_candidates as u32,
                    num_combos as u32,
                    tile_h as u32,
                    tile_w as u32,
                    m as u32,
                    n as u32,
                    k as u32,
                    rank as u32,
                );
                if err != 0 {
                    anyhow::bail!("Failed to update mining graph state");
                }
            }
            Ok(())
        }

        pub fn launch(
            &self,
            a_noised: &pyo3::Py<pyo3::PyAny>,
            b_noised_t: &pyo3::Py<pyo3::PyAny>,
            a_rows: &pyo3::Py<pyo3::PyAny>,
            b_cols: &pyo3::Py<pyo3::PyAny>,
            jobs: &pyo3::Py<pyo3::PyAny>,
            jackpots: &pyo3::Py<pyo3::PyAny>,
            hashes: &pyo3::Py<pyo3::PyAny>,
            winner_flags: &pyo3::Py<pyo3::PyAny>,
        ) -> Result<()> {
            // FIX #2: extract all 8 data_ptr() values in a single GIL acquisition,
            // removing 7 redundant lock/unlock round-trips per kernel launch.
            let (a_ptr, b_ptr, ar_ptr, bc_ptr, j_ptr, jp_ptr, h_ptr, f_ptr) =
                Python::with_gil(|py| -> Result<(usize,usize,usize,usize,usize,usize,usize,usize)> {
                    let dp = "data_ptr";
                    let a   = a_noised.bind(py).call_method0(dp)?.extract::<usize>()?;
                    let b   = b_noised_t.bind(py).call_method0(dp)?.extract::<usize>()?;
                    let ar  = a_rows.bind(py).call_method0(dp)?.extract::<usize>()?;
                    let bc  = b_cols.bind(py).call_method0(dp)?.extract::<usize>()?;
                    let jb  = jobs.bind(py).call_method0(dp)?.extract::<usize>()?;
                    let jp  = jackpots.bind(py).call_method0(dp)?.extract::<usize>()?;
                    let h   = hashes.bind(py).call_method0(dp)?.extract::<usize>()?;
                    let f   = winner_flags.bind(py).call_method0(dp)?.extract::<usize>()?;
                    Ok((a, b, ar, bc, jb, jp, h, f))
                })?;

            unsafe {
                let err = pearl_launch_mining_graph_state(
                    self.state,
                    a_ptr  as *const i32,
                    b_ptr  as *const i32,
                    ar_ptr as *const i32,
                    bc_ptr as *const i32,
                    j_ptr  as *const u8,
                    jp_ptr as *mut u32,
                    h_ptr  as *mut u32,
                    f_ptr  as *mut u8,
                );
                if err != 0 {
                    anyhow::bail!("pearl_launch_mining_graph_state returned CUDA error {}", err);
                }
            }
            Ok(())
        }

        pub fn synchronize(&self) -> Result<()> {
            unsafe {
                let err = pearl_synchronize_mining_graph_state(self.state);
                if err != 0 {
                    anyhow::bail!("Failed to synchronize mining graph state");
                }
            }
            Ok(())
        }
    }

    impl Drop for MiningGraphState {
        fn drop(&mut self) {
            unsafe {
                pearl_destroy_mining_graph_state(self.state);
            }
        }
    }

    pub fn pipeline_get_jackpots(state: *mut c_void, jackpots: &mut [u32], size: &mut usize) -> Result<()> {
        unsafe {
            let err = pearl_pipeline_get_jackpots(state, jackpots.as_mut_ptr(), size as *mut usize);
            if err != 0 {
                anyhow::bail!("Failed to get pipeline jackpots");
            }
        }
        Ok(())
    }

    pub fn pipeline_get_hashes(state: *mut c_void, hashes: &mut [u32], size: &mut usize) -> Result<()> {
        unsafe {
            let err = pearl_pipeline_get_hashes(state, hashes.as_mut_ptr(), size as *mut usize);
            if err != 0 {
                anyhow::bail!("Failed to get pipeline hashes");
            }
        }
        Ok(())
    }
}

