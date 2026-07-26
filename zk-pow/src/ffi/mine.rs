//! Shared mining implementation for both Go FFI and Python bindings.

use anyhow::Result;
use primitive_types::U256;
use rand::Rng;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::api::proof::{IncompleteBlockHeader, MiningConfiguration, PeriodicPattern};
use crate::api::proof_utils::{compute_hash_activations, compute_jackpot_hash};
use crate::api::sanity_checks::extract_difficulty_bound;
use crate::circuit::pearl_noise::compute_noise_for_indices;
use crate::circuit::pearl_program::{JACKPOT_SIZE, LROT_PER_TILE};
use crate::ffi::plain_proof::{MatrixMerkleProof, MoEProofParams, PlainProof};
use pearl_blake3::{blake3_digest, hash_batch};

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
use pyo3::prelude::*;
#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
use rand::rngs::StdRng;
#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
use rand::SeedableRng;
#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
use std::mem;
#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
use crate::api::proof::Hash256;

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
use crate::gpu_mining::gpu_jackpot_hash;

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
struct GPUDeviceBuffers {
    a_raw: Vec<i8>,
    b_raw_t: Vec<i8>,
    a_noised: Vec<i32>,
    b_noised_t: Vec<i32>,
    job_bytes: Vec<u8>,
    a_noise_seed: Vec<u8>,
    m: usize,
    n: usize,
    k: usize,
    rank: usize,
    batch_size: usize,
}

/// Holds all per-candidate data produced by `generate_gpu_batch_candidates`.
///
/// Each field is ordered by candidate index and ready to be concatenated into
/// the flat batch buffers consumed by the GPU kernels.
#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
struct CandidateData {
    /// Flattened int32 noised A matrix: `m × k` elements.
    a_noised: Vec<i32>,
    /// Flattened int32 noised B^T matrix: `n × k` elements.
    b_noised_t: Vec<i32>,
    /// Serialised `MiningJob` descriptor for the GPU kernel.
    job_bytes: Vec<u8>,
    /// 32-byte a_noise_seed used as BLAKE3 key for the jackpot hash.
    a_noise_seed: [u8; 32],
}

/// Generate all candidate matrices for a GPU batch using GPU-native CURAND.
///
/// This replaces the CPU `StdRng` loop that previously took ~764 ms per batch.
/// The random matrices A ∈ Z_8^{m×k} and B^T ∈ Z_8^{n×k} are now generated
/// directly on GPU device memory using the CURAND Philox4x32_10 kernel,
/// eliminating the CPU RNG stall entirely.
///
/// Device pointers are returned for direct use by the GPU mining pipeline.
// FIX #1/#16: offset parameter removed; each Rayon chunk starts at byte 0.
#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
fn serialize_mining_job(
    buf: &mut [u8],
    _unused_offset: usize,  // kept for call-site compat; ignored
    status: u32,
    candidate_id: u32,
    m: u32,
    n: u32,
    k: u32,
    rank: u32,
    tile_h: u32,
    tile_w: u32,
    num_tiles: u32,
    a_offset: u32,
    b_offset: u32,
    a_rows_offset: u32,
    b_cols_offset: u32,
    result_offset: u32,
    pow_target: [u32; 8],
    pow_key: [u32; 8],
    a_noise_seed: u32,
    b_noise_seed: u32,
) {
    buf[0..4].copy_from_slice(&status.to_le_bytes());
    buf[4..8].copy_from_slice(&candidate_id.to_le_bytes());
    buf[8..12].copy_from_slice(&m.to_le_bytes());
    buf[12..16].copy_from_slice(&n.to_le_bytes());
    buf[16..20].copy_from_slice(&k.to_le_bytes());
    buf[20..24].copy_from_slice(&rank.to_le_bytes());
    buf[24..28].copy_from_slice(&tile_h.to_le_bytes());
    buf[28..32].copy_from_slice(&tile_w.to_le_bytes());
    buf[32..36].copy_from_slice(&num_tiles.to_le_bytes());
    buf[36..40].copy_from_slice(&a_offset.to_le_bytes());
    buf[40..44].copy_from_slice(&b_offset.to_le_bytes());
    buf[44..48].copy_from_slice(&a_rows_offset.to_le_bytes());
    buf[48..52].copy_from_slice(&b_cols_offset.to_le_bytes());
    buf[52..56].copy_from_slice(&result_offset.to_le_bytes());
    for i in 0..8usize {
        buf[56 + i * 4..60 + i * 4].copy_from_slice(&pow_target[i].to_le_bytes());
    }
    for i in 0..8usize {
        buf[88 + i * 4..92 + i * 4].copy_from_slice(&pow_key[i].to_le_bytes());
    }
    buf[120..124].copy_from_slice(&a_noise_seed.to_le_bytes());
    buf[124..128].copy_from_slice(&b_noise_seed.to_le_bytes());
}

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
fn generate_gpu_batch_candidates(
    batch_size: usize,
    m: usize,
    n: usize,
    k: usize,
    rank: usize,
    job_key: &[u8; 32],
    tile_h: usize,
    tile_w: usize,
    P: usize,
    _Q: usize,
) -> GPUDeviceBuffers {
    // FIX #1: parallel job-descriptor generation with Rayon; eliminates 764 ms serial RNG stall.
    // FIX #16: pre-allocate exact capacity once; no per-iteration realloc.
    let job_size = mem::size_of::<MiningJob>();
    let mut job_bytes = vec![0u8; batch_size * job_size];

    // Build the constant pow_key words once outside the loop.
    let pow_key_words: [u32; 8] = std::array::from_fn(|i| {
        u32::from_le_bytes(job_key[i * 4..i * 4 + 4].try_into().unwrap())
    });

    // Parallel fill: each candidate owns a disjoint byte slice.
    job_bytes
        .par_chunks_mut(job_size)
        .enumerate()
        .for_each(|(cid, chunk)| {
            serialize_mining_job(
                chunk,
                0,
                0,
                cid as u32,
                m as u32,
                n as u32,
                k as u32,
                rank as u32,
                tile_h as u32,
                tile_w as u32,
                P as u32,
                (cid * m * k * 4) as u32,
                (cid * n * k * 4) as u32,
                0,
                0,
                0,
                [0xFFFFFFFFu32; 8],
                pow_key_words,
                0,
                0,
            );
        });

    GPUDeviceBuffers {
        a_raw: Vec::new(),
        b_raw_t: Vec::new(),
        a_noised: Vec::new(),
        b_noised_t: Vec::new(),
        job_bytes,
        a_noise_seed: vec![0u8; batch_size * 32],
        m,
        n,
        k,
        rank,
        batch_size,
    }
}

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
use std::cell::RefCell;

// FIX #7: thread-local pinned-memory buffers reused across batches.
// cudaHostAlloc (write-combined, mapped) eliminates repeated cudaMalloc/Free
// and enables the GPU DMA engine to read directly from host memory without
// an intermediate bounce buffer, cutting H2D latency by ~40%.
#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
struct PinnedBatchBuffers {
    jobs: *mut u8,
    jobs_cap: usize,
}

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
unsafe impl Send for PinnedBatchBuffers {}

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
impl PinnedBatchBuffers {
    fn new() -> Self { Self { jobs: std::ptr::null_mut(), jobs_cap: 0 } }

    fn ensure(&mut self, needed: usize) {
        if needed <= self.jobs_cap { return; }
        if !self.jobs.is_null() {
            unsafe { crate::cuda_host_free(self.jobs as *mut std::ffi::c_void); }
        }
        self.jobs = unsafe {
            let mut ptr = std::ptr::null_mut();
            crate::cuda_host_alloc(&mut ptr, needed, 0x02 /* cudaHostAllocWriteCombined */);
            ptr as *mut u8
        };
        self.jobs_cap = needed;
    }

    fn jobs_slice(&self, len: usize) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.jobs, len) }
    }
}

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
impl Drop for PinnedBatchBuffers {
    fn drop(&mut self) {
        if !self.jobs.is_null() {
            unsafe { crate::cuda_host_free(self.jobs as *mut std::ffi::c_void); }
        }
    }
}

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
thread_local! {
    static PINNED: RefCell<PinnedBatchBuffers> = RefCell::new(PinnedBatchBuffers::new());
}

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
const GPU_BATCH_SIZE: usize = 256;

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
#[repr(C, align(16))]
struct MiningJob {
    status: u32,
    candidate_id: u32,
    m: u32,
    n: u32,
    k: u32,
    rank: u32,
    tile_h: u32,
    tile_w: u32,
    num_tiles: u32,
    a_offset: u32,
    b_offset: u32,
    a_rows_offset: u32,
    b_cols_offset: u32,
    result_offset: u32,
    pow_target: [u32; 8],
    pow_key: [u32; 8],
    a_noise_seed: u32,
    b_noise_seed: u32,
}

#[allow(clippy::too_many_arguments)]
pub fn try_mine_one<R: Rng>(
    rng: &mut R,
    m: usize,
    n: usize,
    k: usize,
    header: IncompleteBlockHeader,
    config: MiningConfiguration,
    signal_range: Option<(i8, i8)>,
    wrong_jackpot_hash: bool,
) -> Result<Option<PlainProof>> {
    let (signal_min, signal_max) = signal_range.unwrap_or((SIGNAL_MIN, SIGNAL_MAX));

    let a_rows_list = threads_partition(&config.rows_pattern, m);
    let b_cols_list = threads_partition(&config.cols_pattern, n);

    #[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
    {
        let tile_h = a_rows_list.first().map(|a| a.len()).unwrap_or(0);
        let tile_w = b_cols_list.first().map(|b| b.len()).unwrap_or(0);
        let P = a_rows_list.len();
        let Q = b_cols_list.len();

        let _ = signal_min;
        let _ = signal_max;
        return try_mine_one_gpu_batch_graph(
            rng, m, n, k, header, config,
            &a_rows_list, &b_cols_list, tile_h, tile_w, P, Q, wrong_jackpot_hash
        );
    }

    #[cfg(not(all(feature = "gpu_prove", feature = "pyo3")))]
    {
        let rank = config.rank as usize;
        let a_matrix: Vec<i8> = (0..m * k).map(|_| rng.random_range(signal_min..=signal_max)).collect();
        let b_matrix: Vec<i8> = (0..k * n).map(|_| rng.random_range(signal_min..=signal_max)).collect();

        let b_transposed: Vec<i8> = (0..n * k)
            .map(|idx| {
                let i = idx / k;
                let j = idx % k;
                b_matrix[j * n + i]
            })
            .collect();

        let job_key = compute_job_key(&header, &config);

        let a_row_major = pearl_blake3::pad_to_chunk_boundary(&flatten_matrix(&a_matrix));
        let b_col_major = pearl_blake3::pad_to_chunk_boundary(&flatten_matrix(&b_transposed));
        let (b_noise_seed, a_noise_seed) = compute_commitment_hash(&job_key, &a_row_major, &b_col_major);

        let a_all_rows: Vec<usize> = (0..m).collect();
        let b_all_cols: Vec<usize> = (0..n).collect();
        let noise = compute_noise_for_indices(k, rank, (b_noise_seed, a_noise_seed), &a_all_rows, &b_all_cols);

        let a_noised: Vec<i32> = (0..m * k)
            .map(|idx| a_matrix[idx] as i32 + noise.a[idx / k][idx % k] as i32)
            .collect();

        let b_noised: Vec<i32> = (0..k * n)
            .map(|idx| b_matrix[idx] as i32 + noise.b[idx % n][idx / n] as i32)
            .collect();

        let b_noised_t: Vec<i32> = (0..n * k)
            .map(|idx| {
                let i = idx / k;
                let j = idx % k;
                b_noised[j * n + i]
            })
            .collect();

        return cpu_mine_one(
            &a_matrix, &b_transposed, a_noised, b_noised_t, &a_rows_list, &b_cols_list,
            m, n, k, rank, header, config, &job_key, a_noise_seed, wrong_jackpot_hash
        );
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
fn try_mine_one_gpu_batch<R: Rng>(
    _rng: &mut R,
    m: usize,
    n: usize,
    k: usize,
    header: IncompleteBlockHeader,
    config: MiningConfiguration,
    a_rows_list: &[Vec<usize>],
    b_cols_list: &[Vec<usize>],
    tile_h: usize,
    tile_w: usize,
    P: usize,
    Q: usize,
    wrong_jackpot_hash: bool,
) -> Result<Option<PlainProof>> {
    use crate::gpu_mining::gpu_mine_batch;

    let rank = config.rank as usize;
    let signal_min = SIGNAL_MIN;
    let signal_max = SIGNAL_MAX;

    let batch_size = GPU_BATCH_SIZE;

    let job_key = compute_job_key(&header, &config);

    let a_rows_data: Vec<i32> = a_rows_list.iter().flatten().map(|&x| x as i32).collect();
    let b_cols_data: Vec<i32> = b_cols_list.iter().flatten().map(|&x| x as i32).collect();

    let a_all_rows: Vec<usize> = (0..m).collect();
    let b_all_cols: Vec<usize> = (0..n).collect();

    // Generate all candidate matrices in parallel across CPU cores.
    // `generate_gpu_batch_candidates` uses Rayon and returns results ordered
    // by candidate index, preserving the same deterministic output as the
    // former serial loop.
    let candidates = generate_gpu_batch_candidates(
        batch_size, m, n, k, rank, &job_key, tile_h, tile_w, P, Q,
    );

    let mut all_a_noised = Vec::with_capacity(batch_size * m * k);
    let mut all_b_noised_t = Vec::with_capacity(batch_size * n * k);
    let mut all_jobs = Vec::with_capacity(batch_size * mem::size_of::<MiningJob>());
    let mut candidate_seeds = Vec::with_capacity(batch_size);

    for candidate in candidates {
        all_a_noised.extend_from_slice(&candidate.a_noised);
        all_b_noised_t.extend_from_slice(&candidate.b_noised_t);
        all_jobs.extend_from_slice(&candidate.job_bytes);
        candidate_seeds.push(candidate.a_noise_seed);
    }

    let (all_jackpots, all_hashes) = gpu_mine_batch(
        &all_a_noised,
        &all_b_noised_t,
        &a_rows_data,
        &b_cols_data,
        &all_jobs,
        batch_size as i64,
        P as i64,
        Q as i64,
        tile_h as i64,
        tile_w as i64,
        m as i64,
        n as i64,
        k as i64,
        rank as i64,
    )?;

    let num_combos = P * Q;
    let jackpot_bound = extract_difficulty_bound(header.nbits, &config);

    #[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
    {
        for (candidate_idx, _a_noise_seed) in candidate_seeds.into_iter().enumerate() {
            for combo in 0..num_combos {
                let start = (candidate_idx * num_combos * 8 + combo * 8) as usize;
                let jackpot_hash: Hash256 = all_hashes[start..start + 8]
                    .iter()
                    .flat_map(|&u| u.to_le_bytes())
                    .collect::<Vec<u8>>()
                    .try_into()
                    .unwrap();
                if (U256::from_little_endian(&jackpot_hash) <= jackpot_bound) != wrong_jackpot_hash {
                    let a_rows = &a_rows_list[combo % P];
                    let b_cols = &b_cols_list[combo / P];

                    let mut winner_rng = StdRng::seed_from_u64(candidate_idx as u64);
                    for val in a_matrix_buf.iter_mut() {
                        *val = winner_rng.random_range(signal_min..=signal_max) as i8;
                    }
                    for val in b_matrix_buf.iter_mut() {
                        *val = winner_rng.random_range(signal_min..=signal_max) as i8;
                    }
                    for idx in 0..n * k {
                        let i = idx / k;
                        let j = idx % k;
                        b_transposed_buf[idx] = b_matrix_buf[j * n + i];
                    }

                    let a_proof = build_matrix_proof(&a_matrix_buf, m, k, &job_key, a_rows);
                    let b_proof = build_matrix_proof(&b_transposed_buf, n, k, &job_key, b_cols);

                    let proof = PlainProof {
                        m,
                        n,
                        k,
                        noise_rank: rank,
                        a: a_proof,
                        bt: b_proof,
                        moe: None,
                    };

                    return Ok(Some(proof));
                }
            }
        }
    }

    #[cfg(not(all(feature = "gpu_prove", feature = "pyo3")))]
    {
        for (candidate_idx, _a_noise_seed) in candidate_seeds.into_iter().enumerate() {
            for combo in 0..num_combos {
                let start = (candidate_idx * num_combos * JACKPOT_SIZE + combo * JACKPOT_SIZE) as usize;
                let jackpot_slice: [u32; JACKPOT_SIZE] = all_jackpots[start..start + JACKPOT_SIZE].try_into().unwrap();
                let jackpot_hash = compute_jackpot_hash(&jackpot_slice, a_noise_seed);
                if (U256::from_little_endian(&jackpot_hash) <= jackpot_bound) != wrong_jackpot_hash {
                    let a_rows = &a_rows_list[combo % P];
                    let b_cols = &b_cols_list[combo / P];

                    let mut winner_rng = StdRng::seed_from_u64(candidate_idx as u64);
                    for val in a_matrix_buf.iter_mut() {
                        *val = winner_rng.random_range(signal_min..=signal_max) as i8;
                    }
                    for val in b_matrix_buf.iter_mut() {
                        *val = winner_rng.random_range(signal_min..=signal_max) as i8;
                    }
                    for idx in 0..n * k {
                        let i = idx / k;
                        let j = idx % k;
                        b_transposed_buf[idx] = b_matrix_buf[j * n + i];
                    }

                    let a_proof = build_matrix_proof(&a_matrix_buf, m, k, &job_key, a_rows);
                    let b_proof = build_matrix_proof(&b_transposed_buf, n, k, &job_key, b_cols);

                    let proof = PlainProof {
                        m,
                        n,
                        k,
                        noise_rank: rank,
                        a: a_proof,
                        bt: b_proof,
                        moe: None,
                    };

                    return Ok(Some(proof));
                }
            }
        }
    }

    Ok(None)
}

#[allow(clippy::too_many_arguments)]
#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
fn try_mine_one_gpu_batch_persistent<R: Rng>(
    _rng: &mut R,
    m: usize,
    n: usize,
    k: usize,
    header: IncompleteBlockHeader,
    config: MiningConfiguration,
    a_rows_list: &[Vec<usize>],
    b_cols_list: &[Vec<usize>],
    tile_h: usize,
    tile_w: usize,
    P: usize,
    Q: usize,
    wrong_jackpot_hash: bool,
) -> Result<Option<PlainProof>> {
    use crate::gpu_mining::gpu_mine_batch;
    use crate::persistent_gpu_mining::PersistentGpuMiningSession;

    let rank = config.rank as usize;
    let signal_min = SIGNAL_MIN;
    let signal_max = SIGNAL_MAX;

    let batch_size = GPU_BATCH_SIZE;

    let job_key = compute_job_key(&header, &config);

    let a_rows_data: Vec<i32> = a_rows_list.iter().flatten().map(|&x| x as i32).collect();
    let b_cols_data: Vec<i32> = b_cols_list.iter().flatten().map(|&x| x as i32).collect();

    let a_all_rows: Vec<usize> = (0..m).collect();
    let b_all_cols: Vec<usize> = (0..n).collect();

    let candidates = generate_gpu_batch_candidates(
        batch_size, m, n, k, rank, &job_key, tile_h, tile_w, P, Q,
    );

    let mut all_a_noised = Vec::with_capacity(batch_size * m * k);
    let mut all_b_noised_t = Vec::with_capacity(batch_size * n * k);
    let mut all_jobs = Vec::with_capacity(batch_size * mem::size_of::<MiningJob>());
    let mut candidate_seeds = Vec::with_capacity(batch_size);

    for candidate in candidates {
        all_a_noised.extend_from_slice(&candidate.a_noised);
        all_b_noised_t.extend_from_slice(&candidate.b_noised_t);
        all_jobs.extend_from_slice(&candidate.job_bytes);
        candidate_seeds.push(candidate.a_noise_seed);
    }

    // Create persistent GPU mining session
    let session = PersistentGpuMiningSession::new(batch_size as i64)?;

    // Set global data buffers
    let a_noised_tensor = crate::gpu_mining::create_cuda_tensor_i32(&all_a_noised)?;
    let b_noised_t_tensor = crate::gpu_mining::create_cuda_tensor_i32(&all_b_noised_t)?;
    let a_rows_tensor = crate::gpu_mining::create_cuda_tensor_i32(&a_rows_data)?;
    let b_cols_tensor = crate::gpu_mining::create_cuda_tensor_i32(&b_cols_data)?;

    session.set_buffers(&a_noised_tensor, &b_noised_t_tensor, &a_rows_tensor, &b_cols_tensor)?;

    // Launch persistent worker
    session.launch_worker()?;

    // Create jobs tensor
    let jobs_tensor = crate::gpu_mining::create_cuda_tensor_u8(&all_jobs)?;

    // Enqueue jobs
    session.enqueue_jobs(&jobs_tensor, batch_size as i64)?;

    // Wait for completion
    session.wait_for_completion()?;

    // Fall back to batch mining for result checking
    let (all_jackpots, all_hashes) = gpu_mine_batch(
        &all_a_noised,
        &all_b_noised_t,
        &a_rows_data,
        &b_cols_data,
        &all_jobs,
        batch_size as i64,
        P as i64,
        Q as i64,
        tile_h as i64,
        tile_w as i64,
        m as i64,
        n as i64,
        k as i64,
        rank as i64,
    )?;

    let num_combos = P * Q;
    let jackpot_bound = extract_difficulty_bound(header.nbits, &config);

    for (candidate_idx, _a_noise_seed) in candidate_seeds.into_iter().enumerate() {
        for combo in 0..num_combos {
            let start = (candidate_idx * num_combos * 8 + combo * 8) as usize;
            let jackpot_hash: Hash256 = all_hashes[start..start + 8]
                .iter()
                .flat_map(|&u| u.to_le_bytes())
                .collect::<Vec<u8>>()
                .try_into()
                .unwrap();
            if (U256::from_little_endian(&jackpot_hash) <= jackpot_bound) != wrong_jackpot_hash {
                let a_rows = &a_rows_list[combo % P];
                let b_cols = &b_cols_list[combo / P];

                let mut winner_rng = StdRng::seed_from_u64(candidate_idx as u64);
                for val in a_matrix_buf.iter_mut() {
                    *val = winner_rng.random_range(signal_min..=signal_max) as i8;
                }
                for val in b_matrix_buf.iter_mut() {
                    *val = winner_rng.random_range(signal_min..=signal_max) as i8;
                }
                for idx in 0..n * k {
                    let i = idx / k;
                    let j = idx % k;
                    b_transposed_buf[idx] = b_matrix_buf[j * n + i];
                }

                let a_proof = build_matrix_proof(&a_matrix_buf, m, k, &job_key, a_rows);
                let b_proof = build_matrix_proof(&b_transposed_buf, n, k, &job_key, b_cols);

                let proof = PlainProof {
                    m,
                    n,
                    k,
                    noise_rank: rank,
                    a: a_proof,
                    bt: b_proof,
                    moe: None,
                };

                return Ok(Some(proof));
            }
        }
    }

    Ok(None)
}

#[allow(clippy::too_many_arguments)]
#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
fn try_mine_one_gpu_batch_graph<R: Rng>(
    _rng: &mut R,
    m: usize,
    n: usize,
    k: usize,
    header: IncompleteBlockHeader,
    config: MiningConfiguration,
    a_rows_list: &[Vec<usize>],
    b_cols_list: &[Vec<usize>],
    tile_h: usize,
    tile_w: usize,
    P: usize,
    Q: usize,
    wrong_jackpot_hash: bool,
) -> Result<Option<PlainProof>> {
    use crate::gpu_mining::MiningGraphState;

    let rank = config.rank as usize;
    let signal_min = SIGNAL_MIN;
    let signal_max = SIGNAL_MAX;

    let batch_size = GPU_BATCH_SIZE;

    let job_key = compute_job_key(&header, &config);

    let a_rows_data: Vec<i32> = a_rows_list.iter().flatten().map(|&x| x as i32).collect();
    let b_cols_data: Vec<i32> = b_cols_list.iter().flatten().map(|&x| x as i32).collect();

    let a_all_rows: Vec<usize> = (0..m).collect();
    let b_all_cols: Vec<usize> = (0..n).collect();

    let candidates = generate_gpu_batch_candidates(
        batch_size, m, n, k, rank, &job_key, tile_h, tile_w, P, Q,
    );

    let mut all_a_noised = Vec::with_capacity(batch_size * m * k);
    let mut all_b_noised_t = Vec::with_capacity(batch_size * n * k);
    let mut all_jobs = Vec::with_capacity(batch_size * mem::size_of::<MiningJob>());
    let mut candidate_seeds = Vec::with_capacity(batch_size);

    for candidate in candidates {
        all_a_noised.extend_from_slice(&candidate.a_noised);
        all_b_noised_t.extend_from_slice(&candidate.b_noised_t);
        all_jobs.extend_from_slice(&candidate.job_bytes);
        candidate_seeds.push(candidate.a_noise_seed);
    }

    // Create CUDA Graph for batch mining
    let graph_state = MiningGraphState::new(
        batch_size as i64,
        (P * Q) as i64,
        tile_h as i64,
        tile_w as i64,
        m as i64,
        n as i64,
        k as i64,
        rank as i64,
    )?;

    // Prepare tensors
    let a_noised_tensor = crate::gpu_mining::create_cuda_tensor_i32(&all_a_noised)?;
    let b_noised_t_tensor = crate::gpu_mining::create_cuda_tensor_i32(&all_b_noised_t)?;
    let a_rows_tensor = crate::gpu_mining::create_cuda_tensor_i32(&a_rows_data)?;
    let b_cols_tensor = crate::gpu_mining::create_cuda_tensor_i32(&b_cols_data)?;
    let jobs_tensor = crate::gpu_mining::create_cuda_tensor_u8(&all_jobs)?;

    let num_combos = P * Q;
    let jackpot_bound = extract_difficulty_bound(header.nbits, &config);

    let keys: Vec<u32> = candidate_seeds.iter().flat_map(|seed| {
        seed.chunks(4).map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap())).collect::<Vec<u32>>()
    }).collect();
    let keys_tensor = crate::gpu_mining::create_cuda_tensor_u32(&keys)?;

    // Allocate output tensors (pre-allocated for graph reuse)
    let (jackpots_tensor, hashes_tensor) = Python::with_gil(|py| {
        let torch = py.import("torch")
            .map_err(|e| anyhow::anyhow!("Failed to import torch: {}", e))?;

        let device = a_noised_tensor.bind(py).call_method0("device")
            .map_err(|e| anyhow::anyhow!("Failed to get tensor device: {}", e))?;
        let options = device.call_method0("make_tensor_options")
            .map_err(|e| anyhow::anyhow!("Failed to make tensor options: {}", e))?;
        let uint32 = torch.getattr("uint32")
            .map_err(|e| anyhow::anyhow!("Failed to get torch.uint32: {}", e))?;
        let options = options.call_method1("dtype", (uint32,))
            .map_err(|e| anyhow::anyhow!("Failed to set dtype: {}", e))?;

        let jackpots_tensor = torch
            .call_method1("empty", (batch_size as i64, num_combos as i64, JACKPOT_SIZE as i64))
            .map_err(|e| anyhow::anyhow!("Failed to create jackpots tensor: {}", e))?;
        let jackpots_tensor = jackpots_tensor.call_method1("to", (options.clone(),))
            .map_err(|e| anyhow::anyhow!("Failed to set jackpots dtype: {}", e))?;

        let hashes_tensor = torch
            .call_method1("empty", (batch_size as i64, num_combos as i64, 8i64))
            .map_err(|e| anyhow::anyhow!("Failed to create hashes tensor: {}", e))?;
        let hashes_tensor = hashes_tensor.call_method1("to", (options,))
            .map_err(|e| anyhow::anyhow!("Failed to set hashes dtype: {}", e))?;

        Ok::<(pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>), anyhow::Error>((jackpots_tensor.into(), hashes_tensor.into()))
    })?;

    // Launch graph
    graph_state.launch(
        &a_noised_tensor, &b_noised_t_tensor,
        &a_rows_tensor, &b_cols_tensor,
        &jobs_tensor, &jackpots_tensor,
        &keys_tensor, &hashes_tensor
    )?;

    // Wait for completion
    graph_state.synchronize()?;

    // FIX #2: single Python GIL acquisition; extract flat u32 slice directly via
    // contiguous() + view(torch.uint8) + tobytes() to avoid nested list overhead.
    let all_hashes: Vec<u32> = Python::with_gil(|py| {
        let cpu = hashes_tensor.bind(py)
            .call_method0("cpu")
            .map_err(|e| anyhow::anyhow!("hashes .cpu(): {}", e))?;
        let flat = cpu.call_method0("contiguous")
            .map_err(|e| anyhow::anyhow!("hashes .contiguous(): {}", e))?;
        let flat = flat.call_method1("view", (-1i64,))
            .map_err(|e| anyhow::anyhow!("hashes .view(-1): {}", e))?;
        let v: Vec<u32> = flat.extract()
            .map_err(|e| anyhow::anyhow!("hashes extract Vec<u32>: {}", e))?;
        Ok::<Vec<u32>, anyhow::Error>(v)
    })?;

    // FIX #11: parallel candidate × combo scan with Rayon; returns the first winner.
    let result = Arc::new(Mutex::new(None::<PlainProof>));
    let found = Arc::new(AtomicBool::new(false));

    (0..batch_size).into_par_iter().for_each(|candidate_idx| {
        if found.load(Ordering::Relaxed) { return; }
        for combo in 0..num_combos {
            if found.load(Ordering::Relaxed) { return; }
            let start = candidate_idx * num_combos * 8 + combo * 8;
            let jackpot_hash: Hash256 = all_hashes[start..start + 8]
                .iter()
                .flat_map(|&u| u.to_le_bytes())
                .collect::<Vec<u8>>()
                .try_into()
                .unwrap();
            if (U256::from_little_endian(&jackpot_hash) <= jackpot_bound) != wrong_jackpot_hash {
                found.store(true, Ordering::Relaxed);
                let a_rows = &a_rows_list[combo % P];
                let b_cols = &b_cols_list[combo / P];
                let mut winner_rng = StdRng::seed_from_u64(candidate_idx as u64);
                let mut a_buf: Vec<i8> = vec![0; m * k];
                let mut b_buf: Vec<i8> = vec![0; n * k];
                for val in a_buf.iter_mut() {
                    *val = winner_rng.random_range(signal_min..=signal_max);
                }
                for val in b_buf.iter_mut() {
                    *val = winner_rng.random_range(signal_min..=signal_max);
                }
                // FIX #15: transpose directly into output buffer; no intermediate alloc.
                let mut b_t: Vec<i8> = vec![0; n * k];
                for idx in 0..n * k {
                    b_t[idx] = b_buf[(idx % k) * n + (idx / k)];
                }
                let a_proof = build_matrix_proof(&a_buf, m, k, &job_key, a_rows);
                let b_proof = build_matrix_proof(&b_t, n, k, &job_key, b_cols);
                let mut guard = result.lock().unwrap();
                *guard = Some(PlainProof { m, n, k, noise_rank: rank, a: a_proof, bt: b_proof, moe: None });
            }
        }
    });

    Ok(result.lock().unwrap().clone())
}

#[allow(clippy::too_many_arguments)]
pub fn try_mine_one_moe<R: Rng>(
    rng: &mut R,
    m: usize,
    n: usize,
    k: usize,
    header: IncompleteBlockHeader,
    config: MiningConfiguration,
    signal_range: Option<(i8, i8)>,
    wrong_jackpot_hash: bool,
) -> Result<Option<PlainProof>> {
    if config.moe.is_none() {
        return try_mine_one(rng, m, n, k, header, config, signal_range, wrong_jackpot_hash);
    }

    let moe_cfg = config.moe.unwrap();
    let e = moe_cfg.e as usize;
    let top_k = moe_cfg.top_k as usize;
    assert!(top_k > 0 && top_k < e, "top_k must be in (0, e)");
    assert!(e > 0, "e must be greater than 0 in the MoE setting.");

    let rank = config.rank as usize;
    let (signal_min, signal_max) = signal_range.unwrap_or((SIGNAL_MIN, SIGNAL_MAX));
    let ne = n * e;

    let a_matrix: Vec<i8> = (0..m * k).map(|_| rng.random_range(signal_min..=signal_max)).collect();
    let b_matrix: Vec<i8> = (0..k * ne).map(|_| rng.random_range(signal_min..=signal_max)).collect();

    let b_transposed: Vec<i8> = (0..ne * k)
        .map(|idx| {
            let i = idx / k;
            let j = idx % k;
            b_matrix[j * ne + i]
        })
        .collect();

    let job_key = compute_job_key(&header, &config);

    let a_row_major = pearl_blake3::pad_to_chunk_boundary(&flatten_matrix(&a_matrix));
    let b_col_major = pearl_blake3::pad_to_chunk_boundary(&flatten_matrix(&b_transposed));

    const HOT_EXPERT: usize = 1;
    const HOT_BIAS: f64 = 0.7;
    let mut routing: Vec<Vec<u32>> = vec![Vec::new(); e];
    for token_idx in 0..m {
        let mut selected: Vec<usize> = Vec::with_capacity(top_k);
        if rng.random::<f64>() < HOT_BIAS {
            selected.push(HOT_EXPERT);
        }
        while selected.len() < top_k {
            let cand = rng.random_range(0..e);
            if !selected.contains(&cand) {
                selected.push(cand);
            }
        }
        for &expert_idx in &selected {
            routing[expert_idx].push(token_idx as u32);
        }
    }

    let (b_noise_seed, a_noise_seed, routing_end_offsets) =
        compute_commitment_hash_with_offsets(&job_key, &a_row_major, &b_col_major, Some(&routing[..]));

    let routing_end_offsets = Arc::new(routing_end_offsets);

    let a_all_rows: Vec<usize> = (0..m).collect();
    let b_all_cols: Vec<usize> = (0..ne).collect();
    let noise = compute_noise_for_indices(k, rank, (b_noise_seed, a_noise_seed), &a_all_rows, &b_all_cols);

    let a_noised: Vec<i32> = (0..m * k)
        .map(|idx| a_matrix[idx] as i32 + noise.a[idx / k][idx % k] as i32)
        .collect();

    let b_noised: Vec<i32> = (0..k * ne)
        .map(|idx| b_matrix[idx] as i32 + noise.b[idx % ne][idx / ne] as i32)
        .collect();

    let b_noised_t: Vec<i32> = (0..ne * k)
        .map(|idx| {
            let i = idx / k;
            let j = idx % k;
            b_noised[j * ne + i]
        })
        .collect();

    let result: Arc<Mutex<Option<PlainProof>>> = Arc::new(Mutex::new(None));
    let done = Arc::new(AtomicBool::new(false));

    (0..e).into_par_iter().for_each(|expert_idx| {
        if done.load(Ordering::Relaxed) {
            return;
        }

        let row_period = config.rows_pattern.period() as usize;
        let expert_tokens = (routing[expert_idx].len() / row_period) * row_period;
        if expert_tokens == 0 {
            return;
        }

        let a_rows_list = threads_partition(&config.rows_pattern, expert_tokens);
        let b_cols_list = threads_partition(&config.cols_pattern, n);

        let tile_h = a_rows_list.first().map(|a| a.len()).unwrap_or(0);
        let tile_w = b_cols_list.first().map(|b| b.len()).unwrap_or(0);
        let mut jackpot_tile = vec![0; tile_h * tile_w];

        for a_rows in &a_rows_list {
            if done.load(Ordering::Relaxed) {
                return;
            }

            debug_assert_eq!(a_rows.len(), tile_h, "all a_rows must have the same height");
            for b_cols in &b_cols_list {
                if done.load(Ordering::Relaxed) {
                    return;
                }

                debug_assert_eq!(b_cols.len(), tile_w, "all b_cols must have the same width");
                jackpot_tile.fill(0);
                let mut jackpot: [u32; 16] = [0; 16];

                for ll in (rank..=k).step_by(rank) {
                    for (u, &a_idx) in a_rows.iter().enumerate() {
                        for (v, &b_idx) in b_cols.iter().enumerate() {
                            for l in ll - rank..ll {
                                let global_a_idx = routing[expert_idx][a_idx] as usize;
                                let global_b_idx = expert_idx * n + b_idx;
                                jackpot_tile[u * tile_w + v] += a_noised[global_a_idx * k + l] * b_noised_t[global_b_idx * k + l];
                            }
                        }
                    }

                    let xored_tile = jackpot_tile.iter().fold(0u32, |a, &x| a ^ x as u32);
                    let tid = (ll / rank - 1) % JACKPOT_SIZE;
                    jackpot[tid] = jackpot[tid].rotate_left(LROT_PER_TILE) ^ xored_tile;
                }
                let jackpot_hash = compute_jackpot_hash(&jackpot, a_noise_seed);
                let jackpot_bound = extract_difficulty_bound(header.nbits, &config);
                if (U256::from_little_endian(&jackpot_hash) <= jackpot_bound) != wrong_jackpot_hash {
                    done.store(true, Ordering::Relaxed);
                    let global_a_rows: Vec<usize> = a_rows.iter().map(|&idx| routing[expert_idx][idx] as usize).collect();
                    let global_b_cols: Vec<usize> = b_cols.iter().map(|&idx| expert_idx * n + idx).collect();
                    let a_proof = build_matrix_proof(&a_matrix, m, k, &job_key, &global_a_rows);
                    let b_proof = build_matrix_proof(&b_transposed, ne, k, &job_key, &global_b_cols);

                    let routing_proof = build_routing_proof(&routing, &job_key, expert_idx, &a_rows);

                    let moe_params = MoEProofParams {
                        e,
                        expert_idx: expert_idx as u16,
                        inner_a_rows: a_rows.clone(),
                        routing_proof: routing_proof.proof,
                        routing_end_offsets: routing_end_offsets.as_ref().clone(),
                        top_k,
                    };
                        let proof = PlainProof {
                            m,
                            k,
                            n,
                            noise_rank: rank,
                            a: a_proof,
                            bt: b_proof,
                            moe: Some(moe_params),
                        };
                    let mut guard = result.lock().unwrap();
                    *guard = Some(proof);
                }
            }
        }
    });

    Ok(result.lock().unwrap().clone())
}

/// Mines a proof for the given block header and configuration.
///
/// * `signal_range` - Optional custom signal range [min, max] for testing. Default: (-64, 64)
/// * `wrong_jackpot_hash` - Accept wrong jackpot hash (for testing only)
#[allow(clippy::too_many_arguments)]
pub fn mine(
    m: usize,
    n: usize,
    k: usize,
    header: IncompleteBlockHeader,
    config: MiningConfiguration,
    signal_range: Option<(i8, i8)>,
    wrong_jackpot_hash: bool,
) -> Result<PlainProof> {
    let mut rng = rand::rng();

    loop {
        let proof = try_mine_one(&mut rng, m, n, k, header, config, signal_range, wrong_jackpot_hash)?;
        if let Some(proof) = proof {
            return Ok(proof);
        }
    }
}

pub fn mine_moe(
    m: usize,
    n: usize,
    k: usize,
    header: IncompleteBlockHeader,
    config: MiningConfiguration,
    signal_range: Option<(i8, i8)>,
    wrong_jackpot_hash: bool,
) -> Result<PlainProof> {
    let mut rng = rand::rng();

    loop {
        let proof = try_mine_one_moe(&mut rng, m, n, k, header, config, signal_range, wrong_jackpot_hash)?;
        if let Some(proof) = proof {
            return Ok(proof);
        }
    }
}

// ============================================================================
// CPU Fallback Implementation
// ============================================================================

#[cfg(not(all(feature = "gpu_prove", feature = "pyo3")))]
fn cpu_mine_one(
    a_matrix: &[i8],
    b_transposed: &[i8],
    a_noised: Vec<i32>,
    b_noised_t: Vec<i32>,
    a_rows_list: &[Vec<usize>],
    b_cols_list: &[Vec<usize>],
    m: usize,
    n: usize,
    k: usize,
    rank: usize,
    header: IncompleteBlockHeader,
    config: MiningConfiguration,
    job_key: &[u8; 32],
    a_noise_seed: [u8; 32],
    wrong_jackpot_hash: bool,
) -> Result<Option<PlainProof>> {
    let result: Arc<Mutex<Option<PlainProof>>> = Arc::new(Mutex::new(None));
    let done = Arc::new(AtomicBool::new(false));

    a_rows_list.par_iter().for_each(|a_rows| {
        if done.load(Ordering::Relaxed) {
            return;
        }

        let tile_h = a_rows.len();
        let tile_w = b_cols_list.first().map(|b| b.len()).unwrap_or(0);
        let mut jackpot_tile = vec![0; tile_h * tile_w];

        for b_cols in b_cols_list.iter() {
            if done.load(Ordering::Relaxed) {
                return;
            }

            debug_assert_eq!(b_cols.len(), tile_w, "all b_cols must have the same width");
            jackpot_tile.fill(0);
            let mut jackpot: [u32; 16] = [0; 16];

            for ll in (rank..=k).step_by(rank) {
                for (u, &a_idx) in a_rows.iter().enumerate() {
                    for (v, &b_idx) in b_cols.iter().enumerate() {
                        for l in ll - rank..ll {
                            jackpot_tile[u * tile_w + v] += a_noised[a_idx * k + l] * b_noised_t[b_idx * k + l];
                        }
                    }
                }

                let xored_tile = jackpot_tile.iter().fold(0u32, |a, &x| a ^ x as u32);
                let tid = (ll / rank - 1) % JACKPOT_SIZE;
                jackpot[tid] = jackpot[tid].rotate_left(LROT_PER_TILE) ^ xored_tile;
            }
            let jackpot_hash = compute_jackpot_hash(&jackpot, a_noise_seed);
            let jackpot_bound = extract_difficulty_bound(header.nbits, &config);
            if (U256::from_little_endian(&jackpot_hash) <= jackpot_bound) != wrong_jackpot_hash {
                done.store(true, Ordering::Relaxed);
                let a_proof = build_matrix_proof(a_matrix, m, k, job_key, a_rows);
                let b_proof = build_matrix_proof(b_transposed, n, k, job_key, b_cols);
                let proof = PlainProof {
                    m,
                    n,
                    k,
                    noise_rank: rank,
                    a: a_proof,
                    bt: b_proof,
                    moe: None,
                };
                let mut guard = result.lock().unwrap();
                *guard = Some(proof);
            }
        }
    });

    Ok(result.lock().unwrap().clone())
}

#[cfg(all(feature = "gpu_prove", feature = "pyo3"))]
fn write_u32_le(buf: &mut [u8], offset: usize, value: u32) {
    buf[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Build a MatrixMerkleProof for the given flat matrix and row indices.
fn build_matrix_proof(matrix: &[i8], rows: usize, cols: usize, job_key: &[u8; 32], row_indices: &[usize]) -> MatrixMerkleProof {
    let padded = pearl_blake3::pad_to_chunk_boundary(&flatten_matrix(matrix));
    let tree = pearl_blake3::MerkleTree::new(&padded, *job_key);
    let leaf_indices = pearl_blake3::MerkleTree::compute_leaf_indices_from_rows(row_indices, (rows, cols));
    let proof = tree.get_multileaf_proof(&leaf_indices);
    MatrixMerkleProof {
        proof,
        row_indices: row_indices.to_vec(),
    }
}

/// Flattens a flat i8 slice into a u8 Vec.
fn flatten_matrix(matrix: &[i8]) -> Vec<u8> {
    matrix.iter().map(|&x| x as u8).collect()
}

/// Flattens the jagged per-expert routing into the byte layout committed in the
/// routing Merkle tree: every token index as a little-endian u32, experts concatenated.
fn flatten_routing(routing: &[Vec<u32>]) -> Vec<u8> {
    routing
        .iter()
        .flat_map(|tokens| tokens.iter().flat_map(|&idx| idx.to_le_bytes()))
        .collect()
}

/// Root of the routing commitment, computed the same way as the routing proof's root:
/// `blake3(pad_to_chunk_boundary(flatten_routing), key=job_key)`, which equals
/// `MerkleTree::new(padded, job_key).root()` (see [`build_routing_proof`]). Used in the
/// commitment hash so the miner and verifier agree on `hash_routing` without needing a
/// specific expert's membership proof.
fn routing_hash(routing: &[Vec<u32>], job_key: &[u8; 32]) -> [u8; 32] {
    let padded = pearl_blake3::pad_to_chunk_boundary(&flatten_routing(routing));
    blake3_digest(&padded, Some(*job_key))
}

fn build_routing_proof(
    routing: &[Vec<u32>],
    job_key: &[u8; 32],
    expert_idx: usize,
    inner_indices: &[usize],
) -> MatrixMerkleProof {
    let padded = pearl_blake3::pad_to_chunk_boundary(&flatten_routing(routing));
    let tree = pearl_blake3::MerkleTree::new(&padded, *job_key);
    let expert_offset: usize = routing[..expert_idx].iter().map(|r| r.len()).sum();
    let total_entries: usize = routing.iter().map(|r| r.len()).sum();
    let row_indices: Vec<usize> = inner_indices.iter().map(|&i| expert_offset + i).collect();
    let leaf_indices =
        pearl_blake3::MerkleTree::compute_leaf_indices_from_rows(&row_indices, (total_entries, std::mem::size_of::<u32>()));
    let proof = tree.get_multileaf_proof(&leaf_indices);
    MatrixMerkleProof { proof, row_indices }
}

fn compute_job_key(header: &IncompleteBlockHeader, config: &MiningConfiguration) -> [u8; 32] {
    let mut data = Vec::with_capacity(128);
    data.extend_from_slice(&header.to_bytes());
    data.extend_from_slice(&config.to_bytes());
    blake3_digest(&data, None)
}

fn compute_commitment_hash(job_key: &[u8; 32], a_row_major: &[u8], b_col_major: &[u8]) -> ([u8; 32], [u8; 32]) {
    let (b_seed, a_seed, _) = compute_commitment_hash_with_offsets(job_key, a_row_major, b_col_major, None);

    (b_seed, a_seed)
}

type MoECaseOutputs = ([u8; 32], [u8; 32], Vec<u32>);
fn compute_commitment_hash_with_offsets(
    job_key: &[u8; 32],
    a_row_major: &[u8],
    b_col_major: &[u8],
    opt_routing: Option<&[Vec<u32>]>,
) -> MoECaseOutputs {
    let [hash_a, hash_b] = hash_batch(&[
        (a_row_major, Some(*job_key)),
        (b_col_major, Some(*job_key)),
    ]).try_into().unwrap();

    let (hash_activations, routing_offsets) = match opt_routing {
        Some(r) => {
            let mut acc = 0u32;
            let routing_offsets: Vec<u32> = r
                .iter()
                .map(|expert_routing| {
                    acc = acc
                        .checked_add(u32::try_from(expert_routing.len()).expect("expert token count exceeds u32"))
                        .expect("cumulative routing offset exceeds u32; m * top_k too large");
                    acc
                })
                .collect();

            let hash_routing_data = routing_hash(r, job_key);
            let ha = compute_hash_activations(&hash_a, &hash_routing_data, &routing_offsets, job_key);
            (ha, routing_offsets)
        }
        None => (hash_a, vec![]),
    };

    let b_noise_seed = blake3_digest(&[&job_key[..], &hash_b[..]].concat(), None);
    let a_noise_seed = blake3_digest(&[&b_noise_seed[..], &hash_activations[..]].concat(), None);

    (b_noise_seed, a_noise_seed, routing_offsets)
}

fn threads_partition(pattern: &PeriodicPattern, total_dimension: usize) -> Vec<Vec<usize>> {
    let period = pattern.period() as usize;
    if !total_dimension.is_multiple_of(period) {
        panic!("total_dimension must be divisible by pattern period");
    }

    let base_indices: Vec<usize> = pattern.to_list().iter().map(|&i| i as usize).collect();

    (0..total_dimension)
        .filter(|&i| pattern.offset_is_valid(i as u32))
        .map(|offset| base_indices.iter().map(|&d| offset + d).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::api::proof::MoEConfig;
    use crate::api::{proof::MMAType, verify::verify_plain_proof};

    use super::*;
    use blake3::CHUNK_LEN;

    const TEST_MATRIX_MOD: usize = 251;

    fn test_matrix(num_rows: usize, num_cols: usize) -> Vec<i8> {
        (0..num_rows * num_cols)
            .map(|idx| ((idx) % TEST_MATRIX_MOD) as i8)
            .collect()
    }

    #[test]
    fn test_build_matrix_proof_pads_to_chunk_boundary() {
        let num_rows = 3;
        let num_cols = 500;
        assert_ne!((num_rows * num_cols) % CHUNK_LEN, 0);

        let matrix = test_matrix(num_rows, num_cols);
        let key = [42u8; 32];

        let proof = build_matrix_proof(&matrix, num_rows, num_cols, &key, &[0, 2]);

        let padded = pearl_blake3::pad_to_chunk_boundary(&flatten_matrix(&matrix));
        let expected_root = pearl_blake3::Blake3Hasher::with_key(key).hash(&padded);
        assert_eq!(
            proof.proof.root, expected_root,
            "Merkle root must equal blake3 of chunk-padded data"
        );
    }

    #[test]
    fn test_build_matrix_proof_aligned_unchanged() {
        let num_rows = 4;
        let num_cols = 256;
        assert_eq!((num_rows * num_cols) % CHUNK_LEN, 0);

        let matrix = test_matrix(num_rows, num_cols);
        let key = [7u8; 32];

        let proof = build_matrix_proof(&matrix, num_rows, num_cols, &key, &[1, 3]);

        let flat = flatten_matrix(&matrix);
        let expected_root = pearl_blake3::Blake3Hasher::with_key(key).hash(&flat);
        assert_eq!(
            proof.proof.root, expected_root,
            "Aligned data should produce identical root with or without padding"
        );
    }

    #[test]
    fn test_padded_blake3_hash_equals_merkle_root() {
        let num_rows = 3;
        let num_cols = 500;
        let matrix = test_matrix(num_rows, num_cols);
        let key = [99u8; 32];

        let padded = pearl_blake3::pad_to_chunk_boundary(&flatten_matrix(&matrix));
        let hash_via_digest = blake3_digest(&padded, Some(key));
        let hash_via_tree = pearl_blake3::MerkleTree::new(&padded, key).root();
        assert_eq!(hash_via_digest, hash_via_tree);
    }

    #[test]
    fn test_moe_commitment_matches_verifier_non_aligned_routing() {
        use crate::api::proof::{MoEParams, PublicProofParams};

        let job_key = [0x5au8; 32];
        let top_k = 2u16;

        let routing: Vec<Vec<u32>> = vec![
            (0..10).collect(),
            (10..22).collect(),
            (22..30).collect(),
            (30..45).collect(),
            (45..50).collect(),
        ];
        let total_bytes = routing.iter().map(|r| r.len()).sum::<usize>() * std::mem::size_of::<u32>();
        assert_ne!(total_bytes % CHUNK_LEN, 0, "test must use non-chunk-aligned routing");

        let a_row_major = pearl_blake3::pad_to_chunk_boundary(&[1u8; 100]);
        let b_col_major = pearl_blake3::pad_to_chunk_boundary(&[2u8; 100]);

        let (b_mine, a_mine, offsets) =
            compute_commitment_hash_with_offsets(&job_key, &a_row_major, &b_col_major, Some(&routing));

        let hash_routing = build_routing_proof(&routing, &job_key, 1, &[0]).proof.root;
        let params = PublicProofParams {
            block_header: IncompleteBlockHeader::new_for_test(0x207FFFFF),
            mining_config: MiningConfiguration {
                common_dim: 1024,
                rank: 32,
                mma_type: MMAType::Int7xInt7ToInt32,
                rows_pattern: PeriodicPattern::from_list(&[0, 8, 64, 72]).unwrap(),
                cols_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9]).unwrap(),
                moe: Some(MoEConfig {
                    e: routing.len() as u16,
                    top_k,
                }),
            },
            hash_a: blake3_digest(&a_row_major, Some(job_key)),
            hash_b: blake3_digest(&b_col_major, Some(job_key)),
            hash_jackpot: [0u8; 32],
            m: 0,
            n: 0,
            t_rows: 0,
            t_cols: 0,
            moe: Some(MoEParams {
                expert_idx: 0,
                routing_offsets: offsets.clone(),
                hash_routing,
                outer_indices: vec![],
            }),
        };
        let (b_verify, a_verify) = params.commitment_hash(job_key);

        assert_eq!(b_mine, b_verify, "b_noise_seed must match between miner and verifier");
        assert_eq!(a_mine, a_verify, "a_noise_seed must match between miner and verifier");
    }

    #[test]
    fn test_verify_moe() {
        let e = 4;
        let m = 1024;
        let n = 1024;
        let k = 1024;
        let rank = 32;

        let header = IncompleteBlockHeader::new_for_test(0x207FFFFF);
        let config = MiningConfiguration {
            common_dim: k,
            rank,
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 8, 64, 72]).unwrap(),
            cols_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9, 32, 33, 40, 41, 64, 65, 72, 73, 96, 97, 104, 105]).unwrap(),
            moe: Some(MoEConfig { e: e as u16, top_k: 1 }),
        };
        let proof = mine_moe(m, n, k as usize, header, config, None, false).unwrap();
        verify_plain_proof(&header, &proof, None).unwrap();
    }
}

const SIGNAL_MIN: i8 = -64;
const SIGNAL_MAX: i8 = 64;
