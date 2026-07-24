//! Shared mining implementation for both Go FFI and Python bindings.

use anyhow::Result;
use primitive_types::U256;
use rand::Rng;

use crate::api::proof::{IncompleteBlockHeader, MiningConfiguration, PeriodicPattern};
use crate::api::proof_utils::{compute_hash_activations, compute_jackpot_hash};
use crate::api::sanity_checks::extract_difficulty_bound;
use crate::circuit::pearl_noise::compute_noise_for_indices;
use crate::circuit::pearl_program::{JACKPOT_SIZE, LROT_PER_TILE};
use crate::ffi::plain_proof::{MatrixMerkleProof, MoEProofParams, PlainProof};
use pearl_blake3::blake3_digest;

const SIGNAL_MIN: i8 = -64;
const SIGNAL_MAX: i8 = 64;

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
    let rank = config.rank as usize;
    let (signal_min, signal_max) = signal_range.unwrap_or((SIGNAL_MIN, SIGNAL_MAX));

    // Generate random matrices A (m×k) and B (k×n) as flat arrays
    let a_matrix: Vec<i8> = (0..m * k).map(|_| rng.random_range(signal_min..=signal_max)).collect();
    let b_matrix: Vec<i8> = (0..k * n).map(|_| rng.random_range(signal_min..=signal_max)).collect();

    // Transpose B for column-major format (n×k)
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

    // Compute noise using shared implementation from pearl_noise
    let a_all_rows: Vec<usize> = (0..m).collect();
    let b_all_cols: Vec<usize> = (0..n).collect();
    let noise = compute_noise_for_indices(k, rank, (b_noise_seed, a_noise_seed), &a_all_rows, &b_all_cols);

    // Add noise to matrices as flat arrays
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

    // Mine using pattern partitions
    for a_rows in threads_partition(&config.rows_pattern, m) {
        for b_cols in threads_partition(&config.cols_pattern, n) {
            let tile_h = a_rows.len();
            let tile_w = b_cols.len();
            let mut jackpot_tile: Vec<i32> = vec![0; tile_h * tile_w];
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
                let a_proof = build_matrix_proof(&a_matrix, m, k, &job_key, &a_rows);
                let b_proof = build_matrix_proof(&b_transposed, n, k, &job_key, &b_cols);

                return Ok(Some(PlainProof {
                    m,
                    n,
                    k,
                    noise_rank: rank,
                    a: a_proof,
                    bt: b_proof,
                    moe: None,
                }));
            }
        }
    }

    Ok(None)
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

    // Generate random matrices A (m × k) and B (k × (n_e × n)) as flat arrays
    let a_matrix: Vec<i8> = (0..m * k).map(|_| rng.random_range(signal_min..=signal_max)).collect();
    let b_matrix: Vec<i8> = (0..k * ne).map(|_| rng.random_range(signal_min..=signal_max)).collect();

    // Transpose B for column-major format (n_e × k)
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

    // Build a random routing where each token is assigned to `top_k` distinct experts.
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

    for expert_idx in 0..e {
        let row_period = config.rows_pattern.period() as usize;
        let expert_tokens = (routing[expert_idx].len() / row_period) * row_period;
        if expert_tokens == 0 {
            continue;
        }
        for a_rows in threads_partition(&config.rows_pattern, expert_tokens) {
            for b_cols in threads_partition(&config.cols_pattern, n) {
                let tile_h = a_rows.len();
                let tile_w = b_cols.len();
                let mut jackpot_tile: Vec<i32> = vec![0; tile_h * tile_w];
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
                        routing_end_offsets,
                        top_k,
                    };
                    return Ok(Some(PlainProof {
                        m,
                        k,
                        n,
                        noise_rank: rank,
                        a: a_proof,
                        bt: b_proof,
                        moe: Some(moe_params),
                    }));
                }
            }
        }
    }

    Ok(None)
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
    let hash_a = blake3_digest(a_row_major, Some(*job_key));
    let hash_b = blake3_digest(b_col_major, Some(*job_key));

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
