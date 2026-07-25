//! Benchmarks for GPU-accelerated STARK proving operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use zk_pow::gpu_stark_prover::GpuStarkProver;

fn bench_gpu_lde_eval(c: &mut Criterion) {
    let prover = GpuStarkProver::new(true);

    // Benchmark with typical STARK trace size
    let coeffs: Vec<u32> = (0..1024).map(|i| (i % 256) as u32).collect();
    let eval_points: Vec<u32> = (0..2048).map(|i| (i % 256) as u32).collect();
    let poly_degree = coeffs.len() - 1;

    c.bench_function("gpu_lde_eval_1024coeffs_2048points", |b| {
        b.iter(|| {
            prover.prove_gpu(
                black_box(&vec![coeffs.clone()]),
                black_box(&eval_points)
            )
        });
    });
}

fn bench_gpu_fri_fold(c: &mut Criterion) {
    let prover = GpuStarkProver::new(true);

    // Benchmark FRI folding with typical layer sizes
    let layer: Vec<u32> = (0..8192).map(|i| (i % 256) as u32).collect();
    let challenges = vec![0x9E3779B9u32, 0x7F4A7C15u32, 0x2B7E1516u32, 0x384F4B89u32];

    c.bench_function("gpu_fri_fold_8192elements", |b| {
        b.iter(|| {
            prover.prove_gpu(
                black_box(&vec![layer.clone()]),
                black_box(&challenges)
            )
        });
    });
}

fn bench_gpu_merkle_tree(c: &mut Criterion) {
    let prover = GpuStarkProver::new(true);

    // Benchmark Merkle tree construction
    let leaves: Vec<u32> = (0..8192).map(|i| (i % 256) as u32).collect();

    c.bench_function("gpu_merkle_tree_8192leaves", |b| {
        b.iter(|| {
            let _ = black_box(prover.prove_gpu(
                black_box(&vec![leaves.clone()]),
                black_box(&[])
            ));
        });
    });
}

fn bench_cpu_vs_gpu_prove(c: &mut Criterion) {
    let cpu_prover = GpuStarkProver::new(false);
    let gpu_prover = GpuStarkProver::new(true);

    // Typical STARK trace: 8192 rows x 64 columns
    let trace: Vec<Vec<u32>> = (0..8192)
        .map(|i| (0..64).map(|j| ((i + j) % 256) as u32).collect())
        .collect();
    let public_inputs: Vec<u32> = (0..64).map(|i| (i % 256) as u32).collect();

    c.bench_function("cpu_stark_prove_8192x64", |b| {
        b.iter(|| {
            cpu_prover.prove(black_box(&trace), black_box(&public_inputs))
        });
    });

    c.bench_function("gpu_stark_prove_8192x64", |b| {
        b.iter(|| {
            gpu_prover.prove(black_box(&trace), black_box(&public_inputs))
        });
    });
}

criterion_group!(
    stark_benches,
    bench_gpu_lde_eval,
    bench_gpu_fri_fold,
    bench_gpu_merkle_tree,
    bench_cpu_vs_gpu_prove
);
criterion_main!(stark_benches);
