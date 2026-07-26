use crate::bench::rtx5090_model::{KernelMetrics, Rtx5090Model};

// ============================================================================
// Kernel Execution Simulators
// Derived from actual kernel source code analysis
// ============================================================================

pub fn simulate_candidate_gen_kernel(
    gpu: &Rtx5090Model,
    batch_size: usize,
    m: usize,
    n: usize,
    k: usize,
) -> KernelMetrics {
    let threads_per_block = 256u32;
    let grid_size = batch_size as u32;
    let block_size = threads_per_block;

    let a_elements = (m * k) as u64;
    let b_elements = (n * k) as u64;
    let total_bytes = (batch_size as u64 * (a_elements + b_elements) * 2) as u64;

    let memory_time_us = total_bytes as f64 / gpu.memory_bandwidth_bytes_per_sec() * 1e6;
    let estimated_time_us = memory_time_us + gpu.kernel_launch_overhead_us;

    KernelMetrics {
        name: "gpu_generate_raw_matrices_kernel",
        launch_overhead_us: gpu.kernel_launch_overhead_us,
        grid_size,
        block_size,
        shared_mem_per_block: 0,
        registers_per_thread: 16,
        occupancy_ratio: 1.0,
        active_warps_per_sm: 64,
        total_warps: grid_size * (block_size / gpu.warp_size),
        estimated_cycles: (estimated_time_us * gpu.clock_cycles_per_sec() / 1e6) as u64,
        estimated_time_us,
        compute_time_us: 0.0,
        memory_time_us,
        tma_time_us: 0.0,
        global_reads_bytes: 0,
        global_writes_bytes: total_bytes,
        l2_traffic_bytes: total_bytes,
        dram_traffic_bytes: total_bytes,
        instruction_count: (total_bytes / 4) as u64,
        warp_efficiency: 0.95,
        tensor_core_utilization: 0.0,
        compute_utilization: 0.1,
        memory_utilization: 0.9,
        branch_divergence: 0.0,
        register_spill_bytes: 0,
        bottleneck: "Memory Bandwidth",
    }
}

pub fn simulate_noise_generation_kernel(
    gpu: &Rtx5090Model,
    batch_size: usize,
    m: usize,
    n: usize,
    k: usize,
    _rank: usize,
) -> KernelMetrics {
    let total_elements = (batch_size * (m * k + n * k)) as u64;
    let total_bytes = total_elements * 2;
    let memory_time_us = (total_bytes * 2) as f64 / gpu.memory_bandwidth_bytes_per_sec() * 1e6;
    let estimated_time_us = memory_time_us + gpu.kernel_launch_overhead_us;

    KernelMetrics {
        name: "run_noise_generation (A+B)",
        launch_overhead_us: gpu.kernel_launch_overhead_us,
        grid_size: ((batch_size * m * k) as f64 / 256.0).ceil() as u32,
        block_size: 256,
        shared_mem_per_block: 0,
        registers_per_thread: 24,
        occupancy_ratio: 0.8,
        active_warps_per_sm: 48,
        total_warps: ((batch_size * m * k / 256) as u32).max(1),
        estimated_cycles: (estimated_time_us * gpu.clock_cycles_per_sec() / 1e6) as u64,
        estimated_time_us,
        compute_time_us: 0.0,
        memory_time_us,
        tma_time_us: 0.0,
        global_reads_bytes: total_bytes,
        global_writes_bytes: total_bytes,
        l2_traffic_bytes: total_bytes,
        dram_traffic_bytes: total_bytes * 2,
        instruction_count: (total_elements * 4) as u64,
        warp_efficiency: 0.9,
        tensor_core_utilization: 0.0,
        compute_utilization: 0.3,
        memory_utilization: 0.85,
        branch_divergence: 0.0,
        register_spill_bytes: 0,
        bottleneck: "Memory Bandwidth",
    }
}

pub fn simulate_denoise_converter_kernel(
    gpu: &Rtx5090Model,
    batch_size: usize,
    m: usize,
    n: usize,
    _rank: usize,
) -> KernelMetrics {
    let total_elements = (batch_size * (m + n)) as u64;
    let total_bytes = total_elements * 2;
    let memory_time_us = (total_bytes * 2) as f64 / gpu.memory_bandwidth_bytes_per_sec() * 1e6;
    let estimated_time_us = memory_time_us + gpu.kernel_launch_overhead_us;

    KernelMetrics {
        name: "run_denoise_converter",
        launch_overhead_us: gpu.kernel_launch_overhead_us,
        grid_size: ((batch_size * (m + n)) as f64 / 256.0).ceil() as u32,
        block_size: 256,
        shared_mem_per_block: 0,
        registers_per_thread: 48,
        occupancy_ratio: 0.75,
        active_warps_per_sm: 48,
        total_warps: ((batch_size * (m + n) / 256) as u32).max(1),
        estimated_cycles: (estimated_time_us * gpu.clock_cycles_per_sec() / 1e6) as u64,
        estimated_time_us,
        compute_time_us: 0.0,
        memory_time_us,
        tma_time_us: 0.0,
        global_reads_bytes: total_bytes,
        global_writes_bytes: total_bytes,
        l2_traffic_bytes: total_bytes,
        dram_traffic_bytes: total_bytes * 2,
        instruction_count: (total_elements * 6) as u64,
        warp_efficiency: 0.88,
        tensor_core_utilization: 0.0,
        compute_utilization: 0.4,
        memory_utilization: 0.7,
        branch_divergence: 0.0,
        register_spill_bytes: 0,
        bottleneck: "Compute (FP16 conversion)",
    }
}

pub fn simulate_gemm_kernel(
    gpu: &Rtx5090Model,
    batch_size: usize,
    m: usize,
    n: usize,
    k: usize,
    _rank: usize,
    tile_h: usize,
    tile_w: usize,
    tile_k: usize,
    stages: usize,
) -> KernelMetrics {
    let cta_size = if tile_h >= 256 { 512 } else if tile_h >= 128 { 256 } else { 128 };

    let blocks_m = (m + tile_h - 1) / tile_h;
    let blocks_n = (n + tile_w - 1) / tile_w;
    let k_blocks = (k + tile_k - 1) / tile_k;
    let total_tiles = blocks_m * blocks_n * k_blocks;
    let grid_size = batch_size * total_tiles;

    let tile_m = tile_h;
    let tile_n = tile_w;

    let smem_a = (tile_m * tile_k) as u32;
    let smem_b = (tile_n * tile_k) as u32;
    let smem_c = (tile_m * tile_n * 2) as u32;
    let smem_per_block = smem_a + smem_b + smem_c;

    let occupancy_sm = gpu.max_shared_mem_per_block / smem_per_block.max(1);
    let occupancy_ratio = (occupancy_sm as f64 / gpu.max_blocks_per_sm as f64).min(1.0);

    let total_macs = (grid_size * tile_m * tile_n * tile_k) as u64;
    let tensor_core_util = if smem_per_block <= gpu.shared_mem_per_block_optin { 0.85 } else { 0.55 };
    let peak_int8_tops = gpu.peak_fp32_tflops() * 4.0;
    let effective_tops = peak_int8_tops * tensor_core_util;
    let compute_time_us = (total_macs as f64 * 2.0) / (effective_tops * 1e12) * 1e6;

    let global_reads_a = (grid_size * tile_m * tile_k) as u64;
    let global_reads_b = (grid_size * tile_n * tile_k) as u64;
    let global_writes_c = (grid_size * tile_m * tile_n * 2) as u64;
    let total_memory_bytes = global_reads_a + global_reads_b + global_writes_c;
    let memory_time_us = total_memory_bytes as f64 / gpu.memory_bandwidth_bytes_per_sec() * 1e6;

    let tma_cycles = stages as u64 * gpu.tma_latency_cycles as u64;
    let tma_time_us = tma_cycles as f64 / gpu.clock_cycles_per_sec() * 1e6;

    let estimated_time_us = compute_time_us.max(memory_time_us) + tma_time_us + gpu.kernel_launch_overhead_us;

    KernelMetrics {
        name: "hopper_gemm_ws",
        launch_overhead_us: gpu.kernel_launch_overhead_us,
        grid_size: grid_size as u32,
        block_size: cta_size as u32,
        shared_mem_per_block: smem_per_block,
        registers_per_thread: 160,
        occupancy_ratio,
        active_warps_per_sm: (occupancy_ratio * gpu.max_warps_per_sm as f64) as u32,
        total_warps: (grid_size * cta_size / 32) as u32,
        estimated_cycles: (estimated_time_us * gpu.clock_cycles_per_sec() / 1e6) as u64,
        estimated_time_us,
        compute_time_us,
        memory_time_us,
        tma_time_us,
        global_reads_bytes: global_reads_a + global_reads_b,
        global_writes_bytes: global_writes_c,
        l2_traffic_bytes: total_memory_bytes / 2,
        dram_traffic_bytes: total_memory_bytes,
        instruction_count: (estimated_time_us * gpu.clock_cycles_per_sec() / 1e6 * 2.0) as u64,
        warp_efficiency: 0.92,
        tensor_core_utilization: tensor_core_util * 100.0,
        compute_utilization: (compute_time_us / estimated_time_us.max(1e-9)) * 100.0,
        memory_utilization: (memory_time_us / estimated_time_us.max(1e-9)) * 100.0,
        branch_divergence: 0.02,
        register_spill_bytes: 0,
        bottleneck: if compute_time_us > memory_time_us { "Compute (Tensor Core)" } else { "Memory" },
    }
}

pub fn simulate_fused_mining_wmma_kernel(
    gpu: &Rtx5090Model,
    batch_size: usize,
    _m: usize,
    _n: usize,
    _k: usize,
    rank: usize,
    _tile_h: usize,
    _tile_w: usize,
    p: usize,
    q: usize,
    k: usize,
) -> KernelMetrics {
    let num_combos = p * q;
    let total_work = batch_size * num_combos;
    let threads_per_block = 256u32;
    let warps_per_block = 8u32;
    let grid_size = ((total_work as u32 + warps_per_block - 1) / warps_per_block).max(1);
    let block_size = threads_per_block;

    let num_steps = k / rank;
    let total_macs = total_work as u64 * num_steps as u64 * 4096;
    let peak_int8 = gpu.peak_int8_tops() * 1e12;
    let compute_time_us = total_macs as f64 / peak_int8 * 1e6;

    let global_reads_bytes = total_work as u64 * num_steps as u64 * 512;
    let global_writes_bytes = total_work as u64 * 96;
    let total_memory_bytes = global_reads_bytes + global_writes_bytes;
    let memory_time_us = total_memory_bytes as f64 / gpu.memory_bandwidth_bytes_per_sec() * 1e6;

    let blake3_cycles = 1200u64;
    let blake3_time_us = total_work as f64 * blake3_cycles as f64 / gpu.clock_cycles_per_sec() * 1e6;

    let estimated_time_us = compute_time_us.max(memory_time_us * 0.7).max(blake3_time_us) + gpu.kernel_launch_overhead_us;

    KernelMetrics {
        name: "gpu_mining_wmma_kernel",
        launch_overhead_us: gpu.kernel_launch_overhead_us,
        grid_size,
        block_size,
        shared_mem_per_block: 256,
        registers_per_thread: 64,
        occupancy_ratio: 1.0,
        active_warps_per_sm: gpu.max_warps_per_sm,
        total_warps: grid_size * (block_size / gpu.warp_size),
        estimated_cycles: (estimated_time_us * gpu.clock_cycles_per_sec() / 1e6) as u64,
        estimated_time_us,
        compute_time_us,
        memory_time_us,
        tma_time_us: 0.0,
        global_reads_bytes,
        global_writes_bytes,
        l2_traffic_bytes: total_memory_bytes / 2,
        dram_traffic_bytes: total_memory_bytes,
        instruction_count: (total_macs * 2) + (total_work as u64 * 256),
        warp_efficiency: 0.95,
        tensor_core_utilization: 92.0,
        compute_utilization: (compute_time_us / estimated_time_us.max(1e-9)) * 100.0,
        memory_utilization: (memory_time_us / estimated_time_us.max(1e-9)) * 100.0,
        branch_divergence: 0.01,
        register_spill_bytes: 0,
        bottleneck: if memory_time_us > compute_time_us { "Memory Bandwidth" } else { "Tensor Core" },
    }
}

pub fn simulate_mining_kernel(
    gpu: &Rtx5090Model,
    batch_size: usize,
    _m: usize,
    _n: usize,
    _k: usize,
    _rank: usize,
    tile_h: usize,
    tile_w: usize,
    p: usize,
    q: usize,
) -> KernelMetrics {
    if tile_h == 16 && tile_w == 16 {
        return simulate_fused_mining_wmma_kernel(
            gpu,
            batch_size,
            _m,
            _n,
            _k,
            _rank,
            tile_h,
            tile_w,
            p,
            q,
            _k,
        );
    }

    let threads_per_block = 128u32;
    let grid_size = batch_size as u32;
    let block_size = threads_per_block;

    let num_combos = p * q;
    let total_evaluations = (batch_size * num_combos * tile_h * tile_w) as u64;
    let ops_per_eval = 4;
    let total_ops = total_evaluations * ops_per_eval;

    let peak_fp32 = gpu.peak_fp32_tflops() * 1e12;
    let effective_peak = peak_fp32 * 0.5;
    let compute_time_us = total_ops as f64 / effective_peak * 1e6;

    let shared_mem = ((tile_h * tile_w + tile_h * 33 + 33 * tile_w + 16) * 4) as u32;
    let occupancy_sm = gpu.max_shared_mem_per_block / shared_mem.max(1);
    let occupancy_ratio = (occupancy_sm as f64 / gpu.max_blocks_per_sm as f64).min(1.0);

    let estimated_time_us = compute_time_us + gpu.kernel_launch_overhead_us;

    KernelMetrics {
        name: "gpu_mining_kernel",
        launch_overhead_us: gpu.kernel_launch_overhead_us,
        grid_size,
        block_size,
        shared_mem_per_block: shared_mem,
        registers_per_thread: 80,
        occupancy_ratio,
        active_warps_per_sm: (occupancy_ratio * gpu.max_warps_per_sm as f64) as u32,
        total_warps: grid_size * (block_size / gpu.warp_size),
        estimated_cycles: (estimated_time_us * gpu.clock_cycles_per_sec() / 1e6) as u64,
        estimated_time_us,
        compute_time_us,
        memory_time_us: 0.0,
        tma_time_us: 0.0,
        global_reads_bytes: total_evaluations * 8,
        global_writes_bytes: (batch_size * num_combos * 16 * 4) as u64,
        l2_traffic_bytes: (total_evaluations * 8 + (batch_size * num_combos * 64) as u64) / 2,
        dram_traffic_bytes: total_evaluations * 8 + (batch_size * num_combos * 64) as u64,
        instruction_count: (total_ops * 2) as u64,
        warp_efficiency: 0.85,
        tensor_core_utilization: 0.0,
        compute_utilization: 0.6,
        memory_utilization: 0.3,
        branch_divergence: 0.05,
        register_spill_bytes: 0,
        bottleneck: "Compute",
    }
}

pub fn simulate_jackpot_hash_kernel(
    gpu: &Rtx5090Model,
    batch_size: usize,
    p: usize,
    q: usize,
) -> KernelMetrics {
    let threads_per_block = 256u32;
    let num_combos = p * q;
    let total_hashes = batch_size * num_combos;
    let grid_size = ((total_hashes + threads_per_block as usize - 1) / threads_per_block as usize) as u32;
    let block_size = threads_per_block;

    let cycles_per_hash = 1200u64;
    let total_cycles = (total_hashes * cycles_per_hash as usize) as u64;
    let compute_time_us = total_cycles as f64 / gpu.clock_cycles_per_sec() * 1e6;

    let memory_bytes = (total_hashes * 96) as u64;
    let memory_time_us = memory_bytes as f64 / gpu.memory_bandwidth_bytes_per_sec() * 1e6;
    let estimated_time_us = compute_time_us.max(memory_time_us) + gpu.kernel_launch_overhead_us;

    KernelMetrics {
        name: "gpu_jackpot_hash_kernel",
        launch_overhead_us: gpu.kernel_launch_overhead_us,
        grid_size,
        block_size,
        shared_mem_per_block: 0,
        registers_per_thread: 64,
        occupancy_ratio: 0.9,
        active_warps_per_sm: 58,
        total_warps: grid_size * (block_size / gpu.warp_size),
        estimated_cycles: total_cycles,
        estimated_time_us,
        compute_time_us,
        memory_time_us,
        tma_time_us: 0.0,
        global_reads_bytes: (total_hashes * 64) as u64,
        global_writes_bytes: (total_hashes * 32) as u64,
        l2_traffic_bytes: memory_bytes / 2,
        dram_traffic_bytes: memory_bytes,
        instruction_count: total_cycles * 2,
        warp_efficiency: 0.9,
        tensor_core_utilization: 0.0,
        compute_utilization: 0.45,
        memory_utilization: 0.55,
        branch_divergence: 0.0,
        register_spill_bytes: 0,
        bottleneck: "Compute",
    }
}

pub fn simulate_inner_hash_kernel(gpu: &Rtx5090Model, batch_size: usize) -> KernelMetrics {
    let grid_size = batch_size as u32;
    let block_size = 1;

    let cycles_per_hash = 12800u64;
    let total_cycles = (batch_size * cycles_per_hash as usize) as u64;
    let compute_time_us = total_cycles as f64 / gpu.clock_cycles_per_sec() * 1e6;
    let estimated_time_us = compute_time_us + gpu.kernel_launch_overhead_us;

    KernelMetrics {
        name: "inner_hash_kernel",
        launch_overhead_us: gpu.kernel_launch_overhead_us,
        grid_size,
        block_size,
        shared_mem_per_block: 0,
        registers_per_thread: 32,
        occupancy_ratio: 0.01,
        active_warps_per_sm: 1,
        total_warps: grid_size,
        estimated_cycles: total_cycles,
        estimated_time_us,
        compute_time_us,
        memory_time_us: 0.0,
        tma_time_us: 0.0,
        global_reads_bytes: 0,
        global_writes_bytes: 0,
        l2_traffic_bytes: 0,
        dram_traffic_bytes: 0,
        instruction_count: total_cycles * 2,
        warp_efficiency: 1.0,
        tensor_core_utilization: 0.0,
        compute_utilization: 0.1,
        memory_utilization: 0.0,
        branch_divergence: 0.0,
        register_spill_bytes: 0,
        bottleneck: "Launch Overhead",
    }
}
