// ============================================================================
// RTX 5090 Hardware Model (Blackwell SM100)
// ============================================================================
// Source: NVIDIA public specifications and CUDA programming guide.
// All values are hard-coded and do NOT depend on the host machine.

#[derive(Debug, Clone)]
pub struct Rtx5090Model {
    pub name: &'static str,
    pub architecture: &'static str,
    pub cuda_cores: u32,
    pub boost_clock_mhz: u32,
    pub memory_bytes: u64,
    pub memory_bus_width: u32,
    pub memory_bandwidth_gbps: u32,
    pub pcie_generation: u32,
    pub tensor_cores: &'static str,
    pub warp_size: u32,
    pub max_threads_per_sm: u32,
    pub max_warps_per_sm: u32,
    pub max_blocks_per_sm: u32,
    pub registers_per_sm: u32,
    pub max_registers_per_thread: u32,
    pub shared_mem_per_block_optin: u32,
    pub max_shared_mem_per_block: u32,
    pub sm_count: u32,
    pub l2_cache_bytes: u32,
    pub tma_latency_cycles: u32,
    pub kernel_launch_overhead_us: f64,
    pub pcie_transfer_overhead_us: f64,
}

impl Rtx5090Model {
    pub const fn new() -> Self {
        Self {
            name: "NVIDIA RTX 5090",
            architecture: "Blackwell",
            cuda_cores: 21760,
            boost_clock_mhz: 2407,
            memory_bytes: 32u64 * 1024 * 1024 * 1024,
            memory_bus_width: 512,
            memory_bandwidth_gbps: 1792,
            pcie_generation: 5,
            tensor_cores: "Blackwell Tensor Cores",
            warp_size: 32,
            max_threads_per_sm: 2048,
            max_warps_per_sm: 64,
            max_blocks_per_sm: 32,
            registers_per_sm: 65536,
            max_registers_per_thread: 255,
            shared_mem_per_block_optin: 228 * 1024,
            max_shared_mem_per_block: 228 * 1024,
            sm_count: 170,
            l2_cache_bytes: 64 * 1024 * 1024,  // 64 MB (estimated)
            tma_latency_cycles: 250,
            kernel_launch_overhead_us: 4.0,
            pcie_transfer_overhead_us: 2.0,
        }
    }

    pub fn peak_fp32_tflops(&self) -> f64 {
        let ops_per_cycle = self.cuda_cores as f64 * 2.0;
        let cycles_per_sec = self.boost_clock_mhz as f64 * 1e6;
        ops_per_cycle * cycles_per_sec / 1e12
    }

    pub fn peak_fp16_tflops(&self) -> f64 {
        self.peak_fp32_tflops() * 2.0
    }

    pub fn peak_int8_tops(&self) -> f64 {
        self.peak_fp32_tflops() * 4.0
    }

    pub fn peak_memory_bandwidth_gbps(&self) -> f64 {
        self.memory_bandwidth_gbps as f64
    }

    pub fn memory_bandwidth_bytes_per_sec(&self) -> f64 {
        self.memory_bandwidth_gbps as f64 * 1e9
    }

    pub fn clock_cycles_per_sec(&self) -> f64 {
        self.boost_clock_mhz as f64 * 1e6
    }

    pub fn registers_per_block(&self, threads_per_block: u32) -> u64 {
        self.registers_per_sm as u64 * self.max_blocks_per_sm as u64
            / (self.max_threads_per_sm as u64 / threads_per_block as u64)
    }
}

impl Default for Rtx5090Model {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Kernel Execution Metrics
// ============================================================================

#[derive(Debug, Clone)]
pub struct KernelMetrics {
    pub name: &'static str,
    pub launch_overhead_us: f64,
    pub grid_size: u32,
    pub block_size: u32,
    pub shared_mem_per_block: u32,
    pub registers_per_thread: u32,
    pub occupancy_ratio: f64,
    pub active_warps_per_sm: u32,
    pub total_warps: u32,
    pub estimated_cycles: u64,
    pub estimated_time_us: f64,
    pub compute_time_us: f64,
    pub memory_time_us: f64,
    pub tma_time_us: f64,
    pub global_reads_bytes: u64,
    pub global_writes_bytes: u64,
    pub l2_traffic_bytes: u64,
    pub dram_traffic_bytes: u64,
    pub instruction_count: u64,
    pub warp_efficiency: f64,
    pub tensor_core_utilization: f64,
    pub compute_utilization: f64,
    pub memory_utilization: f64,
    pub branch_divergence: f64,
    pub register_spill_bytes: u64,
    pub bottleneck: &'static str,
}

// ============================================================================
// Pipeline Stage Metrics
// ============================================================================

#[derive(Debug, Clone)]
pub struct StageMetrics {
    pub name: &'static str,
    pub latency_us: f64,
    pub percentage: f64,
    pub reason: &'static str,
}

// ============================================================================
// Bottleneck Report
// ============================================================================

#[derive(Debug, Clone)]
pub struct BottleneckReport {
    pub component: &'static str,
    pub severity: &'static str,
    pub latency_us: f64,
    pub percentage: f64,
    pub reason: String,
    pub recommended_fix: &'static str,
    pub expected_gain_percent: f64,
    pub difficulty: &'static str,
    pub risk: &'static str,
}

// ============================================================================
// Mining Problem Shape
// ============================================================================

#[derive(Debug, Clone, Copy)]
pub struct MiningProblemShape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub rank: usize,
    pub tile_h: usize,
    pub tile_w: usize,
    pub tile_k: usize,
    pub p: usize,
    pub q: usize,
    pub batch_size: usize,
    pub skip_denoising: bool,
    pub skip_reduction: bool,
}

impl Default for MiningProblemShape {
    fn default() -> Self {
        Self {
            m: 1024,
            n: 1024,
            k: 1024,
            rank: 32,
            tile_h: 128,
            tile_w: 256,
            tile_k: 128,
            p: 8,
            q: 8,
            batch_size: 256,
            skip_denoising: true,
            skip_reduction: true,
        }
    }
}

impl MiningProblemShape {
    pub fn num_combos(&self) -> usize {
        self.p * self.q
    }

    pub fn total_candidates(&self) -> usize {
        self.batch_size * self.num_combos()
    }
}
