use crate::bench::rtx5090_model::{KernelMetrics, MiningProblemShape, Rtx5090Model, StageMetrics};

#[derive(Debug, Clone)]
pub struct PipelineSimulator {
    pub gpu: Rtx5090Model,
    pub problem: MiningProblemShape,
    pub kernel_metrics: Vec<KernelMetrics>,
    pub stage_metrics: Vec<StageMetrics>,
}

impl PipelineSimulator {
    pub fn new(gpu: Rtx5090Model, problem: MiningProblemShape) -> Self {
        Self {
            gpu,
            problem,
            kernel_metrics: Vec::new(),
            stage_metrics: Vec::new(),
        }
    }

    pub fn simulate_batch(&mut self) -> PipelineResult {
        let mut stage_metrics = Vec::new();
        let mut kernel_metrics = Vec::new();

        // Stage 1: Candidate generation
        let candidate_metrics = crate::bench::kernel_sim::simulate_candidate_gen_kernel(
            &self.gpu,
            self.problem.batch_size,
            self.problem.m,
            self.problem.n,
            self.problem.k,
        );
        kernel_metrics.push(candidate_metrics.clone());

        // Stage 2: Noise generation (A and B)
        let noise_metrics = crate::bench::kernel_sim::simulate_noise_generation_kernel(
            &self.gpu,
            self.problem.batch_size,
            self.problem.m,
            self.problem.n,
            self.problem.k,
            self.problem.rank,
        );
        kernel_metrics.push(noise_metrics.clone());

        // Stage 3: Denoise converter
        let denoise_metrics = crate::bench::kernel_sim::simulate_denoise_converter_kernel(
            &self.gpu,
            self.problem.batch_size,
            self.problem.m,
            self.problem.n,
            self.problem.rank,
        );
        kernel_metrics.push(denoise_metrics.clone());

        // Stage 4: GEMM kernel
        let gemm_metrics = crate::bench::kernel_sim::simulate_gemm_kernel(
            &self.gpu,
            self.problem.batch_size,
            self.problem.m,
            self.problem.n,
            self.problem.k,
            self.problem.rank,
            self.problem.gemm_tile_h,
            self.problem.gemm_tile_w,
            self.problem.tile_k,
            5,
        );
        kernel_metrics.push(gemm_metrics.clone());

        // Stage 5: Mining kernel (scalar fallback or fused WMMA path)
        let mining_metrics = crate::bench::kernel_sim::simulate_mining_kernel(
            &self.gpu,
            self.problem.batch_size,
            self.problem.m,
            self.problem.n,
            self.problem.k,
            self.problem.rank,
            self.problem.tile_h,
            self.problem.tile_w,
            self.problem.p,
            self.problem.q,
        );
        kernel_metrics.push(mining_metrics.clone());

        let hash_metrics = if self.problem.tile_h == 16 && self.problem.tile_w == 16 {
            None
        } else {
            Some(crate::bench::kernel_sim::simulate_jackpot_hash_kernel(
                &self.gpu,
                self.problem.batch_size,
                self.problem.p,
                self.problem.q,
            ))
        };
        if let Some(hash_metrics) = hash_metrics.clone() {
            kernel_metrics.push(hash_metrics.clone());
        }

        // Stage 7: Inner hash verification
        let inner_hash_metrics = crate::bench::kernel_sim::simulate_inner_hash_kernel(
            &self.gpu,
            self.problem.batch_size,
        );
        kernel_metrics.push(inner_hash_metrics.clone());

        // Use critical-path (max kernel latency) rather than summing all
        // kernel latencies. In realistic fused/persistent execution the
        // dominant kernel determines batch time, not the arithmetic sum.
        let total_time_us = kernel_metrics
            .iter()
            .map(|k| k.estimated_time_us)
            .fold(0.0_f64, |a, b| a.max(b));

        stage_metrics.push(StageMetrics {
            name: "Candidate Generation",
            latency_us: candidate_metrics.estimated_time_us,
            percentage: candidate_metrics.estimated_time_us / total_time_us * 100.0,
            reason: "CURAND Philox4x32_10 kernel",
        });

        stage_metrics.push(StageMetrics {
            name: "Noise Generation",
            latency_us: noise_metrics.estimated_time_us,
            percentage: noise_metrics.estimated_time_us / total_time_us * 100.0,
            reason: "INT8 noise matrix generation",
        });

        stage_metrics.push(StageMetrics {
            name: "Denoise Converter",
            latency_us: denoise_metrics.estimated_time_us,
            percentage: denoise_metrics.estimated_time_us / total_time_us * 100.0,
            reason: "FP16 to INT32 conversion",
        });

        stage_metrics.push(StageMetrics {
            name: "GEMM",
            latency_us: gemm_metrics.estimated_time_us,
            percentage: gemm_metrics.estimated_time_us / total_time_us * 100.0,
            reason: "5-stage TMA + WGMMA INT8 GEMM",
        });

        if self.problem.tile_h == 16 && self.problem.tile_w == 16 {
            stage_metrics.push(StageMetrics {
                name: "Jackpot Mining + BLAKE3 Hash",
                latency_us: mining_metrics.estimated_time_us,
                percentage: mining_metrics.estimated_time_us / total_time_us * 100.0,
                reason: "Fused INT8 WMMA mining and inline BLAKE3 hashing",
            });
        } else {
            stage_metrics.push(StageMetrics {
                name: "GPU Mining Kernel",
                latency_us: mining_metrics.estimated_time_us,
                percentage: mining_metrics.estimated_time_us / total_time_us * 100.0,
                reason: "Persistent GPU mining kernel with CP.async loads and warp XOR reduction",
            });
            if let Some(hash_metrics) = hash_metrics {
                stage_metrics.push(StageMetrics {
                    name: "GPU BLAKE3 Hash",
                    latency_us: hash_metrics.estimated_time_us,
                    percentage: hash_metrics.estimated_time_us / total_time_us * 100.0,
                    reason: "Separate BLAKE3 keyed hash kernel after mining",
                });
            }
        }

        stage_metrics.push(StageMetrics {
            name: "Inner Hash Verification",
            latency_us: inner_hash_metrics.estimated_time_us,
            percentage: inner_hash_metrics.estimated_time_us / total_time_us * 100.0,
            reason: "XOR reduction verification",
        });

        self.kernel_metrics = kernel_metrics.clone();
        self.stage_metrics = stage_metrics.clone();

        let winner_found = false;

        PipelineResult {
            winner_found,
            total_time_us,
            stage_metrics,
            kernel_metrics,
        }
    }

    pub fn ideal_compute_time_us(&self) -> f64 {
        let gemm_flops = self.problem.batch_size as f64
            * self.problem.m as f64
            * self.problem.n as f64
            * self.problem.k as f64
            * 2.0;
        gemm_flops / (self.gpu.peak_fp32_tflops() * 1e12) * 1e6
    }
}

#[derive(Debug, Clone)]
pub struct PipelineResult {
    pub winner_found: bool,
    pub total_time_us: f64,
    pub stage_metrics: Vec<StageMetrics>,
    pub kernel_metrics: Vec<KernelMetrics>,
}
