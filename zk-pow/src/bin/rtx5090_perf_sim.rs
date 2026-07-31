use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    let batch_size = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);

    let m = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);

    let n = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);

    let k = args
        .get(4)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1024);

    let rank = args
        .get(5)
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);

    let problem = zk_pow::bench::MiningProblemShape {
        m,
        n,
        k,
        rank,
        batch_size,
        ..Default::default()
    };

    let mut benchmark = zk_pow::bench::PerfBenchmark::new(problem);
    let _ = &benchmark.run();
    // Print per-kernel and per-stage timings to identify the dominant kernel
    println!("\nPer-kernel timings (us):");
    for k in &benchmark.pipeline.kernel_metrics {
        println!("- {}: {:.3} us", k.name, k.estimated_time_us);
    }
    println!("\nPer-stage timings (us):");
    for s in &benchmark.pipeline.stage_metrics {
        println!("- {}: {:.3} us ({}%) - {}", s.name, s.latency_us, s.percentage, s.reason);
    }
    benchmark.print_summary();
}
