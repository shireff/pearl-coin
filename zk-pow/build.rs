use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let pearl_gemm_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("miner")
        .join("pearl-gemm");

    let extension_name = if cfg!(target_os = "windows") {
        "pearl_gemm_cuda.pyd"
    } else {
        "pearl_gemm_cuda.so"
    };

    #[cfg(feature = "gpu_prove")]
    {
        let mut found = false;
        let search_paths = [
            pearl_gemm_dir.join("src").join("pearl_gemm").join(extension_name),
            pearl_gemm_dir.join("build").join("lib").join(extension_name),
        ];

        for path in &search_paths {
            if path.exists() {
                found = true;
                println!("cargo:warning=Found pearl_gemm_cuda at {}", path.display());
                break;
            }
        }

        if !found {
            println!("cargo:warning=pearl_gemm_cuda not found, attempting to build...");
            
            let build_script = pearl_gemm_dir.join("build_inplace.py");
            if build_script.exists() {
                let python = env::var("PYTHON").unwrap_or_else(|_| "python".to_string());
                let status = Command::new(&python)
                    .arg(&build_script)
                    .current_dir(&pearl_gemm_dir)
                    .status();

                if status.map(|s| s.success()).unwrap_or(false) {
                    println!("cargo:warning=Successfully built pearl_gemm_cuda");
                } else {
                    println!("cargo:warning=Failed to build pearl_gemm_cuda, will fall back to CPU");
                }
            } else {
                println!("cargo:warning=build_inplace.py not found at {}, will fall back to CPU", build_script.display());
            }
        }

        println!("cargo:rerun-if-changed=build.rs");
    }

    #[cfg(not(feature = "gpu_prove"))]
    {
        let _ = pearl_gemm_dir;
        let _ = extension_name;
    }
}
