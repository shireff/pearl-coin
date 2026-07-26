use std::env;
use std::path::PathBuf;

#[cfg(feature = "gpu_prove")]
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
        let mut lib_path = None;

        let search_paths = [
            pearl_gemm_dir.join("src").join("pearl_gemm").join(extension_name),
            pearl_gemm_dir.join("build").join("lib").join(extension_name),
            pearl_gemm_dir.join("build").join("lib").join("Release").join(extension_name),
            pearl_gemm_dir.join("pearl_gemm_cuda").join(extension_name),
        ];

        for path in &search_paths {
            if path.exists() {
                found = true;
                lib_path = Some(path);
                println!("cargo:warning=Found pearl_gemm_cuda at {}", path.display());
                break;
            }
        }

        if !found {
            println!("cargo:warning=pearl_gemm_cuda not found, attempting to build...");

            let setup_py = pearl_gemm_dir.join("setup.py");
            if setup_py.exists() {
                let python = env::var("PYTHON").unwrap_or_else(|_| "python".to_string());
                println!("cargo:warning=Running setup.py build_ext --inplace...");

                let status = Command::new(&python)
                    .arg("setup.py")
                    .arg("build_ext")
                    .arg("--inplace")
                    .current_dir(&pearl_gemm_dir)
                    .status();

                if status.map(|s| s.success()).unwrap_or(false) {
                    println!("cargo:warning=Build completed, searching for library...");

                    for path in &search_paths {
                        if path.exists() {
                            found = true;
                            lib_path = Some(path);
                            println!("cargo:warning=Found built pearl_gemm_cuda at {}", path.display());
                            break;
                        }
                    }
                } else {
                    println!("cargo:warning=Failed to build pearl_gemm_cuda");
                }
            } else {
                println!("cargo:warning=setup.py not found at {}", setup_py.display());
            }
        }

        if found {
            if let Some(path) = lib_path {
                if let Some(parent) = path.parent() {
                    println!("cargo:rustc-link-search=native={}", parent.display());
                }
            }
            println!("cargo:rustc-link-lib=dylib=pearl_gemm_cuda");
            println!("cargo:rustc-link-lib=dylib=cuda");
            println!("cargo:rustc-link-lib=dylib=cudart");
            println!("cargo:rerun-if-changed=build.rs");
        } else {
            panic!("\n\n\
                =======================================================================\n\
                ERROR: pearl_gemm_cuda not found and build failed.\n\
                \n\
                To fix this, either:\n\
                \n\
                1. Build the CUDA extension manually:\n\
                   cd {}\n\
                   python setup.py build_ext --inplace\n\
                \n\
                2. Or set the PEARL_GEMM_CUDA_PATH environment variable to the\n\
                   directory containing pearl_gemm_cuda.pyd/.so\n\
                \n\
                3. Build without GPU support:\n\
                   cargo build --no-default-features\n\
                =======================================================================\n",
                pearl_gemm_dir.display());
        }
    }

    #[cfg(not(feature = "gpu_prove"))]
    {
        let _ = pearl_gemm_dir;
        let _ = extension_name;
    }
}
