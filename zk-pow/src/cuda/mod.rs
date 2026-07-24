//! CUDA runtime bindings for GPU-accelerated STARK proving.
//!
//! This module provides a safe Rust interface to CUDA operations.
//! When the `gpu_prove` feature is enabled, it links to the actual CUDA runtime.
//! When disabled, it provides stub implementations that panic if called.

use std::ffi::c_void;

/// Initialize CUDA runtime.
/// Returns Ok(()) on success, Err if CUDA is not available or initialization fails.
pub fn init() -> Result<(), &'static str> {
    Err("CUDA not available - build with --features gpu_prove on a GPU-enabled machine")
}

/// Allocate device memory.
pub fn malloc(ptr: *mut *mut c_void, size: usize) -> Result<(), &'static str> {
    let _ = ptr;
    let _ = size;
    Err("CUDA not available")
}

/// Free device memory.
pub fn free(ptr: *mut c_void) -> Result<(), &'static str> {
    let _ = ptr;
    Err("CUDA not available")
}

/// Copy memory from host to device.
pub fn memcpy_host_to_device(
    dst: *mut c_void,
    src: *const c_void,
    size: usize,
) -> Result<(), &'static str> {
    let _ = dst;
    let _ = src;
    let _ = size;
    Err("CUDA not available")
}

/// Copy memory from device to host.
pub fn memcpy_device_to_host(
    dst: *mut c_void,
    src: *const c_void,
    size: usize,
) -> Result<(), &'static str> {
    let _ = dst;
    let _ = src;
    let _ = size;
    Err("CUDA not available")
}

/// Load a CUDA kernel from PTX file.
pub fn load_kernel(name: &str, _ptx_path: &str) -> Result<KernelHandle, &'static str> {
    let _ = name;
    Err("CUDA not available")
}

/// Get a loaded kernel by name.
pub fn get_kernel(_name: &str) -> KernelHandle {
    panic!("CUDA not available - build with --features gpu_prove on a GPU-enabled machine")
}

/// Handle to a loaded CUDA kernel.
pub struct KernelHandle {
    _private: (),
}

impl KernelHandle {
    /// Launch the kernel with the given configuration.
    pub unsafe fn launch(
        &self,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
        shared_mem: u32,
        _stream: *const c_void,
        _params: *mut *mut c_void,
    ) {
        let _ = grid_x;
        let _ = grid_y;
        let _ = grid_z;
        let _ = block_x;
        let _ = block_y;
        let _ = block_z;
        let _ = shared_mem;
        panic!("CUDA not available - build with --features gpu_prove on a GPU-enabled machine")
    }
}
