//! CUDA runtime bindings for GPU-accelerated STARK proving.
//!
//! This module provides a safe Rust interface to CUDA operations.
//! When the `gpu_prove` feature is enabled, it links to the actual CUDA runtime.
//! When disabled, it provides stub implementations that panic if called.

use anyhow::Result;
use std::ffi::c_void;

// Minimal CUDA driver API FFI bindings
#[cfg(feature = "gpu_prove")]
mod cuda_ffi {
    use std::ffi::c_void;

    pub type CUresult = i32;
    pub type CUdeviceptr = u64;
    pub type CUmodule = *mut c_void;
    pub type CUfunction = *mut c_void;
    pub type CUstream = *mut c_void;

    pub const CUDA_SUCCESS: CUresult = 0;

    unsafe extern "C" {
        pub fn cuInit(flags: u32) -> CUresult;
        pub fn cuDriverGetVersion(version: *mut i32) -> CUresult;
        pub fn cuDeviceGet(device: *mut CUdeviceptr, ordinal: i32) -> CUresult;
        pub fn cuCtxCreate(context: *mut CUdeviceptr, flags: u32, device: CUdeviceptr) -> CUresult;
        pub fn cuMemAlloc(ptr: *mut CUdeviceptr, size: usize) -> CUresult;
        pub fn cuMemFree(ptr: CUdeviceptr) -> CUresult;
        pub fn cuMemcpyHtoD(dst: CUdeviceptr, src: *const c_void, size: usize) -> CUresult;
        pub fn cuMemcpyDtoH(dst: *mut c_void, src: CUdeviceptr, size: usize) -> CUresult;
        pub fn cuModuleLoad(module: *mut CUmodule, fname: *const u8) -> CUresult;
        pub fn cuModuleGetFunction(func: *mut CUfunction, module: CUmodule, name: *const u8) -> CUresult;
        pub fn cuLaunchKernel(
            func: CUfunction,
            grid_x: u32, grid_y: u32, grid_z: u32,
            block_x: u32, block_y: u32, block_z: u32,
            shared_mem: u32,
            stream: CUstream,
            params: *mut *mut c_void,
            extra: *mut c_void,
        ) -> CUresult;
    }
}

#[cfg(feature = "gpu_prove")]
use cuda_ffi::*;

/// Initialize CUDA runtime.
pub fn init() -> Result<()> {
    #[cfg(feature = "gpu_prove")]
    unsafe {
        let mut driver_version = 0;
        let status = cuInit(0);
        if status != CUDA_SUCCESS {
            anyhow::bail!("Failed to initialize CUDA driver");
        }
        let status = cuDriverGetVersion(&mut driver_version);
        if status != CUDA_SUCCESS {
            anyhow::bail!("Failed to get CUDA driver version");
        }
        Ok(())
    }
    #[cfg(not(feature = "gpu_prove"))]
    {
        let _ = (&mut 0u8, &0usize, &0u32);
        Ok(())
    }
}

/// Allocate device memory.
pub fn malloc(ptr: *mut *mut c_void, size: usize) -> Result<()> {
    #[cfg(feature = "gpu_prove")]
    unsafe {
        let mut device_ptr: CUdeviceptr = 0;
        let status = cuMemAlloc(&mut device_ptr, size);
        if status != CUDA_SUCCESS {
            anyhow::bail!("Failed to allocate device memory");
        }
        *ptr = device_ptr as *mut c_void;
        Ok(())
    }
    #[cfg(not(feature = "gpu_prove"))]
    {
        let _ = ptr;
        let _ = size;
        anyhow::bail!("CUDA not available - build with --features gpu_prove")
    }
}

/// Free device memory.
pub fn free(ptr: *mut c_void) -> Result<()> {
    #[cfg(feature = "gpu_prove")]
    unsafe {
        let device_ptr = ptr as CUdeviceptr;
        let status = cuMemFree(device_ptr);
        if status != CUDA_SUCCESS {
            anyhow::bail!("Failed to free device memory");
        }
        Ok(())
    }
    #[cfg(not(feature = "gpu_prove"))]
    {
        let _ = ptr;
        anyhow::bail!("CUDA not available - build with --features gpu_prove")
    }
}

/// Copy memory from host to device.
pub fn memcpy_host_to_device(
    dst: *mut c_void,
    src: *const c_void,
    size: usize,
) -> Result<()> {
    #[cfg(feature = "gpu_prove")]
    unsafe {
        let status = cuMemcpyHtoD(dst as CUdeviceptr, src, size);
        if status != CUDA_SUCCESS {
            anyhow::bail!("Failed to copy host to device");
        }
        Ok(())
    }
    #[cfg(not(feature = "gpu_prove"))]
    {
        let _ = dst;
        let _ = src;
        let _ = size;
        anyhow::bail!("CUDA not available - build with --features gpu_prove")
    }
}

/// Copy memory from device to host.
pub fn memcpy_device_to_host(
    dst: *mut c_void,
    src: *const c_void,
    size: usize,
) -> Result<()> {
    #[cfg(feature = "gpu_prove")]
    unsafe {
        let status = cuMemcpyDtoH(dst, src as CUdeviceptr, size);
        if status != CUDA_SUCCESS {
            anyhow::bail!("Failed to copy device to host");
        }
        Ok(())
    }
    #[cfg(not(feature = "gpu_prove"))]
    {
        let _ = dst;
        let _ = src;
        let _ = size;
        anyhow::bail!("CUDA not available - build with --features gpu_prove")
    }
}

/// Load a CUDA kernel from PTX file.
pub fn load_kernel(name: &str, _ptx_path: &str) -> Result<KernelHandle> {
    #[cfg(feature = "gpu_prove")]
    {
        let _ = name;
        let _ = _ptx_path;
        anyhow::bail!("PTX kernel loading not yet implemented");
    }
    #[cfg(not(feature = "gpu_prove"))]
    {
        let _ = name;
        let _ = _ptx_path;
        anyhow::bail!("CUDA not available - build with --features gpu_prove")
    }
}

/// Get a loaded kernel by name.
pub fn get_kernel(_name: &str) -> KernelHandle {
    #[cfg(feature = "gpu_prove")]
    {
        let _ = _name;
        KernelHandle { _private: () }
    }
    #[cfg(not(feature = "gpu_prove"))]
    {
        let _ = _name;
        panic!("CUDA not available - build with --features gpu_prove")
    }
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
        #[cfg(feature = "gpu_prove")]
        {
            let _ = grid_x;
            let _ = grid_y;
            let _ = grid_z;
            let _ = block_x;
            let _ = block_y;
            let _ = block_z;
            let _ = shared_mem;
            let _ = _stream;
            let _ = _params;
            panic!("Kernel launch requires PTX module loading to be implemented first");
        }
        #[cfg(not(feature = "gpu_prove"))]
        {
            let _ = grid_x;
            let _ = grid_y;
            let _ = grid_z;
            let _ = block_x;
            let _ = block_y;
            let _ = block_z;
            let _ = shared_mem;
            let _ = _stream;
            let _ = _params;
            panic!("CUDA not available - build with --features gpu_prove");
        }
    }
}
