#![allow(clippy::needless_range_loop)]

pub mod api;
pub mod circuit;
pub mod cuda;
pub mod ffi;
pub mod v1;

#[cfg(feature = "gpu_prove")]
pub mod gpu_stark_prover;

#[cfg(feature = "gpu_prove")]
pub mod gpu_mining;
