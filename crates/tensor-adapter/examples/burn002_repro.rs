//! Standalone reproduction for the `burn-wgpu` reduction-sentinel bug (issue draft 002).
//!
//! Deliberately uses **only `burn`'s own API** — no part of this project's engine, adapter
//! or comparison logic — so that a maintainer can paste it into a fresh crate and run it.
//! A reproduction that needs our tool to observe the bug is not a reproduction.

use burn::backend::{Wgpu, wgpu::WgpuDevice};
use burn::tensor::{Tensor, TensorData};

type Cpu = burn::backend::Flex;
type Gpu = Wgpu;

fn show(label: &str, cpu: TensorData, gpu: TensorData) {
    let c = cpu.to_vec::<f32>().unwrap();
    let g = gpu.to_vec::<f32>().unwrap();
    let flag = if c == g { "  " } else { "<-" };
    println!("{flag} {label:<34} flex={c:?}   wgpu={g:?}");
}

fn main() {
    let cpu_device = Default::default();
    let gpu_device = WgpuDevice::default();
    let ninf = f32::NEG_INFINITY;
    let inf = f32::INFINITY;

    println!("\nmax over an axis whose true maximum is -inf\n");

    // Reduced axis of length 2, every element -inf. Expected [-inf].
    let c = Tensor::<Cpu, 1>::from_floats([ninf, ninf], &cpu_device).max_dim(0);
    let g = Tensor::<Gpu, 1>::from_floats([ninf, ninf], &gpu_device).max_dim(0);
    show("max([-inf, -inf])", c.to_data(), g.to_data());

    // Reduced axis of length **one**: the reduction is an identity, yet the value changes.
    let c = Tensor::<Cpu, 2>::from_floats([[0.0, -1e30, ninf, -3.0]], &cpu_device).max_dim(0);
    let g = Tensor::<Gpu, 2>::from_floats([[0.0, -1e30, ninf, -3.0]], &gpu_device).max_dim(0);
    show("max over axis of length 1", c.to_data(), g.to_data());

    println!("\nmin over an axis whose true minimum is +inf\n");

    let c = Tensor::<Cpu, 1>::from_floats([inf, inf], &cpu_device).min_dim(0);
    let g = Tensor::<Gpu, 1>::from_floats([inf, inf], &gpu_device).min_dim(0);
    show("min([+inf, +inf])", c.to_data(), g.to_data());

    println!("\nunaffected: a finite element is present, or the operation is sum\n");

    let c = Tensor::<Cpu, 1>::from_floats([ninf, -5.0], &cpu_device).max_dim(0);
    let g = Tensor::<Gpu, 1>::from_floats([ninf, -5.0], &gpu_device).max_dim(0);
    show("max([-inf, -5])", c.to_data(), g.to_data());

    let c = Tensor::<Cpu, 1>::from_floats([ninf, ninf], &cpu_device).sum_dim(0);
    let g = Tensor::<Gpu, 1>::from_floats([ninf, ninf], &gpu_device).sum_dim(0);
    show("sum([-inf, -inf])", c.to_data(), g.to_data());

    println!();
}
