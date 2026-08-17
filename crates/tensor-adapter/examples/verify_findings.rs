//! Sanity check: does each candidate finding still reproduce, today, on this build?
//!
//! **Uses only `burn`'s public API** — no engine, adapter, oracle or normaliser. A finding
//! that needs our tool to observe it is not a finding a maintainer can act on, and a report
//! whose reproduction imports our crate will be closed unread.
//!
//! This exists because reproduction decays. The `ndarray`-to-`flex` backend swap invalidated
//! **810 of 814** recorded findings at a stroke; a report filed from a stale recording is
//! worse than no report. So every candidate is re-run from scratch before it is drafted.
//!
//! Run: `cargo run --release -p tensor-adapter --example verify_findings`

use burn::backend::{LibTorch, Wgpu, wgpu::WgpuDevice};
use burn::tensor::{Tensor, TensorData};

type Flex = burn::backend::Flex;
type Torch = LibTorch<f32>;
type Gpu = Wgpu;

fn row(label: &str, flex: TensorData, tch: TensorData, gpu: TensorData) -> bool {
    let f = flex.to_vec::<f32>().unwrap();
    let t = tch.to_vec::<f32>().unwrap();
    let g = gpu.to_vec::<f32>().unwrap();
    // Bitwise, so `NaN` compares equal to `NaN` and `-0.0` does not equal `0.0`. An
    // approximate comparison here would be the tool's job, and the tool is deliberately absent.
    let bits = |v: &Vec<f32>| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let diverges = bits(&f) != bits(&t) || bits(&f) != bits(&g);
    println!(
        "  {}  {label}",
        if diverges { "REPRODUCES" } else { "agrees    " }
    );
    println!("      flex = {f:?}");
    println!("      tch  = {t:?}");
    println!("      wgpu = {g:?}");
    diverges
}

fn main() {
    let (cpu, gpu) = (Default::default(), WgpuDevice::default());
    let tcpu = Default::default();
    let ninf = f32::NEG_INFINITY;
    let mut results = Vec::new();

    println!("\n=== burn-001 / burn#5284 — matmul overflow: inf against NaN ===\n");
    {
        let l = [1e9f32, 0.0, 0.0, -1e9];
        let r = [-1e30f32, 0.0, -0.0, -1e30];
        let f = Tensor::<Flex, 2>::from_floats([l], &cpu).matmul(Tensor::<Flex, 2>::from_floats(
            [[r[0]], [r[1]], [r[2]], [r[3]]],
            &cpu,
        ));
        let t = Tensor::<Torch, 2>::from_floats([l], &tcpu).matmul(
            Tensor::<Torch, 2>::from_floats([[r[0]], [r[1]], [r[2]], [r[3]]], &tcpu),
        );
        let g = Tensor::<Gpu, 2>::from_floats([l], &gpu).matmul(Tensor::<Gpu, 2>::from_floats(
            [[r[0]], [r[1]], [r[2]], [r[3]]],
            &gpu,
        ));
        results.push((
            "burn-001 matmul overflow",
            row(
                "[1,4] x [4,1], products overflow f32",
                f.to_data(),
                t.to_data(),
                g.to_data(),
            ),
        ));
    }

    println!("\n=== burn-002 — max/min return the sentinel instead of an infinity ===\n");
    {
        let f = Tensor::<Flex, 1>::from_floats([ninf, ninf], &cpu).max_dim(0);
        let t = Tensor::<Torch, 1>::from_floats([ninf, ninf], &tcpu).max_dim(0);
        let g = Tensor::<Gpu, 1>::from_floats([ninf, ninf], &gpu).max_dim(0);
        results.push((
            "burn-002 max sentinel",
            row("max([-inf, -inf])", f.to_data(), t.to_data(), g.to_data()),
        ));

        let f = Tensor::<Flex, 1>::from_floats([f32::INFINITY, f32::INFINITY], &cpu).min_dim(0);
        let t = Tensor::<Torch, 1>::from_floats([f32::INFINITY, f32::INFINITY], &tcpu).min_dim(0);
        let g = Tensor::<Gpu, 1>::from_floats([f32::INFINITY, f32::INFINITY], &gpu).min_dim(0);
        results.push((
            "burn-002 min sentinel",
            row("min([+inf, +inf])", f.to_data(), t.to_data(), g.to_data()),
        ));
    }

    println!("\n=== burn-003 — conv2d padding and a non-finite weight ===\n");
    {
        use burn::tensor::module::conv2d;
        use burn::tensor::ops::ConvOptions;
        let opts = || ConvOptions::<2>::new([1, 1], [1, 1], [1, 1], 1);

        // One input element, one weight element, padding 1 => a 3x3 output whose eight
        // outer positions see only padded zeros.
        let f = conv2d(
            Tensor::<Flex, 4>::from_floats([[[[9.910715f32]]]], &cpu),
            Tensor::<Flex, 4>::from_floats([[[[ninf]]]], &cpu),
            None,
            opts(),
        );
        let t = conv2d(
            Tensor::<Torch, 4>::from_floats([[[[9.910715f32]]]], &tcpu),
            Tensor::<Torch, 4>::from_floats([[[[ninf]]]], &tcpu),
            None,
            opts(),
        );
        let g = conv2d(
            Tensor::<Gpu, 4>::from_floats([[[[9.910715f32]]]], &gpu),
            Tensor::<Gpu, 4>::from_floats([[[[ninf]]]], &gpu),
            None,
            opts(),
        );
        results.push((
            "burn-003 conv2d padding",
            row(
                "conv2d(x=[9.91], w=[-inf], padding=1)",
                f.to_data(),
                t.to_data(),
                g.to_data(),
            ),
        ));

        // The control: an identical case with a finite weight must agree everywhere, or the
        // finding is about padding in general rather than about non-finite weights.
        let f = conv2d(
            Tensor::<Flex, 4>::from_floats([[[[9.910715f32]]]], &cpu),
            Tensor::<Flex, 4>::from_floats([[[[2.0f32]]]], &cpu),
            None,
            opts(),
        );
        let t = conv2d(
            Tensor::<Torch, 4>::from_floats([[[[9.910715f32]]]], &tcpu),
            Tensor::<Torch, 4>::from_floats([[[[2.0f32]]]], &tcpu),
            None,
            opts(),
        );
        let g = conv2d(
            Tensor::<Gpu, 4>::from_floats([[[[9.910715f32]]]], &gpu),
            Tensor::<Gpu, 4>::from_floats([[[[2.0f32]]]], &gpu),
            None,
            opts(),
        );
        results.push((
            "burn-003 control (finite weight)",
            row(
                "conv2d(x=[9.91], w=[2.0], padding=1) — must agree",
                f.to_data(),
                t.to_data(),
                g.to_data(),
            ),
        ));
    }

    println!("\n=== summary ===\n");
    for (name, diverges) in &results {
        println!(
            "  {:<34} {}",
            name,
            if *diverges { "reproduces" } else { "agrees" }
        );
    }
    println!();
}
