//! Does `burn-wgpu` clamp an infinity in a max/min reduction?
use diff_fuzzer_core::Implementation;
use tensor_adapter::input::ReduceOp;
use tensor_adapter::{TensorOp, TensorValue, flex, libtorch, wgpu};

fn run(shape: &[usize], data: Vec<f32>, kind: ReduceOp, axis: usize) -> Vec<(String, Vec<f32>)> {
    let case = TensorOp::reduce(kind, TensorValue::new(shape.to_vec(), data), axis);
    let backends: Vec<Box<dyn Implementation<In = TensorOp, Out = burn::tensor::TensorData>>> =
        vec![Box::new(flex()), Box::new(libtorch()), Box::new(wgpu())];
    backends
        .iter()
        .filter_map(|b| {
            b.run(&case)
                .ok()
                .and_then(|o| o.to_vec::<f32>().ok())
                .map(|v| (b.name().to_string(), v))
        })
        .collect()
}

fn show(label: &str, shape: &[usize], data: Vec<f32>, kind: ReduceOp, axis: usize) {
    let out = run(shape, data, kind, axis);
    let rendered: Vec<String> = out.iter().map(|(n, v)| format!("{n}={:?}", v)).collect();
    println!("  {label:<40} {}", rendered.join("  "));
}

fn main() {
    let inf = f32::INFINITY;
    println!("A — reduced axis of length 1 (result must equal the input)\n");
    show("max([-inf])", &[1], vec![-inf], ReduceOp::Max, 0);
    show("max([+inf])", &[1], vec![inf], ReduceOp::Max, 0);
    show("min([-inf])", &[1], vec![-inf], ReduceOp::Min, 0);
    show("min([+inf])", &[1], vec![inf], ReduceOp::Min, 0);

    println!("\nB — reduced axis longer than 1\n");
    show("max([-inf, -5])", &[2], vec![-inf, -5.0], ReduceOp::Max, 0);
    show("max([+inf, 5])", &[2], vec![inf, 5.0], ReduceOp::Max, 0);
    show(
        "max([-inf, -inf])",
        &[2],
        vec![-inf, -inf],
        ReduceOp::Max,
        0,
    );
    show("min([+inf, +inf])", &[2], vec![inf, inf], ReduceOp::Min, 0);
    show("min([-inf, 5])", &[2], vec![-inf, 5.0], ReduceOp::Min, 0);

    println!("\nD — narrowing: rank and reduced-axis length\n");
    show("rank1 [1]   axis0 len1", &[1], vec![-inf], ReduceOp::Max, 0);
    show(
        "rank2 [1,1] axis0 len1",
        &[1, 1],
        vec![-inf],
        ReduceOp::Max,
        0,
    );
    show(
        "rank2 [1,4] axis0 len1",
        &[1, 4],
        vec![0.0, -1e30, -inf, -3.0],
        ReduceOp::Max,
        0,
    );
    show(
        "rank2 [4,1] axis1 len1",
        &[4, 1],
        vec![0.0, -1e30, -inf, -3.0],
        ReduceOp::Max,
        0,
    );
    show(
        "rank2 [2,2] axis0 len2",
        &[2, 2],
        vec![-inf, 1.0, -inf, 2.0],
        ReduceOp::Max,
        0,
    );
    show(
        "rank3 [1,1,2] axis0 len1",
        &[1, 1, 2],
        vec![-inf, 3.0],
        ReduceOp::Max,
        0,
    );

    println!("\nC — does sum behave the same way?\n");
    show("sum([-inf])", &[1], vec![-inf], ReduceOp::Sum, 0);
    show("sum([+inf, 1])", &[2], vec![inf, 1.0], ReduceOp::Sum, 0);
    show("mean([+inf, 1])", &[2], vec![inf, 1.0], ReduceOp::Mean, 0);
}
