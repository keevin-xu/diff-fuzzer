//! Which of burn-flex's three padded paths is which, measured rather than inferred.
//!
//! The report claims the three reachable paths under padding disagree. The channel sweep
//! only shows two *behaviours*, so this pins the third: a depthwise convolution
//! (groups == channels_in == channels_out, one channel per group) takes the depthwise path,
//! and a large-channel depthwise case stays on it regardless of the small-channel thresholds.
use burn::backend::Flex;
use burn::tensor::module::conv2d;
use burn::tensor::{Tensor, TensorData, ops::ConvOptions};

type B = Flex;

fn run(channels: usize, groups: usize, channels_out: usize) -> f32 {
    let device = Default::default();
    let per_group = channels / groups;
    let x_data: Vec<f32> = (0..channels * 16).map(|i| 1.0 + i as f32).collect();
    let x: Tensor<B, 4> = Tensor::from_data(TensorData::new(x_data, [1, channels, 4, 4]), &device);
    let mut w_data = vec![1.0f32; channels_out * per_group * 9];
    w_data[0] = f32::NEG_INFINITY;
    let w: Tensor<B, 4> = Tensor::from_data(
        TensorData::new(w_data, [channels_out, per_group, 3, 3]),
        &device,
    );
    conv2d(
        x,
        w,
        None,
        ConvOptions::<2>::new([1, 1], [1, 1], [1, 1], groups),
    )
    .to_data()
    .to_vec::<f32>()
    .unwrap()[0]
}

fn main() {
    println!("\nburn-flex, 3x3 conv, padding 1, one -inf weight. out[0] reads only pad.\n");
    println!("  path          shape                                   out[0]");
    // Depthwise: groups == channels_in == channels_out, per_group == 1.
    for c in [2usize, 8, 32, 64] {
        println!(
            "  depthwise     c_in={c:<3} c_out={c:<3} groups={c:<3}          {:?}",
            run(c, c, c)
        );
    }
    // Small-channel: groups == 1, c_in <= 4, c_out <= 16.
    for (ci, co) in [(2usize, 2usize), (4, 16)] {
        println!(
            "  small-channel c_in={ci:<3} c_out={co:<3} groups=1            {:?}",
            run(ci, 1, co)
        );
    }
    // Generic: groups == 1, past either threshold.
    for (ci, co) in [(5usize, 1usize), (2, 17)] {
        println!(
            "  generic       c_in={ci:<3} c_out={co:<3} groups=1            {:?}",
            run(ci, 1, co)
        );
    }
    println!();
}
