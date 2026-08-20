//! The channel sweep from burn-003, run on all three backends.
//!
//! The draft asserted that `burn-wgpu` "skips" the pad on the strength of one shape at one
//! autotune setting. This widens that to the same sweep `burn-flex` fails, so the report can
//! state exactly what was tested rather than generalising from a single point.
use burn::tensor::backend::Backend;
use burn::tensor::module::conv2d;
use burn::tensor::{Tensor, TensorData, ops::ConvOptions};

fn out0<B: Backend>(channels_in: usize, channels_out: usize) -> f32 {
    let device = B::Device::default();
    let x_data: Vec<f32> = (0..channels_in * 16).map(|i| 1.0 + i as f32).collect();
    let x: Tensor<B, 4> =
        Tensor::from_data(TensorData::new(x_data, [1, channels_in, 4, 4]), &device);
    let mut w_data = vec![1.0f32; channels_out * channels_in * 9];
    w_data[0] = f32::NEG_INFINITY;
    let w: Tensor<B, 4> = Tensor::from_data(
        TensorData::new(w_data, [channels_out, channels_in, 3, 3]),
        &device,
    );
    conv2d(x, w, None, ConvOptions::<2>::new([1, 1], [1, 1], [1, 1], 1))
        .to_data()
        .to_vec::<f32>()
        .unwrap()[0]
}

fn sweep<B: Backend>(name: &str) {
    print!("{name:<12} channels_in :");
    for ci in 1..=6 {
        print!(" {}={:?}", ci, out0::<B>(ci, 1));
    }
    println!();
    print!("{name:<12} channels_out:");
    for co in [15usize, 16, 17, 18] {
        print!(" {}={:?}", co, out0::<B>(2, co));
    }
    println!();
}

fn main() {
    let autotune = std::env::var("AUTOTUNE_LABEL").unwrap_or_default() == "on";
    println!(
        "\nburn 0.21.0 | wgpu autotune: {}\n",
        if autotune { "ON" } else { "OFF" }
    );
    sweep::<burn::backend::Flex>("burn-flex");
    sweep::<burn::backend::LibTorch<f32>>("burn-tch");
    sweep::<burn::backend::Wgpu>("burn-wgpu");
    println!();
}
