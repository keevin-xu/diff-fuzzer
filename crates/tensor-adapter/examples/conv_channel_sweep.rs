use burn::backend::Flex;
use burn::tensor::module::conv2d;
use burn::tensor::{Tensor, TensorData, ops::ConvOptions};

type B = Flex;

// 3x3 convolution, padding 1, over a 4x4 input, groups = 1.
// One weight element is -inf; every other weight and the whole input are finite.
fn conv_with_one_inf_weight(channels_in: usize, channels_out: usize) -> Vec<f32> {
    let device = Default::default();

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
        .unwrap()
}

fn main() {
    for channels_in in 1..=6 {
        println!(
            "channels_in = {channels_in}  ->  out[0] = {:?}",
            conv_with_one_inf_weight(channels_in, 1)[0]
        );
    }
    for channels_out in [15usize, 16, 17, 18] {
        println!(
            "channels_out = {channels_out} ->  out[0] = {:?}",
            conv_with_one_inf_weight(2, channels_out)[0]
        );
    }
}
