use tch::Tensor;

fn main() {
    println!("Cuda available: {}", tch::Cuda::is_available());
    println!("Cudnn available: {}", tch::Cuda::cudnn_is_available());
    let device = tch::Device::cuda_if_available();

    let x = Tensor::from(5.0);
    let y = 3 * &x * &x;

    x.print();

}