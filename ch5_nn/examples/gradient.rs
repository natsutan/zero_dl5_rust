use burn::tensor::Tensor;
use burn::backend::{Autodiff, Wgpu};

type Backend = Wgpu;
type AutoDIffBackend = Autodiff<Backend>;


fn rosenbrok(x0: &Tensor<AutoDIffBackend, 1>, x1: &Tensor<AutoDIffBackend, 1>) -> Tensor<AutoDIffBackend, 1> {
    let x0 = x0.clone();
    let x1 = x1.clone();

    let y = (x1 - x0.clone().powf_scalar(2.0)).powf_scalar(2.0) * 100.0 + (x0  - 1.0).powf_scalar(2.0);
    y
}


fn main() {

    let device = Default::default();


    let mut new_x0 = 0.0;
    let mut new_x1 = 2.0;

    let lr = 0.001;
    let iter = 10000;
    for i in 0..iter {
        if i % 1000 == 0 {
           println!("iter: {} x0 = {}, x1 = {}", i, new_x0, new_x1);
        }
        let x0 = Tensor::<AutoDIffBackend, 1>::from_floats([new_x0], &device).require_grad();
        let x1 = Tensor::<AutoDIffBackend, 1>::from_floats([new_x1], &device).require_grad();

        let y = rosenbrok(&x0, &x1);
        let mut gradients = y.backward();

        let tensor_grad0 = x0.grad_remove(&mut gradients);
        let tensor_grad1 = x1.grad_remove(&mut gradients);

        let grad0 = tensor_grad0.unwrap().into_scalar();
        let grad1 = tensor_grad1.unwrap().into_scalar();

        new_x0 = x0.into_scalar() - grad0 * lr;
        new_x1 = x1.into_scalar() - grad1 * lr;

    }
    println!("x0 = {}, x1 = {}", new_x0, new_x1);

}