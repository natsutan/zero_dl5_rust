//https://github.com/tracel-ai/burn/tree/main/examples/simple-regression/src

use plotly::common::Mode;
use burn::tensor::Tensor;
use burn::backend::{Autodiff, Wgpu};
use burn::backend::wgpu::WgpuDevice;

type Backend = Wgpu;
type AutoDIffBackend = Autodiff<Backend>;
use plotly::Scatter;

fn toy_data() -> (Vec<f32>, Vec<f32>) {
    // 0.0～1.0の範囲でランダムなデータを100個生成してListを作成
    //let device = Default::default();
    let x = (0..100).map(|_| rand::random::<f32>()).collect::<Vec<f32>>();

    // ノイズeを0.0から1.0の範囲で100個生成
    let e = (0..100).map(|_| rand::random::<f32>()).collect::<Vec<f32>>();

    // y = 2 * x + 5 + e
    let y = x.iter().zip(e.iter()).map(|(x, e)| 2.0 * x + 5.0 + e).collect::<Vec<f32>>();

    (x, y)
}

fn predict(x:&Vec<f32>,  w:&Tensor<AutoDIffBackend, 1>, b:&Tensor<AutoDIffBackend, 1>, device:&WgpuDevice ) -> Tensor<AutoDIffBackend, 1> {
    let x = Tensor::<AutoDIffBackend, 1>::from_floats(x.as_slice(), &device);
    let y = x * w.clone() + b.clone();
    y
}

fn mean_squared_error(y: &Vec<f32>, y_hat: &Tensor<AutoDIffBackend, 1>) -> Tensor<AutoDIffBackend, 1> {
    let y_hat = y_hat.clone();
    let n = y.len() as f32;
    let y = Tensor::<AutoDIffBackend, 1>::from_floats(y.as_slice(), &y_hat.device());
    let loss = (y - y_hat).powf_scalar(2.0).sum() / n as f32;
    loss
}

fn main() {
    // 0.0～1.0の範囲でランダムなデータを100個生成してListを作成
    let device = Default::default();

    let (x, y) = toy_data();

    let lr = 0.1;
    let iter = 100;

    let mut new_w = 0.0;
    let mut new_b = 2.0;

    for i in 0..iter {
        let w = Tensor::<AutoDIffBackend, 1>::from_floats([new_w], &device).require_grad();
        let b = Tensor::<AutoDIffBackend, 1>::from_floats([new_b], &device).require_grad();

        let y_hat = predict(&x, &w, &b, &device);
        let loss = mean_squared_error(&y, &y_hat);

        let mut gradients = loss.backward();

        let tensor_grad_w = w.grad_remove(&mut gradients);
        let tensor_grad_b = b.grad_remove(&mut gradients);

        new_w = w.into_scalar() - tensor_grad_w.unwrap().into_scalar() * lr;
        new_b = b.into_scalar() - tensor_grad_b.unwrap().into_scalar() * lr;

        if i % 10 == 0 {
            println!("iter: {} loss = {}", i, loss.into_scalar());
        }
    }

    println!("w = {}, b = {}", new_w, new_b);

    let trace = Scatter::new(x.clone(), y.clone()).mode(Mode::Markers);
    let mut plot = plotly::Plot::new();
    plot.add_trace(trace);

    let y_hat = x.iter().map(|x| new_w * x + new_b).collect::<Vec<f32>>();

    let trace = Scatter::new(x.clone(), y_hat.clone()).mode(Mode::Lines);
    plot.add_trace(trace);
    plot.show();

}
