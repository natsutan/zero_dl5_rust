//https://github.com/tracel-ai/burn/tree/main/examples/simple-regression/src

use plotly::common::Mode;
use burn::backend::{Autodiff, Wgpu};
use burn::backend::wgpu::WgpuDevice;

type Backend = Wgpu;

type AutoDIffBackend = Autodiff<Backend>;
use plotly::Scatter;
use ch6_nn::inference::infer;
use ch6_nn::training;

static ARTIFACT_DIR: &str = "tmp/burn-example-regression";

fn toy_data() -> (Vec<f32>, Vec<f32>) {
    // 0.0～1.0の範囲でランダムなデータを100個生成してListを作成
    //let device = Default::default();
    let x = (0..100).map(|_| rand::random::<f32>()).collect::<Vec<f32>>();

    // ノイズeを0.0から1.0の範囲で100個生成
    let e = (0..100).map(|_| rand::random::<f32>()).collect::<Vec<f32>>();

    // y = sin(2 * pi * x) + e
    let y = x.iter().zip(e.iter()).map(|(x, e)| (2.0 * std::f32::consts::PI * x).sin() + e).collect::<Vec<f32>>();

    (x, y)
}

fn main() {
    let (x, y) = toy_data();

    let device = WgpuDevice::default();

    training::run::<AutoDIffBackend>(ARTIFACT_DIR, device.clone(), x.clone(), y.clone());

    // x_test = 0.0から1.0までの範囲で0.01刻みで100個生成
    let x_test = (0..100).map(|i| i as f32 / 100.0).collect::<Vec<f32>>();
    let y_test = infer::<Backend>(ARTIFACT_DIR, device, x_test.clone(), y.clone());

    // Plot the data
    // rを並べて表示
    let trace = Scatter::new(x.clone(), y.clone()).mode(Mode::Markers);
    let mut plot = plotly::Plot::new();
    plot.add_trace(trace);

    let trace = Scatter::new(x_test.clone(), y_test.clone()).mode(Mode::Markers);
    plot.add_trace(trace);


    plot.show();
}