use std::f64::consts::PI;
use nalgebra::{Matrix2, OMatrix, Vector2, U1};
use plotly::Surface;
use plotly::Plot;

fn multivariate_normal_2d(x: Vector2<f64>, mu: Vector2<f64>, cov: Matrix2<f64>) -> OMatrix<f64, U1, U1> {
    let det = cov.determinant();
    let inv = cov.try_inverse().unwrap();
    let d = x.shape().0 as f64;
    let z = 1.0 / ((2.0* PI).powf(d) * det ).sqrt();
    let y = z * (-0.5 * (x - mu).transpose() * inv * (x - mu)).exp();
    y
}
fn main() {
    let mu = Vector2::new(0.5, -0.2);
    let cov = Matrix2::new(2.0, 0.3, 0.3, 0.5);

    let n: usize = 80;
    // -2から2までの等間隔なn個の数列を生成
    let max = 4.0;
    let min = -4.0;
    let step = (max - min) / (n - 1) as f64;
    let x: Vec<f64> = (0..n).map(|i| min + step * i as f64).collect();
    let y: Vec<f64> = (0..n).map(|i| min + step * i as f64).collect();

    let z: Vec<Vec<f64>> = x.iter().map(|&xi| y.iter().map(|&yi| {
        let x = Vector2::new(xi, yi);
        let y = multivariate_normal_2d(x, mu, cov);
        y[0]
    }).collect()).collect();

    let trace = Surface::new(z.clone()).x(x.clone()).y(y.clone());
    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.show();

    //plotlyでcontour plotをplotする。
    let trace = plotly::contour::Contour::new(x, y, z);
    let mut plot = plotly::Plot::new();
    plot.add_trace(trace);

    plot.show();
}