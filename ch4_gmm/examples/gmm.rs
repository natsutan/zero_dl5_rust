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

fn gmm(x: Vector2<f64> ,phis: Vec<f64>, mus: Vec<Vector2<f64>>, covs: Vec<Matrix2<f64>>) -> f64 {
    let k = phis.len();
    let mut y = 0.0;
    for i in 0..k {
        y += phis[i] * multivariate_normal_2d(x, mus[i], covs[i])[0];
    }
    y

}

fn main() {

    let mus = vec![Vector2::new(2.0, 54.5), Vector2::new(4.3, 80.0)];
    let covs = vec![
        Matrix2::new(0.07, 0.44, 0.44, 33.7),
        Matrix2::new(1.17, 0.94, 0.94, 36.0),
    ];
    let phis:Vec<f64> = vec![0.35, 0.65];

    //x は0～6まで0.1刻み、yは40～100まで0.1刻み
    let x_min = 0.0;
    let x_max = 6.0;
    let y_min = 40.0;
    let y_max = 100.0;
    let n = 60;
    let step_x = (x_max - x_min) / n as f64;
    let step_y = (y_max - y_min) / n as f64;
    let x: Vec<f64> = (0..n).map(|i| x_min + step_x * i as f64).collect();
    let y: Vec<f64> = (0..n).map(|i| y_min + step_y * i as f64).collect();

    let z: Vec<Vec<f64>> = (0..n).map(|i| {
        let x = x_min + step_x * i as f64;
        (0..n).map(|j| {
            let y = y_min + step_y * j as f64;
            let x = Vector2::new(x, y);
            gmm(x, phis.clone(), mus.clone(), covs.clone())
        }).collect()
    }).collect();

    //plotlyでSurfaceを描く
    let trace = Surface::new(z.clone()).x(x.clone()).y(y.clone());
    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.show();

    //plotlyでcontour plotをplotする。
    let trace = plotly::contour::Contour::new(x, y, z);
    let mut plot = plotly::Plot::new();
    plot.add_trace(trace);

    plot.show();
    println!("gmm");
}
