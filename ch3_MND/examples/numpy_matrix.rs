use nalgebra::{Matrix2, Matrix2x3, OMatrix, Vector2, U1};
use std::f64::consts::PI;

// xは1次元の配列
// muは1次元の配列
// covは2次元の配列
fn multivariate_normal_2d(x: Vector2<f64>, mu: Vector2<f64>, cov: Matrix2<f64>) -> OMatrix<f64, U1, U1> {
    let det = cov.determinant();
    let inv = cov.try_inverse().unwrap();
    let d = x.shape().0 as f64;
    let z = 1.0 / ((2.0* PI).powf(d) * det ).sqrt();
    let y = z * (-0.5 * (x - mu).transpose() * inv * (x - mu)).exp();
    y
}

fn main() {
    let a = Matrix2x3::new(1, 2, 3, 4, 5, 6);
    println!("{}", a);
    println!("{}", a.transpose());

    let a = Matrix2::new(3.0, 4.0, 5.0, 6.0);
    // 行列式を求める
    let d = a.determinant();
    println!("{}", d);

    let x = Vector2::new(0.0, 0.0);
    let mu = Vector2::new(1.0, 2.0);
    let cov = Matrix2::new(1.0, 0.0, 0.0, 1.0);
    let y = multivariate_normal_2d(x, mu, cov);
    println!("{}", y);
}