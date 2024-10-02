use std::f64::consts::PI;
use std::fs::File;
use std::io::prelude::*;
use nalgebra::{Matrix2, OMatrix, Vector2, U1};
use plotly::Plot;
/// https://github.com/oreilly-japan/deep-learning-from-scratch-5/blob/main/step03/mle.py

fn read_file_2d(file_name: &str) -> Vec<(f64, f64)> {
    let mut file = File::open(file_name).expect("File not found");
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("Error reading file");
    // 各行x yの形状のデータを読み込む
    let mut data: Vec<(f64, f64)> = vec![];

    for line in contents.lines() {
        let mut iter = line.split_whitespace();
        let x: f64 = iter.next().unwrap().parse().unwrap();
        let y: f64 = iter.next().unwrap().parse().unwrap();
        data.push((x, y));
    }

    data
}

fn multivariate_normal_2d(x: Vector2<f64>, mu: Vector2<f64>, cov: Matrix2<f64>) -> OMatrix<f64, U1, U1> {
    let det = cov.determinant();
    let inv = cov.try_inverse().unwrap();
    let d = x.shape().0 as f64;
    let z = 1.0 / ((2.0* PI).powf(d) * det ).sqrt();
    let y = z * (-0.5 * (x - mu).transpose() * inv * (x - mu)).exp();
    y
}

fn main() {
    let xs = read_file_2d("examples/height_weight.txt");

    //先頭の500個を取り出し、散布図にして表示する。
    let x: Vec<f64> = xs.iter().map(|(x, _)| *x).collect();
    let y: Vec<f64> = xs.iter().map(|(_, y)| *y).collect();

    let x_small:Vec<f64> = x.iter().take(500).cloned().collect();
    let y_small:Vec<f64> = y.iter().take(500).cloned().collect();


    //全データを使って2次元正規分布の最尤推定を行う
    let n = xs.len() as f64;
    let mu_x = x.iter().sum::<f64>() / n;
    let mu_y = y.iter().sum::<f64>() / n;
    let mu = (mu_x, mu_y);

    let cov = xs.iter().fold([[0.0, 0.0], [0.0, 0.0]], |cov, (x, y)| {
        let dx = x - mu_x;
        let dy = y - mu_y;
        [[cov[0][0] + dx * dx, cov[0][1] + dx * dy], [cov[1][0] + dy * dx, cov[1][1] + dy * dy]]
    });

    let cov = [[cov[0][0] / n, cov[0][1] / n], [cov[1][0] / n, cov[1][1] / n]];
    println!("mu: {:?}", mu);
    println!("cov: {:?}", cov);

    //muとcovを使って2次元正規分布の等高線をplotする
    let n: usize = 80;
    let x_max = 200.0;
    let x_min = 140.0;
    let y_max = 80.0;
    let y_min = 40.0;
    let step_x = (x_max - x_min) / (n - 1) as f64;
    let step_y = (y_max - y_min) / (n - 1) as f64;

    let x: Vec<f64> = (0..n).map(|i| x_min + step_x * i as f64).collect();
    let y: Vec<f64> = (0..n).map(|i| y_min + step_y * i as f64).collect();

    let mu = Vector2::new(mu.0, mu.1);
    let cov = Matrix2::new(cov[0][0], cov[0][1], cov[1][0], cov[1][1]);


    let mut z: Vec<Vec<f64>> = vec![];

    for y in y.iter() {
        let mut row = vec![];
        for x in x.iter() {
            let pdf = multivariate_normal_2d(Vector2::new(*x, *y), mu, cov);
            row.push(pdf[0]);
        }
        z.push(row);
    }

    //上の散布図と等高線を重ねて表示する
    let trace = plotly::Scatter::new(x_small.clone(), y_small.clone()).mode(plotly::common::Mode::Markers);
    let mut plot = Plot::new();
    plot.add_trace(trace);

    let trace = plotly::Contour::new(x, y, z);
    plot.add_trace(trace);

    plot.show();


}