use nalgebra::{Matrix2, Vector2};
use plotly::{Plot, Scatter};
use plotly::common::Mode;
use rand::Rng;
use rand_distr::{Normal, Distribution};

fn sample(mus: Matrix2<f64>, covs: Vec<Matrix2<f64>>, phis: Vector2<f64>) -> Vector2<f64> {
    let mut rng = rand::thread_rng();
    let z = if rng.gen::<f64>() < phis[0] { 0 } else { 1 };

    let mu = mus.row(z);
    let cov = covs[z];

    //正規分布の乱数を作る
    //https://qiita.com/osanshouo/items/110176cf2e143b9788f5
    let normal = Normal::new(0.0, 1.0).unwrap();
    let x = Vector2::new(normal.sample(&mut rng), normal.sample(&mut rng));


    let l = cov.cholesky().unwrap().l();
    let y :Vector2<f64> = l * x + mu.transpose();
    y

}

fn main() {
    let mus = Matrix2::new(2.0, 54.50, 4.3, 80.0);
    let covs = vec![
        Matrix2::new(0.07, 0.44, 0.44, 33.7),
        Matrix2::new(1.17, 0.94, 0.94, 36.0),
    ];
    let phis = Vector2::new(0.35, 0.65);

    let n = 500;
    let mut xs: Vec<Vector2<f64>> = vec![];
    for _ in 0..n {
        let x = sample(mus, covs.clone(), phis);
        xs.push(x);
    }

    let x: Vec<f64> = xs.iter().map(|x| x[0]).collect();
    let y: Vec<f64> = xs.iter().map(|x| x[1]).collect();

    let trace = Scatter::new(x, y).mode(Mode::Markers);
    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.show();



}