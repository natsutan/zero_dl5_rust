use std::io::prelude::*;
use nalgebra as na;
use nalgebra::{Matrix2, Vector2};
use plotly::common::Mode;
use plotly::{Plot, Scatter};
use rand::Rng;
use rand_distr::{Normal, Distribution};

fn read_file() ->na::OMatrix<f64,  na::Dyn, na::Dyn> {
    let file_name = "examples/old_faithful.txt";
    let mut file = std::fs::File::open(file_name).expect("File not found");
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("Error reading file");
    // 各行x yの形状のデータを読み込む
    let mut xs: Vec<f64> = vec![];
    let mut ys: Vec<f64> = vec![];

    for line in contents.lines() {
        let mut iter = line.split_whitespace();
        let x: f64 = iter.next().unwrap().parse().unwrap();
        let y: f64 = iter.next().unwrap().parse().unwrap();
        xs.push(x);
        ys.push(y);

    }
    let n = xs.len();
    let mut data = na::DMatrix::zeros(n, 2);
    for i in 0..n {
        data[(i, 0)] = xs[i];
        data[(i, 1)] = ys[i];
    }

    data
}


fn sample(mus: &Matrix2<f64>, covs: &Vec<Matrix2<f64>>, phis: &Vector2<f64>) -> Vector2<f64> {
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

    let phis : Vector2<f64> = Vector2::new(0.35, 0.65);
    let mus = Matrix2::new(2.0, 54.50, 4.3, 80.0);
    let covs = vec![
        Matrix2::new(0.07, 0.44, 0.44, 33.7),
        Matrix2::new(0.17, 0.94, 0.94, 36.0),
    ];

    let original_data =  read_file();
    let n = 500;

    // gmmからNこサンプリングし、new_dataに格納
    let mut new_data: Vec<Vector2<f64>> = vec![];
    for _ in 0..n {
        let x = sample(&mus, &covs, &phis);
        new_data.push(x);
    }

    // plotlyで両方のデータを描画
    let x: Vec<f64> = original_data.column(0).iter().map(|x| *x).collect();
    let y: Vec<f64> = original_data.column(1).iter().map(|x| *x).collect();
    let trace0 = Scatter::new(x.clone(), y.clone()).mode(Mode::Markers).name("original");

    //new data
    let x: Vec<f64> = new_data.iter().map(|x| x[0]).collect();
    let y: Vec<f64> = new_data.iter().map(|x| x[1]).collect();
    let trace1 = Scatter::new(x.clone(), y.clone()).mode(Mode::Markers).name("sampled");

    let mut plot = Plot::new();
    plot.add_trace(trace0);
    plot.add_trace(trace1);
    plot.show();


}