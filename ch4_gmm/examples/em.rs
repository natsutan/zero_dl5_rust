use std::io::prelude::*;
use nalgebra as na;

fn read_file()->na::OMatrix<f64,  na::Dyn, na::Dyn> {
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

fn multivariate_normal(x: &na::DVector<f64>, mu: &na::DVector<f64>, cov: &na::DMatrix<f64>) -> f64 {
    let d = x.len();
    let det = cov.determinant();
    let inv = cov.clone().try_inverse().unwrap();
    let a = (x.clone() - mu.clone()).transpose() * inv * (x - mu);
    let b = (-0.5 * a[0]).exp();
    let c = 1.0 / ((2.0 * std::f64::consts::PI).powf(d as f64 / 2.0) * det.sqrt());
    b * c

}

fn gmm(x: &na::DVector<f64>, phis: &na::DVector<f64>, mus: &na::DMatrix<f64>, covs: &Vec<na::DMatrix<f64>>) -> f64 {
    let k = phis.len();
    let mut y = 0.0;
    for i in 0..k {
        let p = phis[i];
        //mはf64のベクトル
        let m : na::DVector<f64> = mus.row(i).transpose();
        let c = covs[i].clone();
        y += p * multivariate_normal(x, &m, &c);
    }
    y
}

fn likelihood(x: &na::OMatrix<f64, na::Dyn, na::Dyn>, phis: &na::DVector<f64>, mus: &na::DMatrix<f64>, covs: &Vec<na::DMatrix<f64>>) -> f64 {
    let eps = 1e-8;
    let n = x.nrows();
    let mut l = 0.0;
    for i in 0..n {
        let xi = x.row(i).transpose().to_owned();
        let y = gmm(&xi, phis, mus, covs);
        l += (y+eps).ln();
    }

    l / n as f64
}

fn main() {
    let data = read_file();

    let mut phis = na::DVector::from_vec(vec![0.5, 0.5]);
    let mut mus = na::DMatrix::from_row_slice(2, 2, &[0.0, 50.0, 0.0, 100.0]);
    let mut covs = vec![
        na::DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]),
        na::DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]),
    ];

    let K = 2;
    let N = data.nrows();
    let MAX_ITER = 100;
    let THRESHOLD = 1e-4;

    let mut current_likelihood = likelihood(&data, &phis, &mus, &covs);
    println!("initial likelihood: {}", current_likelihood);

    for iter in 0..MAX_ITER {
        //E step
        let mut qs = na::DMatrix::zeros(N, K);
        for n in 0..N {
            let x = data.row(n).transpose().into_owned();
            let mut sum = 0.0;
            for k in 0..K {
                let p = phis[k];
                let m = mus.row(k).transpose();
                let c = covs[k].clone();
                let y = p * multivariate_normal(&x, &m, &c);
                qs[(n, k)] = y;
                sum += y;
            }
            qs.row_mut(n).scale_mut(1.0 / sum);
        }
        //M step
        let qs_sum = qs.column_sum();
        for k in 0..K {
            // 1. phis
            phis[k] = qs_sum[k] / N as f64;

            // 2. mus
            let mut c = na::DVector::zeros(data.ncols());
            for n in 0..N {
                let x = data.row(n).transpose().into_owned();
                c += qs[(n, k)] * x;
            }
            mus.row_mut(k).copy_from(&(&c / qs_sum[k]));

            // 3. covs
            let mut c = na::DMatrix::zeros(data.ncols(), data.ncols());
            for n in 0..N {
                let x = data.row(n).transpose().into_owned();
                let d = x - mus.row(k).transpose();
                c += qs[(n, k)] * d.clone() * d.transpose();
            }
            covs[k] = c / qs_sum[k];
        }


        let new_likelihood = likelihood(&data, &phis, &mus, &covs);
        let diff = new_likelihood - current_likelihood;
        println!("iter: {}, likelihood: {}", iter, new_likelihood);
        if diff.abs() < THRESHOLD {
            break;
        }
        current_likelihood = new_likelihood;


    }

    //結果の表示
    println!("phis: {:?}", phis);
    println!("mus: {:?}", mus);
    println!("covs: {:?}", covs);

}

