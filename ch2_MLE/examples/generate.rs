use std::fs::File;
use std::io::prelude::*;
use rand_distr::Normal;
use rand::prelude::{Distribution, thread_rng};

fn read_file(file_name: &str) -> Vec<f64> {
    let mut file = File::open(file_name).expect("File not found");
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("Error reading file");
    let data: Vec<f64> = contents
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    data
}

fn sample_from_normal_distribution(mu: f64, sigma: f64, n: usize) -> Vec<f64> {
    let normal = Normal::new(mu, sigma).unwrap();
    let mut rng = thread_rng();
    let samples: Vec<f64> = (0..n).map(|_| normal.sample(&mut rng)).collect();
    samples
}


fn main() {
    let data = read_file("examples/height.txt");

    let mu = data.iter().sum::<f64>() / data.len() as f64;
    let sigma = (data.iter().map(|x| (x - mu) * (x - mu)).sum::<f64>() / data.len() as f64).sqrt();

    let samples = sample_from_normal_distribution(mu, sigma, data.len());

    //plotryでヒストグラムをplotする
    let trace = plotly::Histogram::new(data).name("height").opacity(0.75);
    let mut plot = plotly::Plot::new();
    plot.add_trace(trace);

    //plotlyでサンプルを色を変えてplotする
    let trace = plotly::Histogram::new(samples).name("sample").opacity(0.75);
    plot.add_trace(trace);
    
    plot.show();
}

