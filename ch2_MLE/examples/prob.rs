use std::fs::File;
use std::io::prelude::*;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;
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

fn main() {
    let data = read_file("examples/height.txt");

    let mu = data.iter().sum::<f64>() / data.len() as f64;
    let sigma = (data.iter().map(|x| (x - mu) * (x - mu)).sum::<f64>() / data.len() as f64).sqrt();

    //mu, sigmaを使って正規分布のcdfを求める
    let normal = Normal::new(mu, sigma).unwrap();
    let cdf = normal.cdf(160.0);
    println!("p(x <= 160) = {}", cdf);

    let cdf = 1.0 - normal.cdf(180.0);
    println!("p(x > 180) = {}", cdf);

}