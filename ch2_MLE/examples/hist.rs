use std::fs::File;
use std::io::prelude::*;

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
    println!("{}", data.len());

    //plotryでヒストグラムをplotする
    let trace = plotly::Histogram::new(data).name("height").opacity(0.75);
    let mut plot = plotly::Plot::new();
    plot.add_trace(trace);
    plot.show();
}