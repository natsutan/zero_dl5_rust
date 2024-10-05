use std::fs::File;
use std::io::prelude::*;
//use nalgebra::{Matrix2, OMatrix, Vector2, U1};
//use plotly::Plot;

fn read_file_2d(file_name: &str) -> (Vec<f64>, Vec<f64>) {
    let mut file = File::open(file_name).expect("File not found");
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

    (xs, ys)
}

fn main() {
    let (x, y) = read_file_2d("examples/old_faithful.txt");
    println!("{:?}", x);
    println!("{:?}", y);

}