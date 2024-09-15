
use std::f64::consts::PI;
use std::f64::consts::E;
use plotly::common::Mode;
use plotly::{Plot, Scatter};

fn normal(x:f64, mu:f64, sigma:f64) -> f64 {
    let a = - (x - mu) * (x - mu) / (2.0 * sigma * sigma);
    E.powf(a) / (sigma * (2.0 * PI).sqrt())

}

fn main() {
    let mu = 0.0;
    let sigma = 1.0;

    let mut xs :Vec<f64> = vec![];
    let mut ys :Vec<f64> = vec![];

    let x_min = -5.0;
    let x_max = 5.0;
    let n = 100;

    let step = (x_max - x_min) / n as f64;
    for i in 0..n {
        let x = x_min + step * i as f64;
        let y = normal(x, mu, sigma);
        xs.push(x);
        ys.push(y);
    }


    let trace = Scatter::new(xs, ys).mode(Mode::Lines).name("normal distribution");

    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.show();

}