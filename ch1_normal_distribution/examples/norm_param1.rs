
use std::f64::consts::PI;
use std::f64::consts::E;
use plotly::common::Mode;
use plotly::{Plot, Scatter};


fn normal(x:f64, mu:f64, sigma:f64) -> f64 {
    let a = - (x - mu) * (x - mu) / (2.0 * sigma * sigma);
    E.powf(a) / (sigma * (2.0 * PI).sqrt())

}

fn main() {

    let mut xs :Vec<f64> = vec![];
    let mut ys0 :Vec<f64> = vec![];
    let mut ys1 :Vec<f64> = vec![];
    let mut ys2 :Vec<f64> = vec![];

    let x_min = -10.0;
    let x_max = 10.0;
    let n = 200;
    let sigma = 1.0;

    let step = (x_max - x_min) / n as f64;
    for i in 0..n {
        let x = x_min + step * i as f64;
        let y0 = normal(x, -3.0, sigma);
        let y1 = normal(x, 0.0, sigma);
        let y2 = normal(x, 5.0, sigma);
        xs.push(x);
        ys0.push(y0);
        ys1.push(y1);
        ys2.push(y2);
    }

    //plotlyで三本のグラフを書く
    let trace0 = Scatter::new(xs.clone(), ys0).mode(Mode::Lines).name("μ=-3.0");
    let trace1 = Scatter::new(xs.clone(), ys1).mode(Mode::Lines).name("μ=0.0");
    let trace2 = Scatter::new(xs.clone(), ys2).mode(Mode::Lines).name("μ=5.0");

    let mut plot = Plot::new();
    plot.add_trace(trace0);
    plot.add_trace(trace1);
    plot.add_trace(trace2);
    plot.show();
}