
use std::f64::consts::PI;
use std::f64::consts::E;
use plotters::prelude::*;

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
    let mu = 0.0;

    let step = (x_max - x_min) / n as f64;
    for i in 0..n {
        let x = x_min + step * i as f64;
        let y0 = normal(x, mu, 0.5);
        let y1 = normal(x, mu, 1.0);
        let y2 = normal(x, mu, 2.0);
        xs.push(x);
        ys0.push(y0);
        ys1.push(y1);
        ys2.push(y2);
    }

    let line0 = LineSeries::new(xs.iter().zip(ys0.iter()).map(|(x, y )| (*x as f32, *y as f32)), &BLUE);
    let line1 = LineSeries::new(xs.iter().zip(ys1.iter()).map(|(x, y )| (*x as f32, *y as f32)), &RED);
    let line2 = LineSeries::new(xs.iter().zip(ys2.iter()).map(|(x, y )| (*x as f32, *y as f32)), &GREEN);

    let root = BitMapBackend::new("plot.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE);

    let mut chart = ChartBuilder::on(&root)
        .caption("normal distribution", ("sans-serif", 35).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_min as f32 ..x_max as f32, 0.0f32..1.0f32 ).unwrap();

    chart.configure_mesh().draw();

    chart.draw_series(line0).unwrap().label("σ=0.5").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
    chart.draw_series(line1).unwrap().label("σ=1.0").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
    chart.draw_series(line2).unwrap().label("σ=2.0").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw().unwrap();

    let _ = root.present();

}