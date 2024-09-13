
use std::f64::consts::PI;
use std::f64::consts::E;
use plotters::prelude::*;

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

    let line = LineSeries::new(xs.iter().zip(ys.iter()).map(|(x, y )| (*x as f32, *y as f32)), &RED);

    let root = BitMapBackend::new("plot.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE);

    let mut chart = ChartBuilder::on(&root)
        .caption("normal distribution", ("sans-serif", 35).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_min as f32 ..x_max as f32, 0.0f32..1.0f32 ).unwrap();

    chart.configure_mesh().draw();

    chart
        .draw_series(line).unwrap()
        .label("y = normal(x, 0.0, 1.0)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw().unwrap();

    let _ = root.present();

}