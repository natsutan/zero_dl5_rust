use plotly::Surface;
use plotly::Plot;


fn main() {
    let n: usize = 40;
    // -2から2までの等間隔なn個の数列を生成
    let max = 2.0;
    let min = -2.0;
    let step = (max - min) / (n - 1) as f64;
    let x: Vec<f64> = (0..n).map(|i| min + step * i as f64).collect();
    let y: Vec<f64> = (0..n).map(|i| min + step * i as f64).collect();

    // z = x ** 2 + y ** 2
    let z: Vec<Vec<f64>> = x.iter().map(|&xi| y.iter().map(|&yi| xi.powi(2) + yi.powi(2)).collect()).collect();

    let trace = Surface::new(z.clone()).x(x.clone()).y(y.clone());
    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.show();

    //plotlyでcontour plotをplotする。
    let trace = plotly::contour::Contour::new(x, y, z);
    let mut plot = plotly::Plot::new();
    plot.add_trace(trace);

    plot.show();
}