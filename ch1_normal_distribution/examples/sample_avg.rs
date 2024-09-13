use rand::Rng;
use plotters::prelude::*;
fn main() {
    let n = 10000;
    let sample_size = 10;
    let mut rng = rand::thread_rng();

    // bins_sizeは log2(n) + 1
    let bins_size:usize = ((n as f64).log2().ceil() as usize + 1) * 10;
    let mut bins: Vec<i32> = vec![0; bins_size];

    for _ in 0..n {
        let mut xs :Vec<f64> = vec![];
        for _ in 0..sample_size {
            let x: f64 = rng.gen();
            xs.push(x);
        }
        let avg = xs.iter().sum::<f64>() / sample_size as f64;
        let bin = (avg * bins_size as f64).floor() as usize;
        bins[bin] += 1;
    }

    let max_bin:i32 = *bins.iter().max().unwrap();
//https://plotters-rs.github.io/book/basic/basic_data_plotting.html

    let root_area = BitMapBackend::new("histgram.png", (640, 480))
        .into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("histgram", ("sans-serif", 20))
        .build_cartesian_2d((0..bins_size - 1).into_segmented(), 0..max_bin)
        .unwrap();

    ctx.configure_mesh().draw().unwrap();


    ctx.draw_series((0..).zip(bins.iter()).map(|(x, y)| {
        let x0 = SegmentValue::Exact(x);
        let x1 = SegmentValue::Exact(x + 1);
        let mut bar = Rectangle::new([(x0  , 0), (x1  , *y)], BLUE.filled());
//        bar.set_margin(0, 0, 5, 5);
        bar
    }))
        .unwrap();

    println!("{:?}", bins);
}
