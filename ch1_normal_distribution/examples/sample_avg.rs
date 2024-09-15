use rand::Rng;

fn main() {
    let n = 10000;
    let sample_size = 5;
    let mut rng = rand::thread_rng();

    let mut avgs: Vec<f64> = vec![];

    for _ in 0..n {
        let mut xs :Vec<f64> = vec![];
        for _ in 0..sample_size {
            let x: f64 = rng.gen();
            xs.push(x);
        }
        let avg = xs.iter().sum::<f64>() / sample_size as f64;
        avgs.push(avg);
    }

    //plotlyでヒストグラムをplotする。
    let trace = plotly::Histogram::new(avgs.clone()).name("sample average").opacity(0.75);
    let mut plot = plotly::Plot::new();
    plot.add_trace(trace);
    plot.show();


}

