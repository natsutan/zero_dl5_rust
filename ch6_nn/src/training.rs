use crate::dataset::{ToyDataBatcher, ToyDataset};
use crate::model::{ModelConfig};
use burn::prelude::*;

use burn::optim::AdamConfig;
use burn::{
    data::{dataloader::DataLoaderBuilder},
    record::{CompactRecorder, NoStdTrainingRecorder},
    tensor::backend::AutodiffBackend,
    train::{metric::LossMetric, LearnerBuilder},
};


#[derive(Config)]
pub struct ExpConfig {
    #[config(default = 1000)]
    pub num_epochs: usize,
    #[config(default = 2)]
    pub num_workers: usize,
    #[config(default = 1337)]
    pub seed: u64,
    pub optimizer: AdamConfig,
    #[config(default = 25)]
    pub batch_size: usize,
}

fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn run<B: AutodiffBackend>(artifact_dir: &str, device: B::Device, x : Vec<f32>, y: Vec<f32>) {
    create_artifact_dir(artifact_dir);

    let optimizer = AdamConfig::new();
    let config = ExpConfig::new(optimizer);
    let model = ModelConfig::new().init(&device);
    B::seed(config.seed);

    let train_dataset = ToyDataset::train(x.clone(), y.clone());
    let validation_dataset = ToyDataset::validation(x, y);

    let batcher_train = ToyDataBatcher::<B>::new(device.clone());
    let batcher_test = ToyDataBatcher::<B::InnerBackend>::new(device.clone());

    let data_loader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let data_loader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(validation_dataset);


    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(model, config.optimizer.init(), 0.2);

    let model_trained = learner.fit(data_loader_train, data_loader_test);

    config.save(format!("{artifact_dir}/config.json").as_str()).unwrap();

    model_trained.save_file(format!("{artifact_dir}/model"), &NoStdTrainingRecorder::new(),)
        .expect("Failed to save model");

}