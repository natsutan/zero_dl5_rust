use burn::data::dataloader::batcher::Batcher;
use crate::{
    dataset::{ToyDataBatcher, ToyDataset, ToyDataItem},
    model::{ModelConfig},
};

use burn::prelude::*;
use burn::record::{NoStdTrainingRecorder, Recorder};


pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, x: Vec<f32>, y: Vec<f32>) -> Vec<f32> {
    let record = NoStdTrainingRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Failed to load model record");

    let model = ModelConfig::new()
        .init(&device)
        .load_record(record);

    let dataset = ToyDataset::test(x.clone(), y.clone());
    let items: Vec<ToyDataItem> = dataset.data;

    let batcher = ToyDataBatcher::<B>::new(device.clone());
    let batch = batcher.batch(items);
    let predicted = model.forward(batch.inputs);
    let _targets = batch.targets;

    let predicted = predicted.squeeze::<1>(1).into_data();
    //let expected = targets.into_data();

    //Vec<f32>に変換して返す
    predicted.iter().map(|x:f32| x as f32).collect()
}