use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::{Dataset},
    },
    prelude::*,
};

pub const NUM_FEATURES: usize = 1;

#[derive(Clone, Debug)]
pub struct ToyDataItem {
    pub x: f32,
    pub y: f32,
}

pub struct ToyDataset {
    pub data: Vec<ToyDataItem>,
}

impl Dataset<ToyDataItem> for ToyDataset {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn get(&self, idx: usize) -> std::option::Option<ToyDataItem> {
        Some(self.data[idx].clone())
    }
}

impl ToyDataset {
    pub fn train(x : Vec<f32>, y: Vec<f32>) -> Self {
       Self::new(x, y)
    }

    pub fn validation(x : Vec<f32>, y: Vec<f32>) -> Self {
        Self::new(x, y)
    }

    pub fn test(x : Vec<f32>, y: Vec<f32>) -> Self {
        Self::new(x, y)
    }

    pub fn new (x : Vec<f32>, y: Vec<f32>) -> Self {
        let data = x.iter().zip(y.iter()).map(|(x, y)| ToyDataItem { x: *x, y: *y }).collect();
        Self { data }
    }

}

#[derive(Clone, Debug)]
pub struct ToyDataBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct ToyDataBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 1>,
}

impl<B: Backend> ToyDataBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self {
            device: device.clone(),
        }
    }
}

impl<B: Backend> Batcher<ToyDataItem, ToyDataBatch<B>> for ToyDataBatcher<B> {
    fn batch(&self, items:Vec<ToyDataItem>) -> ToyDataBatch<B> {
        let mut inputs: Vec<Tensor<B,2>> = Vec::new();
        for item in items.iter() {
            let input_tensor = Tensor::<B, 1>::from_floats(
                [item.x],
                &self.device,
            );
            inputs.push(input_tensor.unsqueeze());
        }

        let inputs = Tensor::cat(inputs, 0);
        let targets = items
            .iter()
            .map(|item| Tensor::<B,1>::from_floats([item.y], &self.device))
            .collect();
        let targets = Tensor::cat(targets, 0);
        ToyDataBatch { inputs, targets }
    }
}