
use burn::prelude::*;
use burn::nn::{Linear, LinearConfig, Sigmoid};
use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction::Mean};
use burn::tensor::backend::AutodiffBackend;
use burn::train::{RegressionOutput, TrainOutput, TrainStep, ValidStep};
use crate::dataset::ToyDataBatch;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Sigmoid,
}

#[derive(Config)]
pub struct ModelConfig {
    #[config(default=10)]
    pub hidden_size: usize,
}

impl ModelConfig {
    pub fn init<B :Backend>(&self, device: &B::Device) -> Model<B> {
        let linear1 = LinearConfig::new(1, self.hidden_size).with_bias(true).init(device);
        let linear2 = LinearConfig::new(self.hidden_size, 1).with_bias(true).init(device);

        Model {
            linear1,
            linear2,
            activation: Sigmoid::new(),
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        let x = self.linear2.forward(x);
        x
    }

    pub fn forward_step(&self, item: ToyDataBatch<B>) -> RegressionOutput<B> {
        let targets : Tensor<B, 2>  = item.targets.unsqueeze_dim(1);
        let output: Tensor<B, 2>  = self.forward(item.inputs);
        let loss = MseLoss::new().forward(output.clone(), targets.clone(), Mean);

        RegressionOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<ToyDataBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, item: ToyDataBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_step(item);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<ToyDataBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, item: ToyDataBatch<B>) -> RegressionOutput<B> {
        self.forward_step(item)
    }
}

