use crate::data::MnistBatch;
use rand::Rng;

use burn::{
    nn::{Linear, LinearConfig, Relu, Sigmoid},
    prelude::*,
    tensor::backend::AutodiffBackend,

};
use burn::tensor::Distribution;
use burn::train::{RegressionOutput, TrainOutput, TrainStep, ValidStep};

const INPUT_DIM: usize = 784;
const HIDDEN_DIM: usize = 200;
const LATENT_DIM: usize = 20;

#[derive(Module, Debug)]
pub struct Vae<B: Backend> {
    encoder: Encoder<B>,
    decoder: Decoder<B>,
}

#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    linear: Linear<B>,
    activation: Relu,
    linear_mu: Linear<B>,
    linear_logvar: Linear<B>
}


pub struct VaeLossOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
}

#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Sigmoid,
}

impl<B: Backend> Vae<B> {
    pub fn new(device: &B::Device) -> Self {
        let encoder = Encoder::new(device);
        let decoder = Decoder::new(device);
        Self { encoder,  decoder }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let (mu, sigma) = self.encoder.forward(input);
        let z = reparameterize(mu, sigma);
        self.decoder.forward(z)
    }

    pub fn forward_vae(&self, item: MnistBatch<B>) -> VaeLossOutput<B> {
        let (mu, sigma) = self.encoder.forward(item.images.clone());
        let z = reparameterize(mu, sigma);
        let x_hat = self.decoder.forward(z);
        let batch_size = item.images.shape().dims[0];
        let l1 = (item.images - x_hat).powf(Tensor::from([2.0])).sum();
        // L2 = - sum(1 + log(sigma^2) - mu^2 - sigma^2)
        let l2 = (Tensor::from([1.0])
            + sigma.powf(Tensor::from([2.0])).log()
            - mu.powf(Tensor::from([2.0]))
            - sigma.powf(Tensor::from([2.0])))
            .sum();

        let loss = (l1 - l2) / batch_size as f32;

        VaeLossOutput { loss }

    }

}

impl <B: Backend> Encoder<B> {
    pub fn new(device: &B::Device) -> Self {
        let linear = LinearConfig::new(INPUT_DIM, HIDDEN_DIM).with_bias(true).init(device);
        let activation = Relu::new();
        let linear_mu = LinearConfig::new(HIDDEN_DIM, LATENT_DIM).with_bias(true).init(device);
        let linear_logvar = LinearConfig::new(HIDDEN_DIM, LATENT_DIM).with_bias(true).init(device);

        Self { linear, activation, linear_mu, linear_logvar }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        //let [batch_size, _] = x.size();

        let x = self.linear.forward(x);
        let x = self.activation.forward(x);
        let mu = self.linear_mu.forward(x.clone());
        let logvar = self.linear_logvar.forward(x) / 2.0;
        let sigma = logvar.exp();
        (mu, sigma)
    }

}


impl <B: Backend> Decoder<B> {
    pub fn new(device: &B::Device) -> Self {
        let linear1 = LinearConfig::new(LATENT_DIM, HIDDEN_DIM).with_bias(true).init(device);
        let linear2 = LinearConfig::new(HIDDEN_DIM, INPUT_DIM).with_bias(true).init(device);
        let activation = Sigmoid::new();
        Self { linear1, linear2, activation }
    }

    pub fn forward(&self, z: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(z);
        let x = self.linear2.forward(x);
        let x_hat = self.activation.forward(x);
        x_hat
    }

}

fn reparameterize<B: Backend>(mu: Tensor<B, 2>, sigma: Tensor<B, 2>) -> Tensor<B, 2> {
    let [batch_size, latent_dim] = mu.shape().dims[..];
    let device = mu.device();
    let eps = Tensor::<B, 2>::random([batch_size, latent_dim], Distribution::Normal(0.0, 1.0),  &device);
    mu + sigma * eps
}

impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, VaeLossOutput<B>> for Vae<B> {
    fn step(&self, item: MnistBatch<B>) -> TrainOutput<VaeLossOutput<B>> {
        let loss = self.forward_vae(item);
        TrainOutput::new(self, loss.loss.backward(), loss)

    }
}

impl<B: Backend> ValidStep<MnistBatch<B>, VaeLossOutput<B>> for Vae<B> {
    fn step(&self, item: MnistBatch<B>) -> VaeLossOutput<B> {
        self.forward_vae(item)
    }
}
