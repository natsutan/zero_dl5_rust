use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*,
};

#[derive(Clone, Debug)]
pub struct MnistBather<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    pub images: Tensor<B, 2>,
}

impl<B: Backend> MnistBather<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl <B: Backend> Batcher<MnistItem, MnistBatch<B>> for MnistBather<B> {
    fn batch(&self, items: Vec<MnistItem>) -> MnistBatch<B> {
        //itemを取り出し28*28の画像を1次元に変換する。
        let images = items
            .iter()
            .map(|item| TensorData::from(item.image))
            .map(|data| Tensor::<B, 2>::from_data(data.convert::<B::FloatElem>(), &self.device))
            .map(|tensor| tensor.unsqueeze_dim(0))
            // normalize: make between [0,1] and make the mean =  0 and std = 1
            // values mean=0.1307,std=0.3081 were copied from Pytorch Mist Example
            // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122
            .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
            .collect();

        let images = Tensor::cat(images, 0);

        MnistBatch { images: images }
    }
}