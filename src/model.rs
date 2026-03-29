use std::collections::BTreeMap;

use burn::nn::{
    Relu,
    loss::{CrossEntropyLoss, CrossEntropyLossConfig},
    modules::{
        Linear, LinearConfig,
        conv::{Conv2d, Conv2dConfig},
    },
};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep};

use crate::{data::MnistBatch, nn_debug::Introspect};

#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
    hidden_size: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            // 1 channel in, 1 channel out (feature map)
            conv1: Conv2dConfig::new([1, 1], [3, 3]).init(device),
            // 1 channel, 26Wx26H
            linear1: LinearConfig::new(1 * 26 * 26, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            activation: Relu::new(),
        }
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
}

impl<B: Backend> Introspect for Model<B> {
    fn get_weights(&self) -> BTreeMap<&str, TensorData> {
        let mut weights = BTreeMap::new();

        weights.insert("linear1", self.linear1.weight.to_data());
        weights.insert("linear2", self.linear2.weight.to_data());

        weights
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();

        let x = images.reshape([batch_size, 1, height, width]);

        let x = self.conv1.forward(x); // batch_size, 28, _, _
        let x = self.activation.forward(x);
        let x = x.reshape([batch_size, 1 * 26 * 26]);
        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        self.linear2.forward(x)
    }

    pub fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());
        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep for Model<B> {
    type Input = MnistBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: MnistBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for Model<B> {
    type Input = MnistBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: MnistBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}
