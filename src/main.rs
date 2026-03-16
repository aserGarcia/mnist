use burn::backend::{Autodiff, NdArray, ndarray::NdArrayDevice};
use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::train::{
    Learner, SupervisedTraining,
    metric::{AccuracyMetric, LossMetric},
};

use mnist::data::{MnistBatcher, MnistDataset};
use mnist::model::ModelConfig;

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn main() {
    type TrainAutodiffBackend = Autodiff<NdArray>;
    // load the dataset
    let batch_size = 4;
    let num_workers = 4;

    let device = NdArrayDevice::default();

    let config = TrainingConfig::new(ModelConfig::new(10, 256), AdamConfig::new());

    let batcher = MnistBatcher::<TrainAutodiffBackend>::new(device);
    let test_batcher = MnistBatcher::<NdArray>::new(device);
    let dataloader_train = DataLoaderBuilder::new(batcher)
        .batch_size(batch_size)
        .shuffle(config.seed)
        .num_workers(num_workers)
        .build(MnistDataset::train());

    let dataloader_test = DataLoaderBuilder::new(test_batcher)
        .batch_size(batch_size)
        .shuffle(config.seed)
        .num_workers(num_workers)
        .build(MnistDataset::test());

    let training = SupervisedTraining::new("checkpoints", dataloader_train, dataloader_test)
        .metrics((AccuracyMetric::new(), LossMetric::new()))
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary();

    let model = config.model.init::<TrainAutodiffBackend>(&device);
    let result = training.launch(Learner::new(
        model,
        config.optimizer.init(),
        config.learning_rate,
    ));

    result.model.save_file("models", &CompactRecorder::new());
}
