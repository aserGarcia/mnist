use burn::backend::{NdArray, ndarray::NdArrayDevice};
use burn::data::dataloader::DataLoaderBuilder;
use mnist::data::{MnistBatcher, MnistDataset};

fn main() {
    // load the dataset
    let batch_size = 4;
    let num_workers = 4;

    let device = NdArrayDevice::default();

    let batcher = MnistBatcher::<NdArray>::new(device);
    let dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(batch_size)
        .num_workers(num_workers)
        .build(MnistDataset::train());

    for (it, batch) in dataloader.iter().enumerate() {
        println!(
            "[Iteration {}] Images {:?} | Targets {:?}",
            it,
            batch.images.dims(),
            batch.targets.dims()
        );
    }
}
