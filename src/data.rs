use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::{
    Dataset, HuggingfaceDatasetLoader, SqliteDataset,
    transform::{Mapper, MapperDataset},
};
use burn::prelude::*;
use serde::{Deserialize, Serialize};

use image::ImageReader;
use std::io::Cursor;

const MNIST_BASE_DIR: &str = "dataset/mnist";
const WIDTH: usize = 28;
const HEIGHT: usize = 28;

#[derive(Deserialize, Debug, Clone)]
struct MnistItemRaw {
    pub image_bytes: Vec<u8>,
    pub label: u8,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct MnistItem {
    pub image: TensorData,
    pub label: u8,
}

struct BytesToImage;

impl Mapper<MnistItemRaw, MnistItem> for BytesToImage {
    fn map(&self, item: &MnistItemRaw) -> MnistItem {
        let img = ImageReader::new(Cursor::new(&item.image_bytes))
            .with_guessed_format()
            .expect("Failed to guess format")
            .decode()
            .expect("Failed to decode image");

        let raw_pixels: Vec<u8> = img.to_luma8().into_raw();

        debug_assert_eq!(raw_pixels.len(), WIDTH * HEIGHT);

        let image_tensor = TensorData::new(raw_pixels.clone(), [WIDTH, HEIGHT]);

        MnistItem {
            image: image_tensor,
            label: item.label,
        }
    }
}

type MappedDataset = MapperDataset<SqliteDataset<MnistItemRaw>, BytesToImage, MnistItemRaw>;

pub struct MnistDataset {
    dataset: MappedDataset,
}

impl MnistDataset {
    pub fn train() -> Self {
        // train split
        Self::new("train")
    }

    pub fn test() -> Self {
        // train split
        Self::new("test")
    }

    pub fn new(split: &str) -> Self {
        let dataset: SqliteDataset<MnistItemRaw> = HuggingfaceDatasetLoader::new("mnist")
            .with_base_dir(MNIST_BASE_DIR)
            .dataset(split)
            .expect("Failed to fetch {split} data");

        let dataset = MapperDataset::new(dataset, BytesToImage);
        Self { dataset }
    }
}

impl Dataset<MnistItem> for MnistDataset {
    fn get(&self, index: usize) -> Option<MnistItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

#[derive(Clone)]
pub struct MnistBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> MnistBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<B, MnistItem, MnistBatch<B>> for MnistBatcher<B> {
    fn batch(&self, items: Vec<MnistItem>, device: &B::Device) -> MnistBatch<B> {
        let images = items
            .iter()
            .map(|item| {
                Tensor::<B, 2, Float>::from_data(item.image.clone(), &self.device)
                    .reshape([1, WIDTH, HEIGHT])
            })
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data(TensorData::from([item.label as i32]), device)
            })
            .collect();

        let images = Tensor::cat(images, 0).to_device(&self.device);
        let targets = Tensor::cat(targets, 0).to_device(&self.device);

        MnistBatch { images, targets }
    }
}
