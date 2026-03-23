use burn::{Tensor, prelude::Backend, tensor::TensorData};

use ndarray::Array2;
//use ndarray_linalg::{Eigh, UPLO};

use std::collections::BTreeMap;

pub trait Introspect {
    fn get_weights(&self) -> BTreeMap<&str, TensorData>;
}

pub struct Debugger<M: Introspect> {
    model: M,
}

impl<M: Introspect> Debugger<M> {
    pub fn new(model: M) -> Self {
        Self { model }
    }

    pub fn analyze<B: Backend>(&self, device: &B::Device) {
        // BTreeMap keeps an ordering lexicographically
        let weights: BTreeMap<&str, TensorData> = self.model.get_weights();
        let weight_data = weights.get("linear1").unwrap().clone();

        // Making a square matrix of (W^T)*W
        let w = Tensor::<B, 2>::from_data(weight_data, &device);
        let x = w.clone().transpose().matmul(w);

        // Finding the ESD
        let x_data = x.to_data();
        let values: Vec<f32> = x_data.to_vec().unwrap();
        let n = x.dims()[0];

        // ndarray matrix
        let matrix = Array2::from_shape_vec((n, n), values).unwrap();
        //let (eigenval, eigenvec) = matrix.eigh(UPLO::Lower).unwrap();

        println!("{:?}", matrix);
    }
}

