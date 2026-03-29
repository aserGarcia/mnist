extern crate nalgebra as na;
use burn::{Tensor, prelude::Backend, tensor::TensorData};
use plotters::prelude::*;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};

pub trait Introspect {
    fn get_weights(&self) -> BTreeMap<&str, TensorData>;
}

pub struct LayerAnalysis {
    name: String,
    eigenvalues: Vec<f32>,
}

pub struct Debugger<M: Introspect> {
    model: M,
    eigenvals: Option<Vec<LayerAnalysis>>,
}

impl<M: Introspect> Debugger<M> {
    pub fn new(model: M) -> Self {
        Self {
            model,
            eigenvals: None,
        }
    }

    pub fn analyze<B: Backend>(&mut self, device: &B::Device) {
        // BTreeMap keeps an ordering lexicographically
        let weights: BTreeMap<&str, TensorData> = self.model.get_weights();
        let eigenvals: Vec<LayerAnalysis> = weights
            .par_iter()
            .map(|(layer, weight)| {
                let weight_data = weight.clone();

                // Making a square matrix of (W^T)*W
                let w = Tensor::<B, 2>::from_data(weight_data, &device);
                let x = w.clone().transpose().matmul(w);

                // Finding the ESD
                let x_data = x.to_data();
                let values: Vec<f32> = x_data.to_vec().unwrap();
                let n = x.dims()[0];
                let matrix = na::DMatrix::from_vec(n, n, values);

                let out = matrix.eigenvalues();
                LayerAnalysis {
                    name: layer.to_string(),
                    eigenvalues: out.unwrap().data.as_vec().clone(),
                }
            })
            .collect();

        for ev in &eigenvals {
            let name = format!("plots/{}.png", ev.name);
            let root = BitMapBackend::new(&name, (600, 400)).into_drawing_area();
            root.fill(&WHITE).unwrap();

            let data: Vec<(usize, f32)> = (0..).zip(ev.eigenvalues.clone()).collect();

            let max_val = data.iter().map(|(_, v)| *v).fold(0.0_f32, f32::max);

            let mut chart = ChartBuilder::on(&root)
                .caption(ev.name.clone(), ("sans-serif", 40))
                .margin(20)
                .x_label_area_size(40)
                .y_label_area_size(50)
                .build_cartesian_2d(
                    0..data.len(),          // x axis: bar indices
                    0.0_f32..max_val * 1.1, // y axis: f32 range with 10% headroom
                )
                .unwrap();

            chart
                .configure_mesh()
                .x_labels(data.len())
                .x_label_formatter(&|i| {
                    data.get(*i)
                        .map(|(label, _)| label.to_string())
                        .unwrap_or_default()
                })
                .draw()
                .unwrap();

            chart
                .draw_series(data.iter().enumerate().map(|(i, (_, val))| {
                    Rectangle::new([(i, 0.0_f32), (i + 1, *val)], BLUE.filled())
                }))
                .unwrap();
            root.present();
        }
        self.eigenvals = Some(eigenvals);
        println!("Done");
    }
}
