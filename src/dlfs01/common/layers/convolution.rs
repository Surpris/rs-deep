//! convolution
//!
//! Convolution layer

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use super::super::param_initializers::weight_init::{initialize_weight, WeightInitEnum};
use super::super::util::*;
use super::layer_base::LayerBase;
use ndarray::{prelude::*, RemoveAxis};
use rand::distributions::Uniform;
use rand::prelude::*;
use std::fmt::{Debug, Display};

/// Convolution layer
///
/// Images to train with must be "channel-first".
#[derive(Clone)]
pub struct Convolution<T: CrateFloat, D> {
    weight: Array<T, D>,
    bias: Array1<T>,
    stride_sizes: Vec<usize>,
    pad_size: usize,
    x: Array<T, D>,
    col: Array<T, D>,
    col_weight: Array<T, D>,
    dw: Array<T, D>,
    db: Array1<T>,
    padded_shape: Vec<usize>,
    output_data_shape: Vec<usize>,
}

pub type Convolution1<T> = Convolution<T, Ix3>;
pub type Convolution2<T> = Convolution<T, Ix4>;
pub type Convolution3<T> = Convolution<T, Ix5>;

impl<T> Convolution2<T>
where
    T: CrateFloat,
{
    /// generate a Convolution2 layer.
    ///
    /// `input_shape` is (n_channel, height, width).
    pub fn new(
        filter_size: usize,
        filter_shape: (usize, usize),
        data_shape: (usize, usize, usize),
        stride_sizes: (usize, usize),
        pad_size: usize,
        weight_init: WeightInitEnum,
        weight_init_std: T,
    ) -> Self {
        let weight_shape: (usize, usize, usize, usize) =
            (filter_size, data_shape.0, filter_shape.0, filter_shape.1);
        let input_shape: (usize, usize, usize, usize) =
            (1usize, data_shape.0, data_shape.1, data_shape.2);
        Self::from(
            &initialize_weight(weight_init.clone(), weight_init_std, weight_shape),
            &initialize_weight(weight_init, weight_init_std, filter_size),
            data_shape,
            stride_sizes,
            pad_size,
        )
    }
    pub fn from(
        weight: &Array4<T>,
        bias: &Array1<T>,
        data_shape: (usize, usize, usize),
        stride_sizes: (usize, usize),
        pad_size: usize,
    ) -> Self {
        let weight_shape = weight.shape();
        let mut output_h: usize = data_shape.1 + 2 * pad_size - weight_shape[2];
        let mut output_w: usize = data_shape.2 + 2 * pad_size - weight_shape[3];
        if output_h % stride_sizes.0 != 0 {
            panic!("Invalid inputs with regard to the shapes.");
        }
        if output_w % stride_sizes.1 != 0 {
            panic!("Invalid inputs with regard to the shapes.");
        }
        output_h = 1usize + output_h / stride_sizes.0;
        output_w = 1usize + output_w / stride_sizes.1;
        let input_shape: (usize, usize, usize, usize) =
            (1usize, data_shape.0, data_shape.1, data_shape.2);
        let padded_shape: Vec<usize> =
            vec![data_shape.1 + 2 * pad_size, data_shape.2 + 2 * pad_size];
        let output_data_shape: Vec<usize> = vec![output_h, output_w];
        Self {
            weight: weight.clone(),
            bias: bias.clone(),
            stride_sizes: vec![stride_sizes.0, stride_sizes.1],
            pad_size,
            x: Array4::zeros(input_shape),
            col: Array4::zeros(input_shape),
            col_weight: Array4::zeros(input_shape),
            dw: Array4::zeros(weight.raw_dim()),
            db: Array1::zeros(bias.raw_dim()),
            padded_shape,
            output_data_shape,
        }
    }
    pub fn get_output_data_shape(&self) -> (usize, usize, usize) {
        (
            self.weight.shape()[1],
            self.output_data_shape[0],
            self.output_data_shape[1],
        )
    }
}

impl<T: 'static> LayerBase<T> for Convolution2<T>
where
    T: CrateFloat,
{
    type A = Array4<T>;

    type B = Array4<T>;

    fn forward(&mut self, x: &Self::A) -> Self::B {
        let n_images: usize = x.shape()[0];
        let dst: Self::A = Array4::zeros((
            n_images,
            self.weight.shape()[1],
            self.output_data_shape[0],
            self.output_data_shape[1],
        ));
        // let mut buff: Array2<T> = Array2::zeros((self.padded_shape[0], self.padded_shape[1]));
        dst
    }

    fn backward(&mut self, dx: &Self::B) -> Self::A {
        todo!()
    }

    fn print_detail(&self) {
        println!("2D convolution layer.");
        println!("weight shape: {:?}", self.weight.shape());
        println!("bias shape: {:?}", self.bias.shape());
        println!(
            "stride size: ({}, {})",
            self.stride_sizes[0], self.stride_sizes[1]
        );
        println!("pad size: {}", self.pad_size);
    }
    fn print_parameters(&self) {
        println!("weight: {:?}", self.weight);
        println!("bias: {:?}", self.bias);
        println!("dw: {:?}", self.dw);
        println!("db: {:?}", self.db);
    }
}
