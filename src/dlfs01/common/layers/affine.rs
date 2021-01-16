//! affine
//!
//! Affine layer

#![allow(unused_imports)]

use super::super::param_initializers::weight_init::{initialize_weight, WeightInitEnum};
use super::super::util::*;
use super::layer_base::LayerBase;
use ndarray::prelude::*;
use rand::distributions::Uniform;
use rand::prelude::*;
use std::fmt::{Debug, Display};

/// Affine layer
#[derive(Clone)]
pub struct Affine<T: CrateFloat> {
    pub weight: Array2<T>,
    pub bias: Array1<T>,
    pub dw: Array2<T>,
    pub db: Array1<T>,
    buff: Array2<T>,
}

impl<T> Affine<T>
where
    T: CrateFloat,
{
    pub fn new(shape: (usize, usize), weight_init: WeightInitEnum, weight_init_std: T) -> Self {
        Affine {
            weight: initialize_weight(weight_init.clone(), weight_init_std, shape),
            bias: initialize_weight(weight_init, weight_init_std, shape.1),
            dw: Array2::<T>::ones(shape),
            db: Array1::<T>::ones(shape.1),
            buff: Array2::<T>::ones(shape),
        }
    }
    pub fn from(weight: &Array2<T>, bias: &Array1<T>) -> Self {
        let shape = weight.shape();
        Affine {
            weight: weight.clone(),
            bias: bias.clone(),
            dw: Array2::<T>::ones((shape[0], shape[1])),
            db: Array1::<T>::ones(shape[1]),
            buff: Array2::<T>::ones((shape[0], shape[1])),
        }
    }
}

impl<T: 'static> LayerBase<T> for Affine<T>
where
    T: CrateFloat,
{
    type A = Array2<T>;
    type B = Array2<T>;
    fn forward(&mut self, x: &Self::A) -> Self::B {
        self.buff = x.clone();
        x.dot(&self.weight) + &self.bias
    }
    fn backward(&mut self, dx: &Self::B) -> Self::A {
        let dst = dx.dot(&self.weight.t());
        self.dw = self.buff.t().dot(dx);
        self.db = dx.sum_axis(Axis(0));
        dst
    }
    fn update(&mut self, lr: T) {
        self.weight.scaled_add(-lr, &self.dw);
        self.bias.scaled_add(-lr, &self.db);
    }
    fn print_detail(&self) {
        println!("affine layer.");
        println!("weight shape: {:?}", self.weight.shape());
        println!("bias shape: {:?}", self.bias.shape());
    }
    fn print_parameters(&self) {
        println!("weight: {:?}", self.weight);
        println!("bias: {:?}", self.bias);
        println!("dw: {:?}", self.dw);
        println!("db: {:?}", self.db);
    }
}
