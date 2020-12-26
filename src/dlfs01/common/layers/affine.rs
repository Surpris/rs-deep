//! affine
//!
//! Affine layer

#![allow(unused_imports)]

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
    pub fn new(shape: (usize, usize)) -> Self {
        let mut rng = rand::thread_rng();
        let gen = Uniform::new(-1.0f32, 1.0f32);
        Affine {
            weight: Array2::<T>::ones(shape).map(|_| cast_t2u(gen.sample(&mut rng))),
            bias: Array1::<T>::ones(shape.1).map(|_| cast_t2u(gen.sample(&mut rng))),
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
        let mut dst: Array2<T> = x.dot(&self.weight);
        for v in dst.indexed_iter_mut() {
            *v.1 = *v.1 + self.bias[v.0 .1];
        }
        dst
    }
    fn backward(&mut self, dx: &Self::B) -> Self::A {
        let dst = dx.dot(&self.weight.t());
        self.dw = self.buff.t().dot(dx);
        self.db = dx.sum_axis(Axis(0));
        dst
    }
    fn update(&mut self, lr: T) {
        for v in self.weight.indexed_iter_mut() {
            *v.1 = *v.1 - lr * self.dw[v.0];
        }
        for v in self.bias.indexed_iter_mut() {
            *v.1 = *v.1 - lr * self.db[v.0];
        }
    }
    fn print_detail(&self) {
        println!("affine layer.");
        println!("weight shape: {:?}", self.dw.shape());
        println!("bias shape: {:?}", self.bias.shape());
    }
    fn print_parameters(&self) {
        println!("weight: {:?}", self.weight);
        println!("bias: {:?}", self.bias);
        println!("dw: {:?}", self.dw);
        println!("db: {:?}", self.db);
    }
}
