//! affine
//!
//! Affine layer

#![allow(unused_imports)]

use super::super::util::cast_t2u;
use super::layer_base::LayerBase;
use ndarray::{Array, Array1, Array2, Axis, Ix2};
use num_traits::Float;
use rand::distributions::Uniform;
use rand::prelude::*;

/// Affine layer
pub struct Affine<T> {
    pub weight: Array2<T>,
    pub bias: Array1<T>,
    pub dw: Array2<T>,
    pub db: Array1<T>,
    buff: Array2<T>,
}

impl<T> Affine<T>
where
    T: Float,
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

impl<T: 'static> LayerBase<T, Ix2> for Affine<T>
where
    T: Float,
{
    fn forward(&mut self, x: &Array2<T>) -> Array2<T> {
        self.buff = x.clone();
        let mut dst: Array2<T> = x.dot(&self.weight);
        for v in dst.indexed_iter_mut() {
            *v.1 = *v.1 + self.bias[v.0 .1];
        }
        dst
    }
    fn backward(&mut self, dx: &Array2<T>) -> Array2<T> {
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
}
