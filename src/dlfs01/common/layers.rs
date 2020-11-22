//! layers
//!
//! basic layers

use rand::prelude::*;
use rand::distributions::Uniform;
use super::util::cast_t2u;
use ndarray::{ArrayD, IxDyn};
use num_traits::{Num, NumCast};
use std::marker::PhantomData;

pub trait Layer<T> {
    fn new(shape: IxDyn) -> Self;
    fn forward(&mut self, x: &ArrayD<T>) -> ArrayD<T>;
    fn backward(&self, dx: &ArrayD<T>) -> ArrayD<T>;
}

pub struct ReLU<T> {
    mask: ArrayD<u8>,
    phantom: PhantomData<T>,
}

impl<T> Layer<T> for ReLU<T>
where
    T: Num + NumCast + Copy + PartialOrd,
{
    fn new(shape: IxDyn) -> Self {
        ReLU {
            mask: ArrayD::<u8>::ones(shape),
            phantom: PhantomData
        }
    }
    fn forward(&mut self, x: &ArrayD<T>) -> ArrayD<T> {
        let zero: T = cast_t2u(0.0);
        self.mask = x.map(|&v| if v <= zero { 1 } else { 0 });
        x.map(|&v| if v <= zero { zero } else { v })
    }
    fn backward(&self, dx: &ArrayD<T>) -> ArrayD<T> {
        let zero: T = cast_t2u(0.0);
        let mut dst = dx.clone();
        for v in self.mask.indexed_iter() {
            if *v.1 != 0u8 {
                dst[v.0] = zero;
            }
        }
        dst
    }
}

pub fn main() {
    println!("< layers sub module> ");
    let mut rng = rand::thread_rng();
    let gen = Uniform::new(-1.0f32, 1.0f32);
    let a = ArrayD::<f32>::zeros(IxDyn(&[2, 3]));
    let a = a.map(|_| gen.sample(&mut rng));
    let mut layer_relu = ReLU::<f32>::new(IxDyn(a.shape()));
    let b = layer_relu.forward(&a);
    let da = ArrayD::<f32>::zeros(IxDyn(a.shape())).map(|_| gen.sample(&mut rng));
    println!("target: {}", a);
    println!("relu forward: {}", b);
    println!("target d: {}", da);
    println!("relu backward: {}", layer_relu.backward(&da));
}