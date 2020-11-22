//! activation
//!
//! Activation layers

use super::super::math::sigmoid;
use super::super::util::cast_t2u;
use ndarray::{ArrayD, IxDyn};
use num_traits::Float;
use rand::distributions::Uniform;
use rand::prelude::*;
use std::marker::PhantomData;

/// Activation layer trait
pub trait ActivationBase<T> {
    fn new(shape: &[usize]) -> Self;
    fn forward(&mut self, x: &ArrayD<T>) -> ArrayD<T>;
    fn backward(&self, dx: &ArrayD<T>) -> ArrayD<T>;
    fn print_detail(&self);
}

/// ReLU layer
pub struct ReLU<T> {
    mask: ArrayD<u8>,
    phantom: PhantomData<T>,
}

impl<T: 'static> ActivationBase<T> for ReLU<T>
where
    T: Float,
{
    fn new(shape: &[usize]) -> Self {
        ReLU {
            mask: ArrayD::<u8>::zeros(IxDyn(shape)),
            phantom: PhantomData,
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
    fn print_detail(&self) {
        println!("ReLU activation layer.");
    }
}

/// Sigmoid layer
pub struct Sigmoid<T> {
    output: ArrayD<T>,
}

impl<T: 'static> ActivationBase<T> for Sigmoid<T>
where
    T: Float,
{
    fn new(shape: &[usize]) -> Self {
        Sigmoid {
            output: ArrayD::<T>::zeros(IxDyn(shape)),
        }
    }
    fn forward(&mut self, x: &ArrayD<T>) -> ArrayD<T> {
        self.output = x.map(|&v| sigmoid(v));
        self.output.clone()
    }
    fn backward(&self, dx: &ArrayD<T>) -> ArrayD<T> {
        let one: T = cast_t2u(1.0);
        let mut dst = dx.clone();
        for v in self.output.indexed_iter() {
            dst[v.0] = *v.1 * (one - *v.1) * *v.1;
        }
        dst
    }
    fn print_detail(&self) {
        println!("sigmoid activation layer.");
    }
}

pub fn main() {
    println!("< activation sub module> ");
    let mut rng = rand::thread_rng();
    let gen = Uniform::new(-1.0f32, 1.0f32);

    let a = ArrayD::<f32>::zeros(IxDyn(&[2, 3]));
    let a = a.map(|_| gen.sample(&mut rng));

    let mut layer_relu = ReLU::<f32>::new(a.shape());
    let b = layer_relu.forward(&a);
    let da = ArrayD::<f32>::zeros(IxDyn(a.shape())).map(|_| gen.sample(&mut rng));
    println!("target: {}", a);
    println!("relu forward: {}", b);
    println!("target d: {}", da);
    println!("relu backward: {}", layer_relu.backward(&da));

    let mut layer_sigmoid = Sigmoid::<f32>::new(a.shape());
    let b = layer_sigmoid.forward(&a);
    let da = ArrayD::<f32>::zeros(IxDyn(a.shape())).map(|_| gen.sample(&mut rng));
    println!("target: {}", a);
    println!("sigmoid forward: {}", b);
    println!("target d: {}", da);
    println!("sigmoid backward: {}", layer_sigmoid.backward(&da));
}
