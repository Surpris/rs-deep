//! activation
//!
//! Activation layers

use super::super::math::sigmoid;
use super::super::util::cast_t2u;
use ndarray::{ArrayD, Axis, IxDyn};
use num_traits::Float;
use rand::distributions::Uniform;
use rand::prelude::*;
use std::marker::PhantomData;

/// Activation layer trait
pub trait ActivationBase<T> {
    fn forward(&mut self, x: &ArrayD<T>) -> ArrayD<T>;
    fn backward(&self, dx: &ArrayD<T>) -> ArrayD<T>;
    fn print_detail(&self);
}

/// ReLU layer
pub struct ReLU<T> {
    mask: ArrayD<u8>,
    phantom: PhantomData<T>,
}

impl<T: 'static> ReLU<T>
where
    T: Float,
{
    pub fn new(shape: &[usize]) -> Self {
        ReLU {
            mask: ArrayD::<u8>::zeros(IxDyn(shape)),
            phantom: PhantomData,
        }
    }
}

impl<T: 'static> ActivationBase<T> for ReLU<T>
where
    T: Float,
{
    fn forward(&mut self, x: &ArrayD<T>) -> ArrayD<T> {
        if self.mask.shape() != x.shape() {
            self.mask = ArrayD::<u8>::zeros(IxDyn(x.shape()))
        }
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

/// sigmoid layer
pub struct Sigmoid<T> {
    output: ArrayD<T>,
}

impl<T: 'static> Sigmoid<T>
where
    T: Float,
{
    pub fn new(shape: &[usize]) -> Self {
        Sigmoid {
            output: ArrayD::<T>::zeros(IxDyn(shape)),
        }
    }
}

impl<T: 'static> ActivationBase<T> for Sigmoid<T>
where
    T: Float,
{
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

/// Activation layer trait
// pub trait SoftmaxBase<T> {
//     // fn new(shape: &[usize], axis: usize) -> Self;
//     fn forward(&mut self, x: &ArrayD<T>) -> ArrayD<T>;
//     fn backward(&self, dx: &ArrayD<T>) -> ArrayD<T>;
//     fn print_detail(&self);
// }

/// softmax layer
pub struct Softmax<T> {
    output: ArrayD<T>,
    axis: usize,
}

impl<T: 'static> Softmax<T>
where
    T: Float,
{
    pub fn new(shape: &[usize], axis: usize) -> Self {
        Softmax {
            output: ArrayD::<T>::zeros(IxDyn(shape)),
            axis,
        }
    }
}

impl<T: 'static> ActivationBase<T> for Softmax<T>
where
    T: Float,
{
    fn forward(&mut self, x: &ArrayD<T>) -> ArrayD<T> {
        let zero: T = cast_t2u(0.0);
        let mut dst = x.clone();
        for mut view in dst.axis_iter_mut(Axis(self.axis)) {
            let v_max = view.fold(zero / zero, |m, &v| v.max(m));
            view.mapv_inplace(|v| T::exp(v - v_max));
            let v_exp_sum = view.sum();
            view.mapv_inplace(|v| v / v_exp_sum);
        }
        self.output = dst;
        self.output.clone()
    }
    fn backward(&self, dx: &ArrayD<T>) -> ArrayD<T> {
        // TODO: correct implementation
        dx.clone()
    }
    fn print_detail(&self) {
        println!("softmax activation layer.");
    }
}

pub fn main() {
    println!("< activation sub module> ");
    let mut rng = rand::thread_rng();
    let gen = Uniform::new(-1.0f32, 1.0f32);

    let a = ArrayD::<f32>::zeros(IxDyn(&[2, 3]));
    let a = a.map(|_| gen.sample(&mut rng));

    println!("ReLU layer");
    let mut layer = ReLU::<f32>::new(a.shape());
    let b = layer.forward(&a);
    let da = ArrayD::<f32>::zeros(IxDyn(a.shape())).map(|_| gen.sample(&mut rng));
    println!("a: {}", a);
    println!("forward: {}", b);
    println!("da: {}", da);
    println!("backward: {}", layer.backward(&da));

    println!("sigmoid layer");
    let mut layer = Sigmoid::<f32>::new(a.shape());
    let b = layer.forward(&a);
    let da = ArrayD::<f32>::zeros(IxDyn(a.shape())).map(|_| gen.sample(&mut rng));
    println!("a: {}", a);
    println!("forward: {}", b);
    println!("da: {}", da);
    println!("backward: {}", layer.backward(&da));

    println!("softmax layer");
    let mut layer = Softmax::<f32>::new(a.shape(), 0);
    let b = layer.forward(&a);
    let da = ArrayD::<f32>::zeros(IxDyn(a.shape())).map(|_| gen.sample(&mut rng));
    println!("a: {}", a);
    println!("forward: {}, {}", b, b.sum_axis(Axis(1)));
    println!("da: {}", da);
    println!("backward: {}", layer.backward(&da));
}
