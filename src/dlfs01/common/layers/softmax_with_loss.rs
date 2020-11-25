//! softmax_with_loss
//!
//! custom layer: combination of softmax and loss

use super::super::util::cast_t2u;
use ndarray::{ArrayD, Axis, IxDyn};
use num_traits::Float;
use rand::distributions::Uniform;
use rand::prelude::*;
use std::f64::consts::E;

const EPS: f64 = 1E-8;

/// SoftmaxWithLoss layer trait
pub trait SoftmaxWithLossBase<T> {
    fn new(shape: &[usize]) -> Self;
    fn forward(&mut self, x: &ArrayD<T>, t: &ArrayD<T>) -> T;
    fn backward(&self, _dx: T) -> ArrayD<T>;
    fn print_detail(&self);
}

/// softmax layer
pub struct SoftmaxWithLoss<T> {
    output: ArrayD<T>,
    target: ArrayD<T>,
    loss: T,
}

impl<T: 'static> SoftmaxWithLossBase<T> for SoftmaxWithLoss<T>
where
    T: Float,
{
    fn new(shape: &[usize]) -> Self {
        SoftmaxWithLoss {
            output: ArrayD::<T>::zeros(IxDyn(shape)),
            target: ArrayD::<T>::zeros(IxDyn(shape)),
            loss: cast_t2u(0.0),
        }
    }
    fn forward(&mut self, x: &ArrayD<T>, t: &ArrayD<T>) -> T {
        let zero: T = cast_t2u(0.0);
        let eps: T = cast_t2u(EPS);
        let e: T = cast_t2u(E);
        let batch_size: T = cast_t2u(x.len_of(Axis(0)));

        let mut dst = x.clone();
        for mut view in dst.axis_iter_mut(Axis(0)) {
            let v_max = view.fold(zero / zero, |m, &v| v.max(m));
            view.mapv_inplace(|v| T::exp(v - v_max));
            let v_exp_sum = view.sum();
            view.mapv_inplace(|v| v / v_exp_sum);
        }
        self.output = dst.clone();
        self.target = t.clone();
        self.loss = dst
            .indexed_iter()
            .fold(zero, |m, (ax, v)| m - t[ax] * (*v + eps).log(e));
        // self.loss = zero;
        // for v in dst.indexed_iter() {
        //     self.loss = self.loss - t[v.0] * (*v.1 + eps).log(e);
        // }
        self.loss / batch_size
    }
    fn backward(&self, _dx: T) -> ArrayD<T> {
        let batch_size: T = cast_t2u(self.target.len_of(Axis(0)));
        // TODO: correct implementation
        let mut dst = ArrayD::<T>::zeros(self.target.shape());
        for v in self.target.indexed_iter() {
            dst[v.0.clone()] = (self.output[v.0.clone()] - *v.1) / batch_size;
        }
        dst
    }
    fn print_detail(&self) {
        println!("softmax-with-loss layer.");
    }
}

pub fn main() {
    println!("< softmax-with-loss sub module >");
    let mut rng = rand::thread_rng();
    let gen = Uniform::new(-1.0f32, 1.0f32);

    let a = ArrayD::<f32>::zeros(IxDyn(&[2, 3]));
    let a = a.map(|_| gen.sample(&mut rng));

    let mut t = ArrayD::<f32>::zeros(IxDyn(a.shape()));
    t[[0, 0]] = 1.0;
    t[[1, 1]] = 1.0;

    println!("softmax-with-loss layer");
    let mut layer: SoftmaxWithLoss<f32> = SoftmaxWithLoss::<f32>::new(a.shape());
    let b = layer.forward(&a, &t);
    println!("a: {}", a);
    println!("t: {}", t);
    println!("output: {}", layer.output);
    println!("forward (loss): {}", b);
    println!("backward: {}", layer.backward(1.0));
}
