//! math_ndarray
//!
//! mathematical functions for ndarray

use super::util::cast_t2u;
use num_traits::{Float, FromPrimitive};
use std::collections::BinaryHeap;
use std::f64::consts::E;
use ndarray::{Array1, Array2, ArrayD, IxDyn};

const BASE_UP: f64 = 1E6;

// >>>>>>>>>>>>> MathFunc >>>>>>>>>>>>>

/// Math functions trait
pub trait MathFunc<T> {
    /// identity function
    fn identity(&self) -> Self;
    /// exp function
    fn exp(&self) -> Self;
    /// log function
    fn log(&self, e: T) -> Self;
    /// log-natural function
    fn log_natural(&self) -> Self;
    /// ReLU function
    fn relu(&self) -> Self;
    /// gradient of ReLU
    fn relu_grad(&self) -> Self;
    /// sigmoid function
    fn sigmoid(&self) -> Self;
    /// gradient of sigmoid
    fn sigmoid_grad(&self) -> Self;
    /// softmax function
    fn softmax(&self) -> Self;
    /// step function
    fn step(&self) -> Self;
    /// sqrt function
    fn sqrt(&self) -> Self;
    /// power function
    fn powf(&self, p: T) -> Self;
    /// maximum function
    fn max(&self) -> T;
    /// minimum function
    fn min(&self) -> T;
    /// variance function
    fn var(&self) -> T;
    /// std function
    fn std(&self) -> T;
    /// argmax function
    fn argmax(&self) -> IxDyn;
}

impl<T> MathFunc<T> for ArrayD<T>
where
    T: Float + FromPrimitive,
{
    fn identity(&self) -> Self {
        self.clone()
    }
    fn exp(&self) -> Self {
        self.map(|&v| v.exp())
    }
    fn log(&self, e: T) -> Self {
        self.map(|&v| v.log(e))
    }
    fn log_natural(&self) -> Self {
        let e: T = cast_t2u(E);
        self.map(|&v| v.log(e))
    }
    fn relu(&self) -> Self {
        let zero: T = cast_t2u(0.0);
        self.map(|&v| T::max(zero, v))
    }
    fn relu_grad(&self) -> Self {
        let zero: T = cast_t2u(0.0);
        let one: T = cast_t2u(1.0);
        self.map(|&v| if v < zero { zero } else { one })
    }
    fn sigmoid(&self) -> Self {
        let one: T = cast_t2u(1.0);
        self.map(|&v| one / (one + T::exp(-v)))
    }
    fn sigmoid_grad(&self) -> Self {
        let one: T = cast_t2u(1.0);
        self.sigmoid().map(|&v| (one - v) * v)
    }
    fn softmax(&self) -> Self {
        let x_max: T = self.max();
        let x2: ArrayD<T> = self.map(|&w| T::exp(w - x_max));
        let x2_sum: T = x2.sum();
        x2.map(|&v| v / x2_sum)
    }
    fn step(&self) -> Self {
        let zero: T = cast_t2u(0.0);
        let one: T = cast_t2u(1.0);
        self.map(|&v| if v <= zero { zero } else { one })
    }
    fn sqrt(&self) -> Self {
        self.map(|v| v.sqrt())
    }
    fn powf(&self, p: T) -> Self {
        self.map(|v| v.powf(p))
    }
    fn max(&self) -> T {
        let zero: T = cast_t2u(0.0);
        self.fold(zero / zero, |m, &v| v.max(m))
    }
    fn min(&self) -> T {
        let zero: T = cast_t2u(0.0);
        self.fold(zero / zero, |m, &v| v.min(m))
    }
    fn var(&self) -> T {
        if let Some(mean) = self.mean(){
            let two: T = cast_t2u(2.0);
            if let Some(res) = self.map(|&v| (v - mean).powf(two)).mean() {
                res
            } else {
                cast_t2u(0.0)
            }
        } else {
            cast_t2u(0.0)
        }
    }
    fn std(&self) -> T {
        self.var().sqrt()
    }
    fn argmax(&self) -> IxDyn {
        let mut q = BinaryHeap::new();
        for v in self.indexed_iter() {
            let mut t: Vec<usize> = Vec::new();
            for ii in 0..self.ndim() {
                t.push(v.0[ii]);
            }
            q.push((
                cast_t2u::<T, u64>(cast_t2u::<f64, T>(BASE_UP) * *v.1),
                t
            ));
        }
        IxDyn(&q.pop().unwrap().1)
    }
}

impl<T> MathFunc<T> for Array2<T>
where
    T: Float + FromPrimitive,
{
    fn identity(&self) -> Self {
        self.clone()
    }
    fn exp(&self) -> Self {
        self.map(|&v| v.exp())
    }
    fn log(&self, e: T) -> Self {
        self.map(|&v| v.log(e))
    }
    fn log_natural(&self) -> Self {
        let e: T = cast_t2u(E);
        self.map(|&v| v.log(e))
    }
    fn relu(&self) -> Self {
        let zero: T = cast_t2u(0.0);
        self.map(|&v| T::max(zero, v))
    }
    fn relu_grad(&self) -> Self {
        let zero: T = cast_t2u(0.0);
        let one: T = cast_t2u(1.0);
        self.map(|&v| if v < zero { zero } else { one })
    }
    fn sigmoid(&self) -> Self {
        let one: T = cast_t2u(1.0);
        self.map(|&v| one / (one + T::exp(-v)))
    }
    fn sigmoid_grad(&self) -> Self {
        let one: T = cast_t2u(1.0);
        self.sigmoid().map(|&v| (one - v) * v)
    }
    fn softmax(&self) -> Self {
        let x_max: T = self.max();
        let x2: Array2<T> = self.map(|&w| T::exp(w - x_max));
        let x2_sum: T = x2.sum();
        x2.map(|&v| v / x2_sum)
    }
    fn step(&self) -> Self {
        let zero: T = cast_t2u(0.0);
        let one: T = cast_t2u(1.0);
        self.map(|&v| if v <= zero { zero } else { one })
    }
    fn sqrt(&self) -> Self {
        self.map(|v| v.sqrt())
    }
    fn powf(&self, p: T) -> Self {
        self.map(|v| v.powf(p))
    }
    fn max(&self) -> T {
        let zero: T = cast_t2u(0.0);
        self.fold(zero / zero, |m, &v| v.max(m))
    }
    fn min(&self) -> T {
        let zero: T = cast_t2u(0.0);
        self.fold(zero / zero, |m, &v| v.min(m))
    }
    fn var(&self) -> T {
        if let Some(mean) = self.mean(){
            let two: T = cast_t2u(2.0);
            if let Some(res) = self.map(|&v| (v - mean).powf(two)).mean() {
                res
            } else {
                cast_t2u(0.0)
            }
        } else {
            cast_t2u(0.0)
        }
    }
    fn std(&self) -> T {
        self.var().sqrt()
    }
    fn argmax(&self) -> IxDyn {
        let mut q = BinaryHeap::new();
        for v in self.indexed_iter() {
            q.push((
                cast_t2u::<T, u64>(cast_t2u::<f64, T>(BASE_UP) * *v.1),
                v.0
            ));
        }
        IxDyn(&[q.pop().unwrap().1.0, q.pop().unwrap().1.1])
    }
}

impl<T> MathFunc<T> for Array1<T>
where
    T: Float + FromPrimitive,
{
    fn identity(&self) -> Self {
        self.clone()
    }
    fn exp(&self) -> Self {
        self.map(|&v| v.exp())
    }
    fn log(&self, e: T) -> Self {
        self.map(|&v| v.log(e))
    }
    fn log_natural(&self) -> Self {
        let e: T = cast_t2u(E);
        self.map(|&v| v.log(e))
    }
    fn relu(&self) -> Self {
        let zero: T = cast_t2u(0.0);
        self.map(|&v| T::max(zero, v))
    }
    fn relu_grad(&self) -> Self {
        let zero: T = cast_t2u(0.0);
        let one: T = cast_t2u(1.0);
        self.map(|&v| if v < zero { zero } else { one })
    }
    fn sigmoid(&self) -> Self {
        let one: T = cast_t2u(1.0);
        self.map(|&v| one / (one + T::exp(-v)))
    }
    fn sigmoid_grad(&self) -> Self {
        let one: T = cast_t2u(1.0);
        self.sigmoid().map(|&v| (one - v) * v)
    }
    fn softmax(&self) -> Self {
        let x_max: T = self.max();
        let x2: Array1<T> = self.map(|&w| T::exp(w - x_max));
        let x2_sum: T = x2.sum();
        x2.map(|&v| v / x2_sum)
    }
    fn step(&self) -> Self {
        let zero: T = cast_t2u(0.0);
        let one: T = cast_t2u(1.0);
        self.map(|&v| if v <= zero { zero } else { one })
    }
    fn sqrt(&self) -> Self {
        self.map(|v| v.sqrt())
    }
    fn powf(&self, p: T) -> Self {
        self.map(|v| v.powf(p))
    }
    fn max(&self) -> T {
        let zero: T = cast_t2u(0.0);
        self.fold(zero / zero, |m, &v| v.max(m))
    }
    fn min(&self) -> T {
        let zero: T = cast_t2u(0.0);
        self.fold(zero / zero, |m, &v| v.min(m))
    }
    fn var(&self) -> T {
        if let Some(mean) = self.mean(){
            let two: T = cast_t2u(2.0);
            if let Some(res) = self.map(|&v| (v - mean).powf(two)).mean() {
                res
            } else {
                cast_t2u(0.0)
            }
        } else {
            cast_t2u(0.0)
        }
    }
    fn std(&self) -> T {
        self.var().sqrt()
    }
    fn argmax(&self) -> IxDyn {
        let mut q = BinaryHeap::new();
        for v in self.indexed_iter() {
            q.push((
                cast_t2u::<T, u64>(cast_t2u::<f64, T>(BASE_UP) * *v.1),
                v.0
            ));
        }
        IxDyn(&[q.pop().unwrap().1])
    }
}
