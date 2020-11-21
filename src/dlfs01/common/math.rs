//! math
//!
//! mathematical functions

use super::operators::Operators;
use super::util::cast_t2u;
use num_traits::Float;
use rand::distributions::Uniform;
use rand::prelude::*;
use std::collections::BinaryHeap;
use std::f64::consts::E;

type Vec2d<T> = Vec<Vec<T>>;
type Vec3d<T> = Vec<Vec2d<T>>;
// type Vec4d<T> = Vec<Vec3d<T>>;

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
    /// summation function
    fn sum(&self) -> T;
    /// mean function
    fn mean(&self) -> T;
    /// variance function
    fn var(&self) -> T;
    /// std function
    fn std(&self) -> T;
    /// argmax function
    fn argmax(&self) -> usize;
}

impl<T> MathFunc<T> for Vec<T>
where
    T: Float,
{
    fn identity(&self) -> Self {
        self.clone()
    }
    fn exp(&self) -> Self {
        self.iter().map(|&v| v.exp()).collect()
    }
    fn log(&self, e: T) -> Self {
        self.iter().map(|&v| v.log(e)).collect()
    }
    fn log_natural(&self) -> Self {
        let e: T = cast_t2u(E);
        self.iter().map(|&v| v.log(e)).collect()
    }
    fn relu(&self) -> Self {
        let zero: T = cast_t2u(0.0);
        self.iter().map(|&v| T::max(zero, v)).collect()
    }
    fn relu_grad(&self) -> Self {
        let zero: T = cast_t2u(0.0);
        let one: T = cast_t2u(1.0);
        self.iter()
            .map(|&v| if v < zero { zero } else { one })
            .collect()
    }
    fn sigmoid(&self) -> Self {
        let one: T = cast_t2u(1.0);
        self.iter().map(|&v| one / (one + T::exp(-v))).collect()
    }
    fn sigmoid_grad(&self) -> Self {
        let one: T = cast_t2u(1.0);
        let vec: Vec<T> = self.sigmoid();
        (0..vec.len()).map(|ii| (one - vec[ii]) * vec[ii]).collect()
    }
    fn softmax(&self) -> Self {
        let x_max: T = self.clone().max();
        let x2: Vec<T> = self.iter().map(|&w| T::exp(w - x_max)).collect();
        let x2_sum: T = x2.clone().sum();
        x2.iter().map(|&v| v / x2_sum).collect()
    }
    fn step(&self) -> Self {
        let zero: T = cast_t2u(0.0);
        let one: T = cast_t2u(1.0);
        self.iter()
            .map(|&v| if v <= zero { zero } else { one })
            .collect()
    }
    fn sqrt(&self) -> Self {
        self.iter().map(|v| v.sqrt()).collect()
    }
    fn powf(&self, p: T) -> Self {
        self.iter().map(|v| v.powf(p)).collect()
    }
    fn max(&self) -> T {
        let zero: T = cast_t2u(0.0);
        self.iter().fold(zero / zero, |m, &v| v.max(m))
    }
    fn min(&self) -> T {
        let zero: T = cast_t2u(0.0);
        self.iter().fold(zero / zero, |m, &v| v.min(m))
    }
    fn sum(&self) -> T {
        let zero: T = cast_t2u(0.0);
        self.iter().fold(zero, |m, &v| m + v)
    }
    fn mean(&self) -> T {
        self.sum() / cast_t2u(self.len())
    }
    fn var(&self) -> T {
        let mean = self.mean();
        let two: T = cast_t2u(2.0);
        self.sub_value(mean).powf(two).mean()
    }
    fn std(&self) -> T {
        let mean = self.mean();
        let n: T = cast_t2u(self.len() - 1);
        let two: T = cast_t2u(2.0);
        (self.sub_value(mean).powf(two).sum() / n).sqrt()
    }
    fn argmax(&self) -> usize {
        let mut q = BinaryHeap::new();
        for ii in 0..self.len() {
            q.push((
                cast_t2u::<T, u64>(cast_t2u::<f64, T>(BASE_UP) * self[ii]),
                ii,
            ));
        }
        q.pop().unwrap().1
    }
}

impl<T> MathFunc<T> for Vec2d<T>
where
    T: Float,
{
    fn identity(&self) -> Self {
        self.clone()
    }
    fn exp(&self) -> Self {
        self.iter().map(|v| v.exp()).collect()
    }
    fn log(&self, e: T) -> Self {
        self.iter().map(|v| v.log(e)).collect()
    }
    fn log_natural(&self) -> Self {
        let e: T = cast_t2u(E);
        self.iter().map(|v| v.log(e)).collect()
    }
    fn relu(&self) -> Self {
        self.iter().map(|v| v.relu()).collect()
    }
    fn relu_grad(&self) -> Self {
        self.iter().map(|v| v.relu_grad()).collect()
    }
    fn sigmoid(&self) -> Self {
        self.iter().map(|v| v.sigmoid()).collect()
    }
    fn sigmoid_grad(&self) -> Self {
        self.iter().map(|v| v.sigmoid_grad()).collect()
    }
    fn softmax(&self) -> Self {
        let x_max: T = self.clone().max();
        let x2: Vec2d<T> = self.iter().map(|w| w.exp().sub_value(x_max)).collect();
        let x2_sum: T = x2.clone().sum();
        x2.iter().map(|v| v.div_value(x2_sum)).collect()
    }
    fn step(&self) -> Self {
        self.iter().map(|v| v.step()).collect()
    }
    fn sqrt(&self) -> Self {
        self.iter().map(|v| v.sqrt()).collect()
    }
    fn powf(&self, p: T) -> Self {
        self.iter().map(|v| v.powf(p)).collect()
    }
    fn max(&self) -> T {
        let zero: T = cast_t2u(0.0);
        self.iter()
            .map(|v| v.max())
            .fold(zero / zero, |m, v| v.max(m))
    }
    fn min(&self) -> T {
        let zero: T = cast_t2u(0.0);
        self.iter()
            .map(|v| v.min())
            .fold(zero / zero, |m, v| v.min(m))
    }
    fn sum(&self) -> T {
        self.iter().map(|v| v.sum()).collect::<Vec<T>>().sum()
    }
    fn mean(&self) -> T {
        self.iter().map(|v| v.mean()).collect::<Vec<T>>().mean()
    }
    fn var(&self) -> T {
        let mean = self.mean();
        let two: T = cast_t2u(2.0);
        self.sub_value(mean).powf(two).mean()
    }
    fn std(&self) -> T {
        let mean = self.mean();
        let n: T = cast_t2u(self.len() - 1);
        let two: T = cast_t2u(2.0);
        (self.sub_value(mean).powf(two).sum() / n).sqrt()
    }
    fn argmax(&self) -> usize {
        let mut q = BinaryHeap::new();
        for ii in 0..self.len() {
            for jj in 0..self[0].len() {
                q.push((
                    cast_t2u::<T, u64>(cast_t2u::<f64, T>(BASE_UP) * self[ii][jj]),
                    (ii, jj),
                ));
            }
        }
        let argmax_2d = q.pop().unwrap().1;
        argmax_2d.0 * self[0].len() + argmax_2d.1
    }
}

impl<T> MathFunc<T> for Vec3d<T>
where
    T: Float,
{
    fn identity(&self) -> Self {
        self.clone()
    }
    fn exp(&self) -> Self {
        self.iter().map(|v| v.exp()).collect()
    }
    fn log(&self, e: T) -> Self {
        self.iter().map(|v| v.log(e)).collect()
    }
    fn log_natural(&self) -> Self {
        let e: T = cast_t2u(E);
        self.iter().map(|v| v.log(e)).collect()
    }
    fn relu(&self) -> Self {
        self.iter().map(|v| v.relu()).collect()
    }
    fn relu_grad(&self) -> Self {
        self.iter().map(|v| v.relu_grad()).collect()
    }
    fn sigmoid(&self) -> Self {
        self.iter().map(|v| v.sigmoid()).collect()
    }
    fn sigmoid_grad(&self) -> Self {
        self.iter().map(|v| v.sigmoid_grad()).collect()
    }
    fn softmax(&self) -> Self {
        let x_max: T = self.clone().max();
        let x2: Vec3d<T> = self.iter().map(|w| w.exp().sub_value(x_max)).collect();
        let x2_sum: T = x2.clone().sum();
        x2.iter().map(|v| v.div_value(x2_sum)).collect()
    }
    fn step(&self) -> Self {
        self.iter().map(|v| v.step()).collect()
    }
    fn sqrt(&self) -> Self {
        self.iter().map(|v| v.sqrt()).collect()
    }
    fn powf(&self, p: T) -> Self {
        self.iter().map(|v| v.powf(p)).collect()
    }
    fn max(&self) -> T {
        let zero: T = cast_t2u(0.0);
        self.iter()
            .map(|v| v.max())
            .fold(zero / zero, |m, v| v.max(m))
    }
    fn min(&self) -> T {
        let zero: T = cast_t2u(0.0);
        self.iter()
            .map(|v| v.min())
            .fold(zero / zero, |m, v| v.min(m))
    }
    fn sum(&self) -> T {
        self.iter().map(|v| v.sum()).collect::<Vec<T>>().sum()
    }
    fn mean(&self) -> T {
        self.iter().map(|v| v.mean()).collect::<Vec<T>>().mean()
    }
    fn var(&self) -> T {
        let mean = self.mean();
        let two: T = cast_t2u(2.0);
        self.sub_value(mean).powf(two).mean()
    }
    fn std(&self) -> T {
        let mean = self.mean();
        let n: T = cast_t2u(self.len() - 1);
        let two: T = cast_t2u(2.0);
        (self.sub_value(mean).powf(two).sum() / n).sqrt()
    }
    fn argmax(&self) -> usize {
        let mut q = BinaryHeap::new();
        for ii in 0..self.len() {
            for jj in 0..self[0].len() {
                for kk in 0..self[0][0].len() {
                    q.push((
                        cast_t2u::<T, u64>(cast_t2u::<f64, T>(BASE_UP) * self[ii][jj][kk]),
                        (ii, jj, kk),
                    ));
                }
            }
        }
        let argmax_2d = q.pop().unwrap().1;
        argmax_2d.0 * self[0].len() * self[0][0].len()
            + argmax_2d.1 * self[0][0].len()
            + argmax_2d.2
    }
}
// <<<<<<<<<<<<< MathFunc <<<<<<<<<<<<<

// >>>>>>>>>>>>> Fundamental algebra >>>>>>>>>>>>>

/// calculate a [`a`, `b`) vector with a step of `step`.
/// This function is similar to numpy.arange(a, b, step).
pub fn arange<T>(a: T, b: T, step: T) -> Vec<T>
where
    T: Float,
{
    let size = cast::usize(cast_t2u::<T, f32>((b - a) / step).floor()).unwrap();
    (0..size)
        .map(|i| a + cast_t2u::<usize, T>(i) * step)
        .collect()
}

/// create a mesh-grid vectors
pub fn mesh_grid<T>(x: &[T], y: &[T]) -> (Vec<Vec<T>>, Vec<Vec<T>>)
where
    T: Float,
{
    let xx: Vec<Vec<T>> = (0..y.len()).map(|_| x.to_vec()).collect();
    let yy: Vec<Vec<T>> = (0..y.len()).map(|ii| vec![y[ii]; x.len()]).collect();
    (xx, yy)
}

/// fill a 1D vector with random values
fn rand_fill_1d<T>(x: &mut Vec<T>, low: T, high: T)
where
    T: Float,
{
    let mut rng = rand::thread_rng();
    let gen = Uniform::new(cast_t2u::<T, f64>(low), cast_t2u::<T, f64>(high));
    for ii in 0..x.len() {
        x[ii] = cast_t2u::<f64, T>(gen.sample(&mut rng));
    }
}

/// fill a 2D vector with random values
fn rand_fill_2d<T>(x: &mut Vec2d<T>, low: T, high: T)
where
    T: Float,
{
    for ii in 0..x.len() {
        rand_fill_1d(&mut x[ii], low, high);
    }
}

/// generate a 1D vector with random values
pub fn rand_uniform_1d<T>(size: usize, low: T, high: T) -> Vec<T>
where
    T: Float,
{
    let mut dst: Vec<T> = vec![cast_t2u(0.0); size];
    rand_fill_1d(&mut dst, low, high);
    dst
}

/// generate a 1D vector with random values
pub fn rand_uniform_2d<T>(h: usize, w: usize, low: T, high: T) -> Vec2d<T>
where
    T: Float,
{
    let mut dst: Vec2d<T> = vec![vec![cast_t2u(0.0); w]; h];
    rand_fill_2d(&mut dst, low, high);
    dst
}

/// ReLU function
pub fn relu<T>(x: T) -> T
where
    T: Float,
{
    let zero: T = cast_t2u(0.0);
    T::max(zero, x)
}

/// sigmoid function
pub fn sigmoid<T>(x: T) -> T
where
    T: Float,
{
    let one: T = cast_t2u(1.0);
    one / (one + T::exp(-x))
}

/// step function
pub fn step<T>(x: T) -> T
where
    T: Float,
{
    let zero: T = cast_t2u(0.0);
    let one: T = cast_t2u(1.0);
    if x <= zero {
        zero
    } else {
        one
    }
}

/// equality within tolerance
pub fn equality_with_tolerance<T>(x: T, y: T, tol: T) -> bool
where
    T: Float,
{
    if (x - y).abs() <= tol {
        true
    } else {
        false
    }
}

pub fn is_any_tol_1d<T>(x: &Vec<T>, y: &Vec<T>, tol: T) -> bool
where
    T: Float,
{
    for ii in 0..x.len() {
        if equality_with_tolerance(x[ii], y[ii], tol) {
            return true;
        }
    }
    false
}

pub fn is_all_tol_1d<T>(x: &Vec<T>, y: &Vec<T>, tol: T) -> bool
where
    T: Float,
{
    for ii in 0..x.len() {
        if !equality_with_tolerance(x[ii], y[ii], tol) {
            return false;
        }
    }
    true
}

/// max function
pub fn max<T>(x: &[T]) -> T
where
    T: Float,
{
    let zero: T = cast_t2u(0.0);
    x.into_iter().fold(zero / zero, |m, &v| v.max(m))
}

/// min function
pub fn min<T>(x: &[T]) -> T
where
    T: Float,
{
    let zero: T = cast_t2u(0.0);
    x.into_iter().fold(zero / zero, |m, &v| v.min(m))
}

// <<<<<<<<<<<<< Fundamental algebra <<<<<<<<<<<<<
