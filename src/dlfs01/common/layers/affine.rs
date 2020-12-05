//! affine
//!
//! Affine layer

use super::super::util::cast_t2u;
use ndarray::{Array, Array1, Array2, Axis};
use num_traits::Float;
use rand::distributions::Uniform;
use rand::prelude::*;

/// Affine layer trait
pub trait AffineBase<T> {
    // fn new(shape: &(usize, usize)) -> Self;
    fn forward(&mut self, x: &Array2<T>) -> Array2<T>;
    fn backward(&mut self, dx: &Array2<T>) -> Array2<T>;
    fn update(&mut self, lr: T);
    fn print_detail(&self);
}

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
    pub fn new(shape: &(usize, usize)) -> Self {
        let mut rng = rand::thread_rng();
        let gen = Uniform::new(-1.0f32, 1.0f32);
        Affine {
            weight: Array2::<T>::ones(*shape).map(|_| cast_t2u(gen.sample(&mut rng))),
            bias: Array1::<T>::ones(shape.1).map(|_| cast_t2u(gen.sample(&mut rng))),
            dw: Array2::<T>::ones(*shape),
            db: Array1::<T>::ones(shape.1),
            buff: Array2::<T>::ones(*shape),
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

impl<T: 'static> AffineBase<T> for Affine<T>
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

pub fn main() {
    println!("< affine sub module> ");
    let mut rng = rand::thread_rng();
    let gen = Uniform::new(-1.0f32, 1.0f32);

    let input_shape: (usize, usize) = (2, 3);
    let a: Array2<f32> =
        Array::from_shape_vec(input_shape, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]).unwrap();

    let affine_shape = (input_shape.1, 2);
    let w: Array2<f32> =
        Array::from_shape_vec(affine_shape, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b: Array1<f32> = Array::linspace(1.0, 3.0, affine_shape.1);
    let mut layer = Affine::<f32>::from(&w, &b);
    layer.print_detail();

    let y = layer.forward(&a);

    let output_shape = (input_shape.0, affine_shape.1);
    let da = Array2::<f32>::zeros(output_shape).map(|_| gen.sample(&mut rng));
    println!("a: {}", a);
    println!("affine forward: {}", y);
    println!("da: {}", da);
    println!("affine backward: {}", layer.backward(&da));
}
