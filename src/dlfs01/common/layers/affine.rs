//! affine
//! 
//! Affine layer

use super::super::util::cast_t2u;
use ndarray::{Array1, Array2, Axis};
use num_traits::Float;
use rand::distributions::Uniform;
use rand::prelude::*;

/// Affine layer trait
pub trait AffineBase<T> {
    fn new(shape: &(usize, usize)) -> Self;
    fn forward(&mut self, x: &Array2<T>) -> Array2<T>;
    fn backward(&mut self, dx: &Array2<T>) -> Array2<T>;
    fn print_detail(&self);
}

/// Affine layer
pub struct Affine<T> {
    weight: Array2<T>,
    bias: Array1<T>,
    dw: Array2<T>,
    db: Array1<T>,
    buff: Array2<T>,
}

impl<T: 'static> AffineBase<T> for Affine<T>
where
    T: Float,
{
    fn new(shape: &(usize, usize)) -> Self {
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
    fn forward(&mut self, x: &Array2<T>) -> Array2<T> {
        let mut dst: Array2<T> = x.dot(&self.weight);
        self.buff = x.clone();
        for v in dst.indexed_iter_mut() {
            *v.1 = *v.1 + self.bias[v.0.1];
        }
        dst
    }
    fn backward(&mut self, dx: &Array2<T>) -> Array2<T> {
        let dst = dx.dot(&self.weight.t());
        self.dw = self.buff.t().dot(&dst);
        self.db = dst.sum_axis(Axis(0));
        dst
    }
    fn print_detail(&self) {
        println!("affine layer.");
        println!("weight shape: {:?}", self.dw.shape());
        println!("bias shape: {:?}", self.bias.shape());
    }
}

pub fn main() {
    println!("< activation sub module> ");
    let mut rng = rand::thread_rng();
    let gen = Uniform::new(-1.0f32, 1.0f32);

    let input_shape: (usize, usize) = (10, 3);
    let a = Array2::<f32>::zeros(input_shape);
    let a = a.map(|_| gen.sample(&mut rng));

    let affine_shape = (input_shape.1, 2);
    let mut layer_affine = Affine::<f32>::new(&affine_shape);
    layer_affine.print_detail();

    let b = layer_affine.forward(&a);

    let output_shape = (input_shape.0, affine_shape.1);
    let da = Array2::<f32>::zeros(output_shape).map(|_| gen.sample(&mut rng));
    println!("a: {}", a);
    println!("affine forward: {}", b);
    println!("da: {}", da);
    println!("affine backward: {}", layer_affine.backward(&da));
}