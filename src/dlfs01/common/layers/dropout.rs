//! dropout
//!
//! Dropout layer

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::fmt::Display;

use crate::prelude::cast_t2u;

use super::layer_base::LayerBase;
use ndarray::prelude::*;
use ndarray_rand::rand::{distributions::Distribution, thread_rng};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use num_traits::Float;

/// Dropout
/// 
/// See http://arxiv.org/abs/1207.0580 in detail
pub struct DropOut<T, D> {
    ratio: T,
    mask: Array<u8, D>,
    trained: bool,
    one: T,
    zero: T,
    one_minus_ratio: T,
}

impl<T, D> DropOut<T, D>
where
    T: Float,
    D: Dimension,
{
    pub fn new<Sh>(ratio: T, shape: Sh) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        let one: T = cast_t2u(1.0);
        let zero: T = cast_t2u(0.0);
        assert!(ratio >= zero && ratio <= one);
        Self {
            ratio,
            mask: Array::<u8, D>::zeros(shape),
            trained: false,
            one,
            zero,
            one_minus_ratio: one - ratio,
        }
    }
}

impl<T, D> LayerBase<T> for DropOut<T, D>
where
    T: Float + Display,
    D: Dimension,
{
    type A = Array<T, D>;
    type B = Array<T, D>;
    fn forward(&mut self, x: &Self::A) -> Self::B {
        // if self.param.shape() != x.shape() {
        //     self.param = Array::<T, D>::zeros(x.raw_dim());
        // }
        if !self.trained {
            self.mask = Array::<f64, D>::random(x.raw_dim(), Uniform::new(0.0, 1.0)).map(|&x| {
                if x < 0.5 {
                    0
                } else {
                    1
                }
            });
            let mut dst = x.clone();
            for (v, d) in self.mask.iter().zip(dst.iter_mut()) {
                if *v != 0u8 {
                    *d = self.zero;
                }
            }
            dst
        } else {
            x.map(|&v| v * self.one_minus_ratio)
        }
    }
    fn backward(&mut self, dx: &Self::B) -> Self::A {
        let mut dst = dx.clone();
        for (v, d) in self.mask.iter().zip(dst.iter_mut()) {
            if *v != 0u8 {
                *d = self.zero;
            }
        }
        dst
    }
    fn update(&mut self, lr: T) {
        return;
    }
    fn print_detail(&self) {
        println!("Dropout layer with ratio of {}", self.ratio);
    }
}
