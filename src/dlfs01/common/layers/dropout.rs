//! dropout
//!
//! Dropout layer

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use super::super::param_initializers::*;
use super::super::util::*;
use super::layer_base::LayerBase;
use ndarray::prelude::*;
use ndarray_rand::rand::{distributions::Distribution, thread_rng};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::fmt::{Debug, Display};

/// Enum for Dropout layer
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum UseDropoutEnum<T: CrateFloat> {
    Use(T),
    None,
}

impl<T> Display for UseDropoutEnum<T>
where
    T: CrateFloat,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UseDropoutEnum::Use(ratio) => write!(f, "Use (ratio: {})", ratio),
            UseDropoutEnum::None => write!(f, "None"),
        }
    }
}

/// validate UseDropoutEnum and generate an appropriate layer
pub fn call_dropout_layer<T: 'static, D: 'static, Sh>(
    use_dropout_enum: UseDropoutEnum<T>,
    shape: Sh,
) -> DropOut<T, D>
where
    T: CrateFloat,
    D: Dimension,
    Sh: ShapeBuilder<Dim = D>,
{
    if let UseDropoutEnum::Use(ratio) = use_dropout_enum {
        DropOut::new(ratio, shape)
    } else {
        DropOut::new(cast_t2u(0.0), shape)
    }
}

/// Dropout
///
/// See http://arxiv.org/abs/1207.0580 in detail
pub struct DropOut<T: CrateFloat, D> {
    ratio: T,
    mask: Array<u8, D>,
    trainable: bool,
    one: T,
    zero: T,
    ratio_f64: f64,
    one_minus_ratio: T,
}

impl<T, D> DropOut<T, D>
where
    T: CrateFloat,
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
            trainable: true,
            one,
            zero,
            ratio_f64: cast_t2u(ratio),
            one_minus_ratio: one - ratio,
        }
    }
}

impl<T, D> LayerBase<T> for DropOut<T, D>
where
    T: CrateFloat,
    D: Dimension,
{
    type A = Array<T, D>;
    type B = Array<T, D>;
    fn forward(&mut self, x: &Self::A) -> Self::B {
        if self.trainable {
            self.mask = Array::<f64, D>::random(x.raw_dim(), Uniform::new(0.0, 1.0)).map(|&x| {
                if x < self.ratio_f64 {
                    0
                } else {
                    1
                }
            });
            let mut dst = x.clone();
            for (v, d) in self.mask.iter().zip(dst.iter_mut()) {
                if *v == 0u8 {
                    *d = self.zero;
                }
            }
            dst
        } else {
            x * self.one_minus_ratio
        }
    }
    fn backward(&mut self, dx: &Self::B) -> Self::A {
        let mut dst = dx.clone();
        for (v, d) in self.mask.iter().zip(dst.iter_mut()) {
            if *v == 0u8 {
                *d = self.zero;
            }
        }
        dst
    }
    fn update(&mut self, lr: T) {
        return;
    }
    fn set_trainable(&mut self, flag: bool) {
        self.trainable = flag;
    }
    fn print_detail(&self) {
        println!("Dropout layer.");
        println!("ratio: {}", self.ratio);
    }
}
