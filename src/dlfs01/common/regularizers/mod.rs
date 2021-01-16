//! regularizers
//!
//! Regularization layers for weights and biases

use super::util::{cast_t2u, CrateFloat};
use ndarray::prelude::*;
use std::{fmt::Display, marker::PhantomData};

/// Enum of regularizer layers
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum RegularizerEnum<T: CrateFloat> {
    L1(T),
    L2(T),
    None,
}

impl<T> Display for RegularizerEnum<T>
where
    T: CrateFloat,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegularizerEnum::L1(x) => write!(f, "L1 (decay lambda: {})", x),
            RegularizerEnum::L2(x) => write!(f, "L2 (decay lambda: {})", x),
            RegularizerEnum::None => write!(f, "None"),
        }
    }
}

/// validate RegularizerEnum and generate an appropriate regularizer
pub fn call_regularizer<T: 'static, D: 'static>(
    regularizer_enum: RegularizerEnum<T>,
) -> Box<dyn RegularizerBase<T, A = Array<T, D>>>
where
    T: CrateFloat,
    D: Dimension,
{
    match regularizer_enum {
        RegularizerEnum::L1(decay_lambda) => Box::new(L1Norm::new(decay_lambda)),
        RegularizerEnum::L2(decay_lambda) => Box::new(L2Norm::new(decay_lambda)),
        RegularizerEnum::None => Box::new(L2Norm::new(cast_t2u(0.0))),
    }
}

/// Regularizer layer trait
pub trait RegularizerBase<T: CrateFloat> {
    type A;
    fn forward(&mut self, x: &Self::A) -> T;
    fn backward(&mut self, x: &Self::A) -> Self::A;
    fn print_detail(&self) {
        return;
    }
    fn print_parameters(&self) {
        return;
    }
}

/// L1-norm regularizer
#[derive(Clone, Debug)]
pub struct L1Norm<T: CrateFloat, D: Dimension> {
    decay_lambda: T,
    zero: T,
    _phantom: PhantomData<D>,
}

impl<T, D> L1Norm<T, D>
where
    T: CrateFloat,
    D: Dimension,
{
    pub fn new(decay_lambda: T) -> Self {
        Self {
            decay_lambda,
            zero: cast_t2u(0.0),
            _phantom: PhantomData,
        }
    }
}

impl<T, D> RegularizerBase<T> for L1Norm<T, D>
where
    T: CrateFloat,
    D: Dimension,
{
    type A = Array<T, D>;

    fn forward(&mut self, x: &Self::A) -> T {
        if self.decay_lambda == self.zero {
            self.zero
        } else {
            self.decay_lambda * x.map(|&v| v.abs()).sum()
        }
    }

    fn backward(&mut self, x: &Self::A) -> Self::A {
        if self.decay_lambda == self.zero {
            Self::A::zeros(x.raw_dim())
        } else {
            x.map(|&v| {
                if v > self.zero {
                    -self.decay_lambda
                } else if v < self.zero {
                    self.decay_lambda
                } else {
                    self.zero
                }
            })
        }
    }

    fn print_detail(&self) {
        println!("L1-norm regularizer.");
        println!("decay lambda: {}", self.decay_lambda);
    }

    fn print_parameters(&self) {
        println!("decay lambda: {}", self.decay_lambda);
    }
}

/// L2-norm regularizer
#[derive(Clone, Debug)]
pub struct L2Norm<T: CrateFloat, D: Dimension> {
    decay_lambda: T,
    zero: T,
    half: T,
    _phantom: PhantomData<D>,
}

impl<T, D> L2Norm<T, D>
where
    T: CrateFloat,
    D: Dimension,
{
    pub fn new(decay_lambda: T) -> Self {
        Self {
            decay_lambda,
            zero: cast_t2u(0.0),
            half: cast_t2u(0.5),
            _phantom: PhantomData,
        }
    }
}

impl<T, D> RegularizerBase<T> for L2Norm<T, D>
where
    T: CrateFloat,
    D: Dimension,
{
    type A = Array<T, D>;

    fn forward(&mut self, x: &Self::A) -> T {
        if self.decay_lambda == self.zero {
            self.zero
        } else {
            self.half * self.decay_lambda * x.map(|&v| v * v).sum()
        }
    }

    fn backward(&mut self, x: &Self::A) -> Self::A {
        if self.decay_lambda == self.zero {
            Self::A::zeros(x.raw_dim())
        } else {
            x.map(|&v| -self.decay_lambda * v)
        }
    }

    fn print_detail(&self) {
        println!("L2-norm regularizer.");
        println!("decay lambda: {}", self.decay_lambda);
    }

    fn print_parameters(&self) {
        println!("decay lambda: {}", self.decay_lambda);
    }
}
