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
            RegularizerEnum::L1(_) => write!(f, "L1"),
            RegularizerEnum::L2(_) => write!(f, "L2"),
            RegularizerEnum::None => write!(f, "None"),
        }
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
        self.decay_lambda * x.map(|&v| v.abs()).sum()
    }

    fn backward(&mut self, x: &Self::A) -> Self::A {
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

/// L2-norm regularizer
#[derive(Clone, Debug)]
pub struct L2Norm<T: CrateFloat, D: Dimension> {
    decay_lambda: T,
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
        self.half * self.decay_lambda * x.map(|&v| v * v).sum()
    }

    fn backward(&mut self, x: &Self::A) -> Self::A {
        x.map(|&v| -self.decay_lambda * v)
    }
}
