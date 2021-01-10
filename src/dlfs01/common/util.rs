//! utils
//!
//! utility functions

use ndarray::ScalarOperand;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num_traits::{Float, FromPrimitive, Num, NumCast};
use serde::Serialize;
use std::fmt::{Debug, Display};
use std::ops;

/// Float trait for this crate
pub trait CrateFloat:
    Float
    + ops::Add<Output = Self>
    + ops::Sub<Output = Self>
    + ops::Mul<Output = Self>
    + ops::Div<Output = Self>
    + ops::AddAssign
    + ops::SubAssign
    + ops::MulAssign
    + ops::DivAssign
    + FromPrimitive
    + SampleUniform
    + ScalarOperand
    + Debug
    + Display
    + Serialize // + Deserialize<'static>
{
}
impl<T> CrateFloat for T where
    T: Float
        + ops::Add<Output = Self>
        + ops::Sub<Output = Self>
        + ops::Mul<Output = Self>
        + ops::Div<Output = Self>
        + ops::AddAssign
        + ops::SubAssign
        + ops::MulAssign
        + ops::DivAssign
        + FromPrimitive
        + SampleUniform
        + ScalarOperand
        + Debug
        + Display
        + Serialize // + Deserialize<'static>
{
}

/// cast a numeric value with type T to one with U
pub fn cast_t2u<T, U>(x: T) -> U
where
    T: Num + NumCast + Copy,
    U: Num + NumCast + Copy,
{
    U::from(x).unwrap()
}
