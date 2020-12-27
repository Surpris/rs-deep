//! utils
//!
//! utility functions

use ndarray_rand::rand_distr::uniform::SampleUniform;
use num_traits::{Float, Num, NumCast};
use std::fmt::{Debug, Display};
pub trait CrateFloat: Float + SampleUniform + Debug + Display {}
impl<T> CrateFloat for T where T: Float + SampleUniform + Debug + Display {}

/// cast a numeric value with type T to one with U
pub fn cast_t2u<T, U>(x: T) -> U
where
    T: Num + NumCast + Copy,
    U: Num + NumCast + Copy,
{
    U::from(x).unwrap()
}
