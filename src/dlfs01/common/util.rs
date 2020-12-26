//! utils
//!
//! utility functions

use num_traits::{Float, Num, NumCast};
use std::fmt::{Debug, Display};
pub trait CrateFloat: Float + Debug + Display {}
impl<T: Float + Debug + Display> CrateFloat for T {}

/// cast a numeric value with type T to one with U
pub fn cast_t2u<T, U>(x: T) -> U
where
    T: Num + NumCast + Copy,
    U: Num + NumCast + Copy,
{
    U::from(x).unwrap()
}
