//! utils
//!
//! utility functions

use num_traits::{Num, NumCast};

/// cast a numeric value with type T to one with U
pub fn cast_t2u<T, U>(x: T) -> U
where
    T: Num + NumCast + Copy,
    U: Num + NumCast + Copy,
{
    U::from(x).unwrap()
}
