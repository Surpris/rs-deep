//! math
//!
//! mathematical functions

use super::util::cast_t2u;
use num_traits::Float;

/// max function
pub fn max<T>(x: &[T]) -> T
where
    T: Float,
{
    let zero: T = cast_t2u(0.0);
    x.into_iter().fold(zero / zero, |m, &v| v.max(m))
}

/// min function
pub fn min<T>(x: &[T]) -> T
where
    T: Float,
{
    let zero: T = cast_t2u(0.0);
    x.into_iter().fold(zero / zero, |m, &v| v.min(m))
}

/// sum function
pub fn sum<T>(x: &[T]) -> T
where
    T: Float,
{
    let zero: T = cast_t2u(0.0);
    x.iter().fold(zero, |m, &v| m + v)
}
