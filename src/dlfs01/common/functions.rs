//! functions
//!
//! functions used for neural network

use super::util::cast_t2u;
use num_traits::Float;

/// identity function
pub fn identity<T>(x: &[T]) -> Vec<T>
where
    T: Float,
{
    x.to_vec()
}

/// ReLU function
pub fn relu<T>(x: &[T]) -> Vec<T>
where
    T: Float,
{
    let zero: T = cast_t2u(0.0);
    x.iter().map(|&v| T::max(zero, v)).collect()
}

/// gradient of ReLU function
pub fn relu_grad<T>(x: &[T]) -> Vec<T>
where
    T: Float,
{
    let zero: T = cast_t2u(0.0);
    let one: T = cast_t2u(1.0);
    x.iter()
        .map(|&v| if v < zero { zero } else { one })
        .collect()
}

/// sigmoid function
pub fn sigmoid<T>(x: &[T]) -> Vec<T>
where
    T: Float,
{
    let one: T = cast_t2u(1.0);
    x.iter().map(|&v| one / (one + T::exp(v))).collect()
}

/// gradient of sigmoid function
pub fn sigmoid_grad<T>(x: &[T]) -> Vec<T>
where
    T: Float,
{
    let one: T = cast_t2u(1.0);
    let vec: Vec<T> = sigmoid(x);
    (0..x.len()).map(|ii| (one - vec[ii]) * vec[ii]).collect()
}

/// step function
pub fn step<T>(x: &[T]) -> Vec<T>
where
    T: Float,
{
    let zero: T = cast_t2u(0.0);
    let one: T = cast_t2u(1.0);
    x.iter()
        .map(|&v| if v <= zero { zero } else { one })
        .collect()
}

/// softmax function
pub fn softmax<T>(x: &[T]) -> Vec<T>
where
    T: Float,
{
    let x_max: T = max(x);
    let x2: Vec<T> = x.iter().map(|&w| T::exp(w - x_max)).collect();
    let x2_sum: T = sum(&x2);
    x2.iter().map(|&v| v / x2_sum).collect()
}

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
