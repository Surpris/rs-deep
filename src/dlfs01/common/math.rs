//! math
//!
//! mathematical functions

use super::util::cast_t2u;
use num_traits::Float;

pub trait MathFunc<T> {
    /// identity function
    fn identity(self) -> Self;
    /// ReLU function
    fn relu(self) -> Self;
    /// gradient of ReLU
    fn relu_grad(self) -> Self;
    /// sigmoid function
    fn sigmoid(self) -> Self;
    /// gradient of sigmoid
    fn sigmoid_grad(self) -> Self;
    /// softmax function
    fn softmax(self) -> Self;
    /// step function
    fn step(self) -> Self;
    /// maximum function
    fn max(self) -> T;
    /// minimum function
    fn min(self) -> T;
    /// summation function
    fn sum(self) -> T;
}

impl<T> MathFunc<T> for Vec<T>
where
    T: Float,
{
    fn identity(self) -> Self {
        self
    }
    fn relu(self) -> Self {
        let zero: T = cast_t2u(0.0);
        self.iter().map(|&v| T::max(zero, v)).collect()
    }
    fn relu_grad(self) -> Self {
        let zero: T = cast_t2u(0.0);
        let one: T = cast_t2u(1.0);
        self.iter()
            .map(|&v| if v < zero { zero } else { one })
            .collect()
    }
    fn sigmoid(self) -> Self {
        let one: T = cast_t2u(1.0);
        self.iter().map(|&v| one / (one + T::exp(v))).collect()
    }
    fn sigmoid_grad(self) -> Self {
        let one: T = cast_t2u(1.0);
        let vec: Vec<T> = self.sigmoid();
        (0..vec.len()).map(|ii| (one - vec[ii]) * vec[ii]).collect()
    }
    fn softmax(self) -> Self {
        let x_max: T = self.clone().max();
        let x2: Vec<T> = self.iter().map(|&w| T::exp(w - x_max)).collect();
        let x2_sum: T = x2.clone().sum();
        x2.iter().map(|&v| v / x2_sum).collect()
    }
    fn step(self) -> Self {
        let zero: T = cast_t2u(0.0);
        let one: T = cast_t2u(1.0);
        self.iter()
            .map(|&v| if v <= zero { zero } else { one })
            .collect()
    }
    fn max(self) -> T {
        let zero: T = cast_t2u(0.0);
        self.iter().fold(zero / zero, |m, &v| v.max(m))
    }
    fn min(self) -> T {
        let zero: T = cast_t2u(0.0);
        self.iter().fold(zero / zero, |m, &v| v.min(m))
    }
    fn sum(self) -> T {
        let zero: T = cast_t2u(0.0);
        self.iter().fold(zero, |m, &v| m + v)
    }
}
