//! math
//!
//! mathematical functions

use super::util::cast_t2u;
use num_traits::Float;

trait MathFunc<T>
where
    T: Float,
{
    fn identity(self) -> Self;
    fn relu(self) -> Self;
    fn relu_grad(self) -> Self;
    fn sigmoid(self) -> Self;
    fn sigmoid_grad(self) -> Self;
    fn step(self) -> Self;
    fn max(self) -> T;
    fn min(self) -> T;
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
