//! gradient
//!
//! gradient operator

use super::math::MathFunc;
use super::util::cast_t2u;
use num_traits::Float;
const EPS: f64 = 1E-4;

pub fn numerical_gradient_1d<T, F: Fn(&Vec<T>) -> T>(f: &F, x: &mut Vec<T>) -> Vec<T>
where
    T: Float,
{
    let eps: T = cast_t2u(EPS);
    let eps2: T = cast_t2u(2.0 * EPS);
    let mut grad = x.clone().zeros_like();
    for ii in 0..x.len() {
        x[ii] = x[ii] + eps;
        let fxh1 = f(x);
        x[ii] = x[ii] - eps2;
        let fxh2 = f(x);
        grad[ii] = (fxh1 - fxh2) / eps2;
        x[ii] = x[ii] + eps;
    }
    grad
}
