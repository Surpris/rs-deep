//! gradient
//!
//! gradient operator

// use super::math::MathFunc;
use super::operators::Operators;
use super::util::cast_t2u;
use num_traits::Float;
const EPS: f64 = 1E-4;

type Vec2d<T> = Vec<Vec<T>>;

/// numerical calculation of the gradient of a given function
/// with a 1D vector
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

/// 1-directional numerical calculation of the gradient of a given function
/// and a 2D vector
///
/// The 2nd input of this function is assumed to be a Vec<Vec<T>> created by
/// horizontally stacking flattened arrays created by mesh_grid().
pub fn numerical_gradient_2d<T, F: Fn(&Vec<T>) -> T>(f: &F, x: &mut Vec2d<T>) -> Vec2d<T>
where
    T: Float,
{
    let mut grad: Vec2d<T> = Vec2d::new();
    for v in x {
        grad.push(numerical_gradient_1d(f, &mut v.clone()));
    }
    grad
}
