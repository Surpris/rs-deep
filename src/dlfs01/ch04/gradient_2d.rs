//! gradient_2d
//!
//! 2D gradient test

use super::super::common::math::*;
use super::super::common::operators::Operators;
use super::super::common::util::cast_t2u;
use num_traits::Float;
// use plotters::prelude::*;
const EPS: f64 = 1E-4;

type Vec2d<T> = Vec<Vec<T>>;

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

/// 1-directional gradient
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

fn function_1<T>(x: &Vec<T>) -> T
where
    T: Float,
{
    x.sum()
}

#[allow(dead_code)]
fn function_2<T>(x: &Vec2d<T>) -> T
where
    T: Float,
{
    let v: Vec<T> = x.iter().map(|v| v.clone().sum()).collect();
    v.sum()
}

pub fn main() {
    println!("< gradient_2d sub module >");
    let x: Vec<f32> = arange(-2.0, 2.5, 0.25);
    let (xx, yy) = mesh_grid(&x, &x);
    let mut target: Vec<Vec<f32>> = Vec::new();
    target.push(xx.clone().flatten());
    target.push(yy.clone().flatten());
    let mut target = target.transpose();
    let _grad = numerical_gradient_2d(&function_1, &mut target).transpose();
}
