//! loss_function
//!
//! functions for loss

use super::math::MathFunc;
use super::operators::Operators;
use super::util::cast_t2u;
use num_traits::Float;
use std::f64::consts::E;

type Vec2d<T> = Vec<Vec<T>>;

const EPS: f64 = 1E-8;

pub trait LossFunc<T>
where
    T: Float,
{
    /// sum of squared error
    fn sum_squared_error(&self, other: &Self) -> T;
    /// cross entropy error
    fn cross_entropy_error(&self, other: &Self) -> T;
    /// softmax loss
    fn softmax_loss(&self, other: &Self) -> T;
}

impl<T> LossFunc<T> for Vec<T>
where
    T: Float,
{
    fn sum_squared_error(&self, other: &Self) -> T {
        let half: T = cast_t2u(0.5);
        self.sub(other)
            .iter()
            .map(|&v| half * v * v)
            .collect::<Vec<T>>()
            .sum()
    }
    fn cross_entropy_error(&self, other: &Self) -> T {
        let e: T = cast_t2u(E);
        let eps: T = cast_t2u(EPS);
        -(0..self.len())
            .map(|ii| other[ii] * (self[ii] + eps).log(e))
            .collect::<Vec<T>>()
            .sum()
    }
    fn softmax_loss(&self, other: &Self) -> T {
        self.softmax().cross_entropy_error(other)
    }
}

impl<T> LossFunc<T> for Vec2d<T>
where
    T: Float,
{
    fn sum_squared_error(&self, other: &Self) -> T {
        (0..self.len())
            .map(|ii| self[ii].sum_squared_error(&other[ii]))
            .collect::<Vec<T>>()
            .sum()
    }
    fn cross_entropy_error(&self, other: &Self) -> T {
        -(0..self.len())
            .map(|ii| self[ii].cross_entropy_error(&other[ii]))
            .collect::<Vec<T>>()
            .sum()
    }
    fn softmax_loss(&self, other: &Self) -> T {
        (0..self.len())
            .map(|ii| self[ii].softmax().cross_entropy_error(&other[ii]))
            .collect::<Vec<T>>()
            .sum()
    }
}

pub fn main() {
    println!("< loss_function sub module >");
    let v: Vec<f32> = vec![1.0; 10];
    let v2: Vec<f32> = vec![0.5; 10];
    println!("{:?}", v.sum_squared_error(&v2));
}
