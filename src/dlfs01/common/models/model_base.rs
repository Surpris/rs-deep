//! model_base
//!
//! base traits for models

#![allow(unused_variables)]

use ndarray::prelude::*;

/// Arbitrary-D model trait
pub trait ModelBase<T, D1, D2> {
    fn predict_prob(&mut self, x: &Array<T, D1>) -> Array<T, D2>;
    fn predict(&mut self, x: &Array<T, D1>) -> Array<T, D2>;
    fn loss(&mut self, x: &Array<T, D1>, t: &Array<T, D2>) -> T;
    fn accuracy(&mut self, x: &Array<T, D1>, t: &Array<T, D2>) -> T;
    fn gradient(&mut self, x: &Array<T, D1>, t: &Array<T, D2>);
    fn update(&mut self, x: &Array<T, D1>, t: &Array<T, D2>, lr: T);
    fn print_detail(&self);
}
