//! model_base
//!
//! base traits for models

#![allow(unused_variables)]

// use ndarray::prelude::*;

/// Arbitrary-D model trait
pub trait ModelBase<T> {
    type A;
    type B;
    fn predict_prob(&mut self, x: &Self::A) -> Self::B;
    fn predict(&mut self, x: &Self::A) -> Self::B;
    fn loss(&mut self, x: &Self::A, t: &Self::B) -> T;
    fn accuracy(&mut self, x: &Self::A, t: &Self::B) -> T;
    fn gradient(&mut self, x: &Self::A, t: &Self::B);
    fn update(&mut self, x: &Self::A, t: &Self::B);
    fn print_detail(&self);
}
