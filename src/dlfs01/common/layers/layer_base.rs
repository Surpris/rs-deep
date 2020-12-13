//! layer_base.rs
//!
//! base traits for layers

#![allow(unused_variables)]

// use ndarray::prelude::*;

/// arbitrary-D layer trait
pub trait LayerBase<T> {
    type A;
    type B;
    fn forward(&mut self, x: &Self::A) -> Self::B;
    fn backward(&mut self, dx: &Self::B) -> Self::A;
    fn update(&mut self, lr: T) {
        return;
    }
    fn print_detail(&self);
}

/// Arbitrary-D loss layer trait
pub trait LossLayerBase<T> {
    type A;
    fn forward(&mut self, x: &Self::A, t: &Self::A) -> T;
    fn backward(&mut self, _dx: T) -> Self::A;
    fn update(&mut self, lr: T) {
        return;
    }
    fn print_detail(&self);
}
