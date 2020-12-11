//! layer_base.rs
//!
//! base traits for layers

#![allow(unused_variables)]

use ndarray::prelude::*;

/// arbitrary-D layer trait
pub trait LayerBase<T, D> {
    fn forward(&mut self, x: &Array<T, D>) -> Array<T, D>;
    fn backward(&mut self, dx: &Array<T, D>) -> Array<T, D>;
    fn update(&mut self, lr: T) {
        return;
    }
    fn print_detail(&self);
}

/// Arbitrary-D loss layer trait
pub trait LossLayerBase<T, D> {
    fn forward(&mut self, x: &Array<T, D>, t: &Array<T, D>) -> T;
    fn backward(&mut self, _dx: T) -> Array<T, D>;
    fn update(&mut self, lr: T) {
        return;
    }
    fn print_detail(&self);
}
