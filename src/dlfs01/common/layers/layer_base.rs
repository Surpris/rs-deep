//! layer_base.rs
//!
//! base traits for layers

#![allow(unused_variables)]

use super::super::util::CrateFloat;

/// arbitrary-D layer trait
pub trait LayerBase<T: CrateFloat> {
    type A;
    type B;
    fn forward(&mut self, x: &Self::A) -> Self::B;
    fn backward(&mut self, dx: &Self::B) -> Self::A;
    fn update(&mut self, lr: T) {
        return;
    }
    fn set_trainable(&mut self, _flag: bool) {
        return;
    }
    fn print_detail(&self) {
        return;
    }
    fn print_parameters(&self) {
        return;
    }
}

/// Arbitrary-D loss layer trait
pub trait LossLayerBase<T: CrateFloat> {
    type A;
    fn forward(&mut self, x: &Self::A, t: &Self::A) -> T;
    fn backward(&mut self, _dx: T) -> Self::A;
    fn update(&mut self, lr: T) {
        return;
    }
    fn print_detail(&self) {
        return;
    }
    fn print_parameters(&self) {
        return;
    }
    fn get_output(&self) -> Self::A;
}
