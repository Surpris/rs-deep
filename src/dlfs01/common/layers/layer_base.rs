//! layer_base.rs
//!
//! base traits for layers

#![allow(unused_variables)]

use ndarray::{Array, Array2, Array3, Array4, Array5, Array6, ArrayD};

// >>>>>>>>>>>>> For basic layers >>>>>>>>>>>>>

/// arbitrary-D layer trait
pub trait LayerBase<T, D> {
    fn forward(&mut self, x: &Array<T, D>) -> Array<T, D>;
    fn backward(&mut self, dx: &Array<T, D>) -> Array<T, D>;
    fn update(&mut self, lr: T) {
        return;
    }
    fn print_detail(&self);
}

// <<<<<<<<<<<<< For basic layers <<<<<<<<<<<<<

// >>>>>>>>>>>>> For loss layers >>>>>>>>>>>>>

/// Arbitrary-D loss layer trait
pub trait LossLayerBase<T, D> {
    fn forward(&mut self, x: &Array<T, D>, t: &Array<T, D>) -> T;
    fn backward(&mut self, _dx: T) -> Array<T, D>;
    fn update(&mut self, lr: T) {
        return;
    }
    fn print_detail(&self);
}

/// 2D loss layer trait
pub trait LossLayerBase2<T> {
    fn forward(&mut self, x: &Array2<T>, t: &Array2<T>) -> T;
    fn backward(&mut self, _dx: T) -> Array2<T>;
    fn update(&mut self, lr: T) {
        return;
    }
    fn print_detail(&self);
}

/// 3D loss layer trait
pub trait LossLayerBase3<T> {
    fn forward(&mut self, x: &Array3<T>, t: &Array3<T>) -> T;
    fn backward(&mut self, _dx: T) -> Array3<T>;
    fn update(&mut self, lr: T) {
        return;
    }
    fn print_detail(&self);
}

/// 4D loss layer trait
pub trait LossLayerBase4<T> {
    fn forward(&mut self, x: &Array4<T>, t: &Array4<T>) -> T;
    fn backward(&mut self, _dx: T) -> Array4<T>;
    fn update(&mut self, lr: T) {
        return;
    }
    fn print_detail(&self);
}

/// 5D loss layer trait
pub trait LossLayerBase5<T> {
    fn forward(&mut self, x: &Array5<T>, t: &Array5<T>) -> T;
    fn backward(&mut self, _dx: T) -> Array5<T>;
    fn update(&mut self, lr: T) {
        return;
    }
    fn print_detail(&self);
}

/// 6D loss layer trait
pub trait LossLayerBase6<T> {
    fn forward(&mut self, x: &Array6<T>, t: &Array6<T>) -> T;
    fn backward(&mut self, _dx: T) -> Array6<T>;
    fn update(&mut self, lr: T) {
        return;
    }
    fn print_detail(&self);
}

/// Dynamic-D loss layer trait
pub trait LossLayerBaseD<T> {
    fn forward(&mut self, x: &ArrayD<T>, t: &ArrayD<T>) -> T;
    fn backward(&mut self, _dx: T) -> ArrayD<T>;
    fn update(&mut self, lr: T) {
        return;
    }
    fn print_detail(&self);
}

// <<<<<<<<<<<<< For loss layers <<<<<<<<<<<<<
