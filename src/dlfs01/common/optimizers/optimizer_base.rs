//! optimizer_base
//!
//! base traits for optimizers

// use ndarray::prelude::*;

pub trait OptimizerBase {
    type Src;
    fn update(&mut self, param: &mut Self::Src, grads: &Self::Src);
}
