//! optimizer_base
//!
//! base traits for optimizers

use ndarray::prelude::*;

pub trait OptimizerBase<T, D> {
    // for (&k, &v) in layer.params.get_mut() {
    //    *v = *v - lr * layer.grads.get_key(&k);
    //}
    fn update(&mut self, param: &mut Array<T, D>, grads: &Array<T, D>);
}
