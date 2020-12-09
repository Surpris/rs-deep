//! optimizer_base
//!
//! base traits for optimizers

pub trait OptimizerBase<T, L> {
    // for (&k, &v) in layer.params.get_mut() {
    //    *v = *v - lr * layer.grads.get_key(&k);
    //}
    fn update(&mut self, layer: L);
}
