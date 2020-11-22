//! layer_base
//!
//! Basic layer traits

use ndarray::ArrayD;

/// Basic layer trait
pub trait LayerBase<T> {
    fn new(shape: &[usize]) -> Self;
    fn forward(&mut self, x: &ArrayD<T>) -> ArrayD<T>;
    fn backward(&self, dx: &ArrayD<T>) -> ArrayD<T>;
}
