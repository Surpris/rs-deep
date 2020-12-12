//! optimizer
//!
//! optimizer struct

use super::optimizer_base::OptimizerBase;
use ndarray::prelude::*;
use num_traits::Float;

/// stochastic gradient descent
pub struct SGD<T> {
    lr: T,
}

impl<T> SGD<T>
where
    T: Float,
{
    pub fn new(lr: T) -> Self {
        Self { lr }
    }
}

impl<T, D> OptimizerBase<T, D> for SGD<T>
where
    T: Float,
    D: Dimension,
{
    fn update(&mut self, param: &mut Array<T, D>, grads: &Array<T, D>) {
        for (v, g) in param.iter_mut().zip(grads.iter()) {
            *v = *v - self.lr * *g;
        }
    }
}
