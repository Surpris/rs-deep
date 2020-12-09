//! sgd
//!
//! stochastic gradient descent

use super::super::models::*;
use super::optimizer_base::OptimizerBase;
use num_traits::Float;

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

// impl<T, M> OptimizerBase<T, M> for SGD <T>
// where T: Float, M: ModelBase2<T>
// {
//     fn update(&mut self, model: M){
//         model.update(self.lr);
//     }
// }
