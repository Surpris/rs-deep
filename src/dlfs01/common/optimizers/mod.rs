//! optimizers
//!
//! optimizers

pub mod optimizer;
pub mod optimizer_base;

use ndarray::{Array, Dimension};
use num_traits::Float;
pub use optimizer::*;
pub use optimizer_base::*;

pub enum OptimizerEnum<T, D> {
    SGD(SGD<T, D>),
    Momentum(Momentum<T, D>),
    Nesterov(Nesterov<T, D>),
    AdaGrad(AdaGrad<T, D>),
    RMSprop(RMSprop<T, D>),
    AdaDelta(AdaDelta<T, D>),
    Adam(Adam<T, D>),
    RMSpropGraves(RMSpropGraves<T, D>),
    SMORMS3(SMORMS3<T, D>),
    AdaMax(AdaMax<T, D>),
    Nadam(Nadam<T, D>),
    Eve(Eve<T, D>),
    Santa(Santa<T, D>),
    GDByGD(GDByGD<T, D>),
    AdaSecant(AdaSecant<T, D>),
    AMSGrad(AMSGrad<T, D>),
    AdaBound(AdaBound<T, D>),
    AMSBound(AMSBound<T, D>),
    AdaBelief(AdaBelief<T, D>),
}

impl<T, D> OptimizerBase for OptimizerEnum<T, D>
where
    T: Float,
    D: Dimension,
{
    type Src = Array<T, D>;
    fn update(&mut self, param: &mut Self::Src, grads: &Self::Src) {
        match self {
            Self::SGD(x) => x.update(param, grads),
            Self::Momentum(x) => x.update(param, grads),
            Self::Nesterov(x) => x.update(param, grads),
            Self::AdaGrad(x) => x.update(param, grads),
            Self::RMSprop(x) => x.update(param, grads),
            Self::Adam(x) => x.update(param, grads),
            _ => panic!("Not implemented error."),
        }
    }
}
