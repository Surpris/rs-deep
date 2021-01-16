//! optimizers
//!
//! optimizers

pub mod optimizer;
pub mod optimizer_base;

use super::util::CrateFloat;
use ndarray::{Array, Dimension, ShapeBuilder};
pub use optimizer::*;
pub use optimizer_base::*;
use std::fmt::Display;

/// Enum of optimizers
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum OptimizerEnum<T: CrateFloat> {
    SGD(T),
    Momentum(T, T),
    Nesterov(T, T),
    AdaGrad(T),
    RMSprop(T, T),
    AdaDelta(T),
    Adam(T, T, T),
    RMSpropGraves(T),
    SMORMS3(T),
    AdaMax(T),
    Nadam(T),
    Eve(T),
    Santa(T),
    GDByGD(T),
    AdaSecant(T),
    AMSGrad(T),
    AdaBound(T),
    AMSBound(T),
    AdaBelief(T),
}

impl<T> Display for OptimizerEnum<T>
where
    T: CrateFloat,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizerEnum::SGD(lr) => write!(f, "SGD (lr: {})", lr),
            OptimizerEnum::Momentum(lr, momentum) => {
                write!(f, "Momentum (lr: {}, momentum: {})", lr, momentum)
            }
            OptimizerEnum::Nesterov(lr, momentum) => {
                write!(f, "Nesterov (lr: {}, momentum: {})", lr, momentum)
            }
            OptimizerEnum::AdaGrad(lr) => write!(f, "AdaGrad (lr: {})", lr),
            OptimizerEnum::RMSprop(lr, decay_rate) => {
                write!(f, "RMSprop (lr: {}, decay_rate: {})", lr, decay_rate)
            }
            OptimizerEnum::AdaDelta(_) => write!(f, "AdaDelta"),
            OptimizerEnum::Adam(lr, beta1, beta2) => {
                write!(f, "Adam (lr: {}, beta1: {}, beta2: {})", lr, beta1, beta2)
            }
            OptimizerEnum::RMSpropGraves(_) => write!(f, "RMSpropGraves"),
            OptimizerEnum::SMORMS3(_) => write!(f, "SMORMS3"),
            OptimizerEnum::AdaMax(_) => write!(f, "AdaMax"),
            OptimizerEnum::Nadam(_) => write!(f, "Nadam"),
            OptimizerEnum::Eve(_) => write!(f, "Eve"),
            OptimizerEnum::Santa(_) => write!(f, "Santa"),
            OptimizerEnum::GDByGD(_) => write!(f, "GDByGD"),
            OptimizerEnum::AdaSecant(_) => write!(f, "AdaSecant"),
            OptimizerEnum::AMSGrad(_) => write!(f, "AMSGrad"),
            OptimizerEnum::AdaBound(_) => write!(f, "AdaBound"),
            OptimizerEnum::AMSBound(_) => write!(f, "AMSBound"),
            OptimizerEnum::AdaBelief(_) => write!(f, "AdaBelief"),
        }
    }
}

/// generate an optimizer
pub fn call_optimizer<T: 'static, D: 'static, Sh>(
    optimizer_enum: OptimizerEnum<T>,
    shape: Sh,
) -> Box<dyn OptimizerBase<Src = Array<T, D>>>
where
    T: CrateFloat,
    D: Dimension,
    Sh: ShapeBuilder<Dim = D>,
{
    match optimizer_enum {
        OptimizerEnum::SGD(lr) => return Box::new(SGD::new(lr)),
        OptimizerEnum::Momentum(lr, momentum) => {
            return Box::new(Momentum::new(lr, momentum, shape))
        }
        OptimizerEnum::Nesterov(lr, momentum) => {
            return Box::new(Nesterov::new(lr, momentum, shape))
        }
        OptimizerEnum::AdaGrad(lr) => return Box::new(AdaGrad::new(lr, shape)),
        OptimizerEnum::RMSprop(lr, decay_rate) => {
            return Box::new(RMSprop::new(lr, decay_rate, shape))
        }
        // OptimizerEnum::AdaDelta(lr) => return Box::new(AdaDelta::new(lr, shape)),
        OptimizerEnum::Adam(lr, beta1, beta2) => {
            return Box::new(Adam::new(lr, beta1, beta2, shape))
        }
        // OptimizerEnum::RMSpropGraves(lr) => return Box::new(RMSpropGraves::new(lr, shape)),
        // OptimizerEnum::SMORMS3(lr) => return Box::new(SMORMS3::new(lr, shape)),
        // OptimizerEnum::AdaMax(lr) => return Box::new(AdaMax::new(lr, shape)),
        // OptimizerEnum::Nadam(lr) => return Box::new(Nadam::new(lr, shape)),
        // OptimizerEnum::Eve(lr) => return Box::new(Eve::new(lr, shape)),
        // OptimizerEnum::Santa(lr) => return Box::new(Santa::new(lr, shape)),
        // OptimizerEnum::GDByGD(lr) => return Box::new(GDByGD::new(lr, shape)),
        // OptimizerEnum::AdaSecant(lr) => return Box::new(AdaSecant::new(lr, shape)),
        // OptimizerEnum::AMSGrad(lr) => return Box::new(AMSGrad::new(lr, shape)),
        // OptimizerEnum::AdaBound(lr) => return Box::new(AdaBound::new(lr, shape)),
        // OptimizerEnum::AMSBound(lr) => return Box::new(AMSBound::new(lr, shape)),
        // OptimizerEnum::AdaBelief(lr) => return Box::new(AdaBelief::new(lr, shape)),
        _ => panic!("Invalid optimizer name: {}", optimizer_enum),
    }
}
