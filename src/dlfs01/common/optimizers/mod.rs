//! optimizers
//!
//! optimizers

pub mod optimizer;
pub mod optimizer_base;

use ndarray::{Array, Dimension, ShapeBuilder};
use num_traits::Float;
pub use optimizer::*;
pub use optimizer_base::*;
use std::fmt::Display;

#[derive(Clone)]
pub enum OptimizerEnum {
    SGD = 0,
    Momentum = 1,
    Nesterov = 2,
    AdaGrad = 3,
    RMSprop = 4,
    AdaDelta = 5,
    Adam = 6,
    RMSpropGraves = 7,
    SMORMS3 = 8,
    AdaMax = 9,
    Nadam = 10,
    Eve = 11,
    Santa = 12,
    GDByGD = 13,
    AdaSecant = 14,
    AMSGrad = 15,
    AdaBound = 16,
    AMSBound = 17,
    AdaBelief = 18,
}

impl Display for OptimizerEnum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizerEnum::SGD => write!(f, "SGD"),
            OptimizerEnum::Momentum => write!(f, "Momentum"),
            OptimizerEnum::Nesterov => write!(f, "Nesterov"),
            OptimizerEnum::AdaGrad => write!(f, "AdaGrad"),
            OptimizerEnum::RMSprop => write!(f, "RMSprop"),
            OptimizerEnum::AdaDelta => write!(f, "AdaDelta"),
            OptimizerEnum::Adam => write!(f, "Adam"),
            OptimizerEnum::RMSpropGraves => write!(f, "RMSpropGraves"),
            OptimizerEnum::SMORMS3 => write!(f, "SMORMS3"),
            OptimizerEnum::AdaMax => write!(f, "AdaMax"),
            OptimizerEnum::Nadam => write!(f, "Nadam"),
            OptimizerEnum::Eve => write!(f, "Eve"),
            OptimizerEnum::Santa => write!(f, "Santa"),
            OptimizerEnum::GDByGD => write!(f, "GDByGD"),
            OptimizerEnum::AdaSecant => write!(f, "AdaSecant"),
            OptimizerEnum::AMSGrad => write!(f, "AMSGrad"),
            OptimizerEnum::AdaBound => write!(f, "AdaBound"),
            OptimizerEnum::AMSBound => write!(f, "AMSBound"),
            OptimizerEnum::AdaBelief => write!(f, "AdaBelief"),
        }
    }
}

pub fn call_optimizer<T: 'static, D: 'static, Sh>(
    name: OptimizerEnum,
    shape: Sh,
    params: &[T],
) -> Box<dyn OptimizerBase<Src = Array<T, D>>>
where
    T: Float,
    D: Dimension,
    Sh: ShapeBuilder<Dim = D>,
{
    match name {
        OptimizerEnum::SGD => return Box::new(SGD::new(params[0])),
        OptimizerEnum::Momentum => return Box::new(Momentum::new(params[0], params[1], shape)),
        OptimizerEnum::Nesterov => return Box::new(Nesterov::new(params[0], params[1], shape)),
        OptimizerEnum::AdaGrad => return Box::new(AdaGrad::new(params[0], shape)),
        OptimizerEnum::RMSprop => return Box::new(RMSprop::new(params[0], params[1], shape)),
        // OptimizerEnum::AdaDelta => return Box::new(AdaDelta::new(params[0], params[1], shape)),
        // OptimizerEnum::Adam => return Box::new(Adam::new(params[0], params[1], shape)),
        // OptimizerEnum::RMSpropGraves => return Box::new(RMSpropGraves::new(params[0], params[1], shape)),
        // OptimizerEnum::SMORMS3 => return Box::new(SMORMS3::new(params[0], params[1], shape)),
        // OptimizerEnum::AdaMax => return Box::new(AdaMax::new(params[0], params[1], shape)),
        // OptimizerEnum::Nadam => return Box::new(Nadam::new(params[0], params[1], shape)),
        // OptimizerEnum::Eve => return Box::new(Eve::new(params[0], params[1], shape)),
        // OptimizerEnum::Santa => return Box::new(Santa::new(params[0], params[1], shape)),
        // OptimizerEnum::GDByGD => return Box::new(GDByGD::new(params[0], params[1], shape)),
        // OptimizerEnum::AdaSecant => return Box::new(AdaSecant::new(params[0], params[1], shape)),
        // OptimizerEnum::AMSGrad => return Box::new(AMSGrad::new(params[0], params[1], shape)),
        // OptimizerEnum::AdaBound => return Box::new(AdaBound::new(params[0], params[1], shape)),
        // OptimizerEnum::AMSBound => return Box::new(AMSBound::new(params[0], params[1], shape)),
        // OptimizerEnum::AdaBelief => return Box::new(AdaBelief::new(params[0], params[1], shape)),
        _ => panic!("Invalid optimizer name: {}", name),
    }
}
