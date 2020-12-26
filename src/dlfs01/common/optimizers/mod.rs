//! optimizers
//!
//! optimizers

pub mod optimizer;
pub mod optimizer_base;

use ndarray::{Array, Dimension, ShapeBuilder};
use num_traits::Float;
pub use optimizer::*;
pub use optimizer_base::*;

pub fn call_optimizer<T: 'static, D: 'static, Sh>(
    name: &str,
    shape: Sh,
    params: &[T],
) -> Box<dyn OptimizerBase<Src = Array<T, D>>>
where
    T: Float,
    D: Dimension,
    Sh: ShapeBuilder<Dim = D>,
{
    match name {
        "SGD" => return Box::new(SGD::new(params[0])),
        "Momentum" => return Box::new(Momentum::new(params[0], params[1], shape)),
        "Nesterov" => return Box::new(Nesterov::new(params[0], params[1], shape)),
        "AdaGrad" => return Box::new(AdaGrad::new(params[0], shape)),
        "RMSprop" => return Box::new(RMSprop::new(params[0], params[1], shape)),
        // "AdaDelta" => return Box::new(AdaDelta::new(params[0], params[1], shape)),
        // "Adam" => return Box::new(Adam::new(params[0], params[1], shape)),
        // "RMSpropGraves" => return Box::new(RMSpropGraves::new(params[0], params[1], shape)),
        // "SMORMS3" => return Box::new(SMORMS3::new(params[0], params[1], shape)),
        // "AdaMax" => return Box::new(AdaMax::new(params[0], params[1], shape)),
        // "Nadam" => return Box::new(Nadam::new(params[0], params[1], shape)),
        // "Eve" => return Box::new(Eve::new(params[0], params[1], shape)),
        // "Santa" => return Box::new(Santa::new(params[0], params[1], shape)),
        // "GDByGD" => return Box::new(GDByGD::new(params[0], params[1], shape)),
        // "AdaSecant" => return Box::new(AdaSecant::new(params[0], params[1], shape)),
        // "AMSGrad" => return Box::new(AMSGrad::new(params[0], params[1], shape)),
        // "AdaBound" => return Box::new(AdaBound::new(params[0], params[1], shape)),
        // "AMSBound" => return Box::new(AMSBound::new(params[0], params[1], shape)),
        // "AdaBelief" => return Box::new(AdaBelief::new(params[0], params[1], shape)),
        _ => panic!("Invalid optimizer name: {}", name),
    }
}

// pub enum OptimizerEnum<T, D> {
//     SGD(SGD<T, D>),
//     Momentum(Momentum<T, D>),
//     Nesterov(Nesterov<T, D>),
//     AdaGrad(AdaGrad<T, D>),
//     RMSprop(RMSprop<T, D>),
//     AdaDelta(AdaDelta<T, D>),
//     Adam(Adam<T, D>),
//     RMSpropGraves(RMSpropGraves<T, D>),
//     SMORMS3(SMORMS3<T, D>),
//     AdaMax(AdaMax<T, D>),
//     Nadam(Nadam<T, D>),
//     Eve(Eve<T, D>),
//     Santa(Santa<T, D>),
//     GDByGD(GDByGD<T, D>),
//     AdaSecant(AdaSecant<T, D>),
//     AMSGrad(AMSGrad<T, D>),
//     AdaBound(AdaBound<T, D>),
//     AMSBound(AMSBound<T, D>),
//     AdaBelief(AdaBelief<T, D>),
// }

// impl<T, D> OptimizerBase for OptimizerEnum<T, D>
// where
//     T: Float,
//     D: Dimension,
// {
//     type Src = Array<T, D>;
//     fn update(&mut self, param: &mut Self::Src, grads: &Self::Src) {
//         match self {
//             Self::SGD(x) => x.update(param, grads),
//             Self::Momentum(x) => x.update(param, grads),
//             Self::Nesterov(x) => x.update(param, grads),
//             Self::AdaGrad(x) => x.update(param, grads),
//             Self::RMSprop(x) => x.update(param, grads),
//             Self::Adam(x) => x.update(param, grads),
//             _ => panic!("Not implemented error."),
//         }
//     }
// }
