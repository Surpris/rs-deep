//! layers
//!
//! Layers

pub mod activation;
pub mod affine;
pub mod batch_normalization;
pub mod convolution;
pub mod dropout;
pub mod layer_base;
pub mod pooling;
pub mod softmax_with_loss;

use super::util::CrateFloat;
pub use activation::{
    Identity, ReLU, ReLU2, ReLU3, ReLU4, ReLU5, ReLU6, ReLUD, Sigmoid, Sigmoid2, Sigmoid3,
    Sigmoid4, Sigmoid5, SigmoidD, Softmax, Softmax2, Softmax3, Softmax4, Softmax5, Softmax6,
    SoftmaxD,
};
pub use affine::Affine;
pub use batch_normalization::{call_batch_norm_layer, BatchNormalization, UseBatchNormEnum};
pub use convolution::Convolution;
pub use dropout::{call_dropout_layer, DropOut, UseDropoutEnum};
pub use layer_base::{LayerBase, LossLayerBase};
use ndarray::{prelude::*, RemoveAxis};
pub use pooling::{MaxPooling, MeanPooling, MinPooling};
pub use softmax_with_loss::{
    SoftmaxWithLoss, SoftmaxWithLoss2, SoftmaxWithLoss3, SoftmaxWithLoss4, SoftmaxWithLoss5,
    SoftmaxWithLoss6, SoftmaxWithLossD,
};
use std::fmt::Display;

/// Enum of basic layers
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BasicLayerEnum {
    Affine = 0,
    Convolution = 1,
}

impl Display for BasicLayerEnum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BasicLayerEnum::Affine => write!(f, "Affine"),
            BasicLayerEnum::Convolution => write!(f, "Convolution"),
        }
    }
}

/// Enum of activators
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ActivatorEnum {
    Identity = 0,
    ReLU = 1,
    Sigmoid = 2,
    Softmax = 3,
}

impl Display for ActivatorEnum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActivatorEnum::Identity => write!(f, "Identity"),
            ActivatorEnum::ReLU => write!(f, "ReLU"),
            ActivatorEnum::Sigmoid => write!(f, "Sigmoid"),
            ActivatorEnum::Softmax => write!(f, "Softmax"),
        }
    }
}

/// generate an activator
pub fn call_activator<T: 'static, D: 'static, Sh>(
    name: ActivatorEnum,
    shape: Sh,
    batch_axis: usize,
) -> Box<dyn LayerBase<T, A = Array<T, D>, B = Array<T, D>>>
where
    T: CrateFloat,
    D: Dimension + RemoveAxis,
    Sh: ShapeBuilder<Dim = D>,
{
    match name {
        ActivatorEnum::Identity => return Box::new(Identity::new(shape)),
        ActivatorEnum::ReLU => return Box::new(ReLU::new(shape)),
        ActivatorEnum::Sigmoid => return Box::new(Sigmoid::new(shape)),
        ActivatorEnum::Softmax => return Box::new(Softmax::new(shape, batch_axis)),
    }
}
