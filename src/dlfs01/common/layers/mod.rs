//! layers
//!
//! Layers

pub mod activation;
pub mod affine;
pub mod layer_base;
pub mod softmax_with_loss;

pub use activation::{
    ReLU, ReLU2, ReLU3, ReLU4, ReLU5, ReLU6, ReLUD, Sigmoid, Sigmoid2, Sigmoid3, Sigmoid4,
    Sigmoid5, SigmoidD, Softmax, Softmax2, Softmax3, Softmax4, Softmax5, Softmax6, SoftmaxD,
};
pub use affine::Affine;
pub use layer_base::*;
pub use softmax_with_loss::{
    SoftmaxWithLoss, SoftmaxWithLoss2, SoftmaxWithLoss3, SoftmaxWithLoss4, SoftmaxWithLoss5,
    SoftmaxWithLoss6, SoftmaxWithLossD,
};
