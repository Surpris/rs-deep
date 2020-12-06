//! layers
//!
//! Layers

pub mod activation;
pub mod affine;
pub mod layer_base;
pub mod softmax_with_loss;

pub use activation::{ReLU, ReLU2, Sigmoid, Sigmoid2, Softmax, Softmax2};
pub use affine::Affine;
pub use layer_base::*;
pub use softmax_with_loss::{SoftmaxWithLoss, SoftmaxWithLoss2};
