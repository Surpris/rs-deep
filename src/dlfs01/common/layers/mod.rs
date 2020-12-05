//! layers
//!
//! Layers

pub mod activation;
pub mod affine;
pub mod softmax_with_loss;

pub use activation::{ActivationBase, ReLU, Sigmoid, Softmax};
pub use affine::{Affine, AffineBase};
pub use softmax_with_loss::{SoftmaxWithLoss, SoftmaxWithLossBase};
