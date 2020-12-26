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

pub use activation::{
    ReLU, ReLU2, ReLU3, ReLU4, ReLU5, ReLU6, ReLUD, Sigmoid, Sigmoid2, Sigmoid3, Sigmoid4,
    Sigmoid5, SigmoidD, Softmax, Softmax2, Softmax3, Softmax4, Softmax5, Softmax6, SoftmaxD,
};
pub use affine::Affine;
pub use batch_normalization::BatchNormalization;
pub use convolution::Convolution;
pub use dropout::DropOut;
pub use layer_base::{LayerBase, LossLayerBase};
use ndarray::{prelude::*, RemoveAxis};
use num_traits::Float;
pub use pooling::{MaxPooling, MeanPooling, MinPooling};
pub use softmax_with_loss::{
    SoftmaxWithLoss, SoftmaxWithLoss2, SoftmaxWithLoss3, SoftmaxWithLoss4, SoftmaxWithLoss5,
    SoftmaxWithLoss6, SoftmaxWithLossD,
};

pub fn call_activator<T: 'static, D: 'static, Sh>(
    name: &str,
    shape: Sh,
    batch_axis: usize,
) -> Box<dyn LayerBase<T, A = Array<T, D>, B = Array<T, D>>>
where
    T: Float,
    D: Dimension + RemoveAxis,
    Sh: ShapeBuilder<Dim = D>,
{
    match name {
        "relu" => return Box::new(ReLU::new(shape)),
        "softmax" => return Box::new(Softmax::new(shape, batch_axis)),
        "sigmoid" => return Box::new(Sigmoid::new(shape)),
        _ => panic!("Invalid activator name: {}", name),
    }
}

// pub enum LayerEnum<T, D> {
//     ReLU(ReLU<T, D>),
//     Sigmoid(Sigmoid<T, D>),
//     Softmax(Softmax<T, D>),
//     Affine(Affine<T>),
// }

// impl<T> LayerBase<T> for LayerEnum<T, D>
// where
//     T: Float,
//     D: Dimension,
// {
//     type Array<T, D>;
//     type B;
//     fn forward(&mut self, x: &Self::A) -> Self::B;
//     fn backward(&mut self, dx: &Self::B) -> Self::A;
//     fn update(&mut self, lr: T) {
//         return;
//     }
//     fn print_detail(&self);
// }

// pub enum LossLayerEnum<T, D> {
//     SoftmaxWithLoss(SoftmaxWithLoss<T, D>),
// }
