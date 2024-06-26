//! prelude
//!
//! crate prelude.
//!
//! This module contains the most used types, type aliases, traits, functions, and macros that you can import easily as a group
//!
//! ```
//!
//! use rs_deep::prelude::*;
//! ```

#[doc(no_inline)]
pub use crate::dlfs01::common::layers::{call_activator, ActivatorEnum, BasicLayerEnum};

#[doc(no_inline)]
pub use crate::dlfs01::common::layers::activation::{
    ReLU, ReLU2, ReLU3, ReLU4, ReLU5, ReLU6, ReLUD, Sigmoid, Sigmoid2, Sigmoid3, Sigmoid4,
    Sigmoid5, SigmoidD, Softmax, Softmax2, Softmax3, Softmax4, Softmax5, Softmax6, SoftmaxD,
};

#[doc(no_inline)]
pub use crate::dlfs01::common::layers::affine::Affine;

#[doc(no_inline)]
pub use crate::dlfs01::common::layers::batch_normalization::{
    call_batch_norm_layer, BatchNormalization, UseBatchNormEnum,
};

#[doc(no_inline)]
pub use crate::dlfs01::common::layers::dropout::{call_dropout_layer, DropOut, UseDropoutEnum};

#[doc(no_inline)]
pub use crate::dlfs01::common::layers::layer_base::{LayerBase, LossLayerBase};

#[doc(no_inline)]
pub use crate::dlfs01::common::layers::softmax_with_loss::{
    SoftmaxWithLoss, SoftmaxWithLoss2, SoftmaxWithLoss3, SoftmaxWithLoss4, SoftmaxWithLoss5,
    SoftmaxWithLoss6, SoftmaxWithLossD,
};

#[doc(no_inline)]
pub use crate::dlfs01::common::models::mlp::MLPClassifier;

#[doc(no_inline)]
pub use crate::dlfs01::common::models::ModelEnum;

#[doc(no_inline)]
pub use crate::dlfs01::common::models::model_base::ModelBase;

#[doc(no_inline)]
pub use crate::dlfs01::common::models::model_params::ModelParameters;

#[doc(no_inline)]
pub use crate::dlfs01::common::optimizers::{call_optimizer, OptimizerEnum};

#[doc(no_inline)]
pub use crate::dlfs01::common::optimizers::optimizer_base::OptimizerBase;

#[doc(no_inline)]
pub use crate::dlfs01::common::optimizers::optimizer::{AdaGrad, Adam, Nesterov, RMSprop, SGD};

#[doc(no_inline)]
pub use crate::dlfs01::common::param_initializers::ndarray_init::{
    initialize_randomized_ndarray, DistributionEnum,
};

#[doc(no_inline)]
pub use crate::dlfs01::common::param_initializers::weight_init::{
    initialize_weight, WeightInitEnum,
};

#[doc(no_inline)]
pub use crate::dlfs01::common::regularizers::{
    call_regularizer, L1Norm, L2Norm, RegularizerBase, RegularizerEnum,
};

#[doc(no_inline)]
pub use crate::dlfs01::common::trainers::{TrainResult, Trainer};

#[doc(no_inline)]
pub use crate::dlfs01::common::util::{cast_t2u, CrateFloat};

// #[doc(no_inline)]
// pub use crate::dlfs01::dataset::{DataSetError, MNISTDataSet, MNISTDataSetArray2};

#[doc(no_inline)]
pub use crate::dlfs01::dataset::mnist::{MNISTDataSet2, MNISTDataSet4};
