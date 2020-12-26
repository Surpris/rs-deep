//! batch_normalization
//!
//! BatchNormalization layer

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use super::super::util::CrateFloat;
use ndarray::prelude::*;
use std::fmt::{Debug, Display};

/// BatchNormalization
///
/// See http://arxiv.org/abs/1502.03167 in detail
pub struct BatchNormalization<T: CrateFloat, D> {
    gamma: T,
    beta: T,
    momentum: T,
    test_mean: T,
    test_var: T,
    batch_size: T,
    xc: Array<T, D>,
    std: Array<T, D>,
    dgamma: T,
    dbeta: T,
}

impl<T, D> BatchNormalization<T, D>
where
    T: CrateFloat,
    D: Dimension,
{
}
