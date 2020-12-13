//! convolution
//!
//! Convolution layer

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use ndarray::prelude::*;
use num_traits::Float;

/// Convolution
pub struct Convolution<T, D> {
    weight: Array<T, D>,
    bias: Array1<T>,
    stride: usize,
    pad_size: usize,
    x: Array<T, D>,
    col: Array<T, D>,
    col_weight: Array<T, D>,
    dw: Array<T, D>,
    db: Array1<T>,
}

impl<T, D> Convolution<T, D>
where
    T: Float,
    D: Dimension,
{
}
