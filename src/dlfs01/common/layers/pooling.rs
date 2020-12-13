//! pooling
//!
//! Pooling layers

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use ndarray::prelude::*;
use num_traits::Float;

/// MaxPooling
pub struct MaxPooling<T, D> {
    pool_height: usize,
    pool_width: usize,
    stride: usize,
    pad_size: usize,
    x: Array<T, D>,
    arg: Array<T, D>,
}

impl<T, D> MaxPooling<T, D>
where
    T: Float,
    D: Dimension,
{
}

/// MinPooling
pub struct MinPooling<T, D> {
    pool_height: usize,
    pool_width: usize,
    stride: usize,
    pad_size: usize,
    x: Array<T, D>,
    arg: Array<T, D>,
}

impl<T, D> MinPooling<T, D>
where
    T: Float,
    D: Dimension,
{
}

/// MeanPooling
pub struct MeanPooling<T, D> {
    pool_height: usize,
    pool_width: usize,
    stride: usize,
    pad_size: usize,
    x: Array<T, D>,
    arg: Array<T, D>,
}

impl<T, D> MeanPooling<T, D>
where
    T: Float,
    D: Dimension,
{
}
