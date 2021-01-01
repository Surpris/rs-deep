//! pooling
//!
//! Pooling layers

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use super::super::util::CrateFloat;
use ndarray::prelude::*;

/// MaxPooling
pub struct MaxPooling<T: CrateFloat, D> {
    pool_height: usize,
    pool_width: usize,
    stride: usize,
    pad_size: usize,
    x: Array<T, D>,
    arg: Array<T, D>,
}

impl<T, D> MaxPooling<T, D>
where
    T: CrateFloat,
    D: Dimension,
{
}

/// MinPooling
pub struct MinPooling<T: CrateFloat, D> {
    pool_height: usize,
    pool_width: usize,
    stride: usize,
    pad_size: usize,
    x: Array<T, D>,
    arg: Array<T, D>,
}

impl<T, D> MinPooling<T, D>
where
    T: CrateFloat,
    D: Dimension,
{
}

/// MeanPooling
pub struct MeanPooling<T: CrateFloat, D> {
    pool_height: usize,
    pool_width: usize,
    stride: usize,
    pad_size: usize,
    x: Array<T, D>,
    arg: Array<T, D>,
}

impl<T, D> MeanPooling<T, D>
where
    T: CrateFloat,
    D: Dimension,
{
}
