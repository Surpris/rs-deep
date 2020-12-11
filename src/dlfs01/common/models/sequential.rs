//! sequential.rs
//!
//! Sequential layer model

#![allow(dead_code)]
#![allow(unused_variables)]

use std::{collections::HashMap, marker::PhantomData};

use ndarray::prelude::*;
use num_traits::Float;

use super::super::{layers::LayerBase, optimizers::OptimizerBase};

/// Affine layer
pub struct Sequential<'a, T, D, L, O> {
    pub params: HashMap<&'a str, L>,
    pub grads: HashMap<&'a str, L>,
    pub optimizer: O,
    _buff: Array<T, D>,
    _phantom: PhantomData<T>,
}

trait Test<T, D, L> {
    fn test(&mut self, x: &Array<T, D>);
}

impl<'a, T, D, L, O> Test<T, D, L> for Sequential<'a, T, D, L, O>
where
    T: Float,
    D: Dimension,
    L: LayerBase<T, D>,
    O: OptimizerBase<T, L>,
{
    fn test(&mut self, x: &Array<T, D>) {
        for (_, val) in self.params.iter_mut() {
            let x = val.forward(x);
        }
    }
}
