//! sequential.rs
//!
//! Sequential layer model

use std::{collections::HashMap, marker::PhantomData};

use ndarray::prelude::*;
use num_traits::Float;

use crate::dlfs01::common::layers::LayerBase;

/// Affine layer
pub struct Sequential<'a, T, D, L> {
    pub params: HashMap<&'a str, L>,
    pub grads: HashMap<&'a str, L>,
    _buff: Array<T, D>,
    _phantom: PhantomData<T>,
}

trait Test<T, D, L> {
    fn test(&mut self, x: &Array<T, D>);
}

impl<'a, T, D, L> Test<T, D, L> for Sequential<'a, T, D, L>
where
    T: Float,
    D: Dimension,
    L: LayerBase<T, D>,
{
    fn test(&mut self, x: &Array<T, D>) {
        for (_, val) in self.params.iter_mut() {
            let x = val.forward(x);
        }
    }
}
