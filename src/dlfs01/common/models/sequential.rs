//! sequential.rs
//!
//! Sequential layer model
//!
//! This model will be constructed in future.

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::{collections::HashMap, marker::PhantomData};

use ndarray::prelude::*;
use num_traits::Float;

use super::super::optimizers::OptimizerBase;

// Sequential model
// pub struct Sequential<'a, T, D, O> {
//     pub params: Vec<LayerEnum<T, D>>,
//     pub grads: Vec<LayerEnum<T, D>>,
//     pub optimizers: HashMap<&'a str, O>,
//     _buff: Array<T, D>,
//     _phantom: PhantomData<T>,
// }

// impl<'a, T: 'static, D, L, O> Sequential<'a, T, D, L, O>
// where
//     T: Float,
//     D: Dimension,
//     L: LayerBase<T, D>,
//     O: OptimizerBase<T, D>,
// {
//     pub fn new<Sh>(
//         params: HashMap<&'a str, L>,
//         grads: HashMap<&'a str, L>,
//         optimizers: HashMap<&'a str, O>,
//     ) -> Self
//     where Sh: ShapeBuilder<Dim = D>
//     {
//         Self {
//             params, grads, optimizers,
//             _buff: Array::<T, D>::zeros((1, 1)),
//             _phantom: PhantomData::<T>::new(),
//         }
//     }
// }

// trait Test<T, D, L> {
//     fn test(&mut self, x: &Array<T, D>);
// }

// impl<'a, T, D, L, O> Test<T, D, L> for Sequential<'a, T, D, L, O>
// where
//     T: Float,
//     D: Dimension,
//     L: LayerBase<T>,
//     O: OptimizerBase<T, L>,
// {
//     fn test(&mut self, x: &Array<T, D>) {
//         for (_, val) in self.params.iter_mut() {
//             let x = val.forward(x);
//         }
//     }
// }
