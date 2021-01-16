//! model_base
//!
//! base traits for models

#![allow(unused_variables)]

use super::super::util::*;
use std::io;
use std::path::Path;

/// Arbitrary-D model trait
pub trait ModelBase<T: CrateFloat> {
    type A;
    type B;
    fn predict_prob(&mut self, x: &Self::A) -> Self::B;
    fn predict(&mut self, x: &Self::A) -> Self::B;
    fn loss(&mut self, x: &Self::A, t: &Self::B) -> T;
    fn accuracy(&mut self, x: &Self::A, t: &Self::B) -> T;
    fn gradient(&mut self, x: &Self::A, t: &Self::B);
    fn update(&mut self, x: &Self::A, t: &Self::B);
    fn set_trainable(&mut self, _flag: bool) {
        return;
    }
    fn print_detail(&self) {
        return;
    }
    fn print_parameters(&self) {
        return;
    }
    fn get_current_loss(&self) -> T {
        cast_t2u(0.0)
    }
    fn get_output(&self) -> Self::B;
    fn write_scheme_to_json(&self, dst: &Path) -> Result<(), io::Error>;
}
