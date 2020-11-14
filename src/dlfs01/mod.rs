//! # dlfs01
//!
//! Re-implementation of codes in `Deep learning from scratch`.

pub mod ch01;
pub mod ch02;
pub mod ch03;
pub mod ch04;
pub mod common;
pub mod dataset;

pub use common::gradient;
pub use common::loss_function::LossFunc;
pub use common::math::MathFunc;
pub use common::operators::Operators;
pub use common::util::cast_t2u;
