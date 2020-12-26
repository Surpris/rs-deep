//! models
//!
//! models

pub mod mlp;
pub mod model_base;
pub mod sequential;

pub use mlp::MLPClassifier;
pub use model_base::ModelBase;
// pub use sequential::Sequential;

use std::fmt::Display;

pub enum ModelEnum {
    MLPClassifier = 0,
}

impl Display for ModelEnum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelEnum::MLPClassifier => write!(f, "MLPClassifier"),
        }
    }
}
