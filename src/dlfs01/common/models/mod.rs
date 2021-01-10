//! models
//!
//! models

pub mod mlp;
pub mod model_base;
pub mod model_params;
pub mod sequential;

pub use mlp::MLPClassifier;
pub use model_base::ModelBase;
pub use model_params::ModelParameters;
// pub use sequential::Sequential;

use std::fmt::Display;

/// Enum of models
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ModelEnum {
    None = -1,
    MLPClassifier = 0,
}

impl Display for ModelEnum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelEnum::None => write!(f, "None"),
            ModelEnum::MLPClassifier => write!(f, "MLPClassifier"),
        }
    }
}
