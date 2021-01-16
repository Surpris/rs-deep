//! weight_init
//!
//! Initializers of weights

use super::super::util::*;
use super::ndarray_init::*;
use ndarray::prelude::*;
use std::fmt::Display;

/// Enum of initializers of weights
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum WeightInitEnum {
    Normal = 0,
    ReLU = 1,
    He = 2,
    Sigmoid = 3,
    Xavier = 4,
}

impl Display for WeightInitEnum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WeightInitEnum::Normal => write!(f, "Normal"),
            WeightInitEnum::ReLU => write!(f, "ReLU"),
            WeightInitEnum::He => write!(f, "He"),
            WeightInitEnum::Sigmoid => write!(f, "Sigmoid"),
            WeightInitEnum::Xavier => write!(f, "Xavier"),
        }
    }
}

pub fn initialize_weight<T, D, Sh>(
    name: WeightInitEnum,
    weight_init_std: T,
    shape: Sh,
) -> Array<T, D>
where
    T: CrateFloat,
    D: Dimension,
    Sh: ShapeBuilder<Dim = D>,
{
    let src: Array<T, D> = initialize_randomized_ndarray(
        DistributionEnum::Normal,
        shape,
        &[cast_t2u(0.0), cast_t2u(1.0)],
    );
    let scale: T = match name {
        WeightInitEnum::Normal => weight_init_std,
        WeightInitEnum::ReLU | WeightInitEnum::He => {
            cast_t2u::<f64, T>(1.0) / cast_t2u::<usize, T>(src.len_of(Axis(0))).sqrt()
        }
        WeightInitEnum::Sigmoid | WeightInitEnum::Xavier => {
            cast_t2u::<f64, T>(2.0).sqrt() / cast_t2u::<usize, T>(src.len_of(Axis(0))).sqrt()
        }
    };
    src.map(|&v| v * scale)
}
