//! model_params
//!
//! Parameters for initialization of models

use super::super::layers::{ActivatorEnum, UseBatchNormEnum};
use super::super::optimizers::OptimizerEnum;
use super::super::param_initializers::WeightInitEnum;
use super::super::util::*;
use super::ModelEnum;
use serde::Deserialize;
use serde_json;
// use std::fs;
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::{fmt::Display, path::Path};

/// Model parameters
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelParameters<T: CrateFloat> {
    pub model_enum: ModelEnum,
    pub input_size: usize,
    pub hidden_sizes: Vec<usize>,
    pub output_size: usize,
    pub batch_axis: usize,
    pub activator_enums: Vec<ActivatorEnum>,
    pub optimizer_enum: OptimizerEnum,
    pub optimizer_params: Vec<T>,
    pub use_batch_norm: UseBatchNormEnum<T>,
    pub weight_init_enum: WeightInitEnum,
    pub weight_init_std: T,
}

impl<T: 'static> ModelParameters<T>
where
    T: CrateFloat,
{
    pub fn new() -> Self {
        Self {
            model_enum: ModelEnum::None,
            input_size: 0,
            hidden_sizes: Vec::new(),
            output_size: 0,
            batch_axis: 0,
            activator_enums: Vec::new(),
            optimizer_enum: OptimizerEnum::SGD,
            optimizer_params: Vec::new(),
            use_batch_norm: UseBatchNormEnum::None,
            weight_init_enum: WeightInitEnum::Normal,
            weight_init_std: cast_t2u(0.0),
        }
    }
    pub fn from(
        model_enum: ModelEnum,
        input_size: usize,
        hidden_sizes: Vec<usize>,
        output_size: usize,
        batch_axis: usize,
        activator_enums: Vec<ActivatorEnum>,
        optimizer_enum: OptimizerEnum,
        optimizer_params: Vec<T>,
        use_batch_norm: UseBatchNormEnum<T>,
        weight_init_enum: WeightInitEnum,
        weight_init_std: T,
    ) -> Self {
        Self {
            model_enum,
            input_size,
            hidden_sizes,
            output_size,
            batch_axis,
            activator_enums,
            optimizer_enum,
            optimizer_params,
            use_batch_norm,
            weight_init_enum,
            weight_init_std,
        }
    }
    pub fn from_json(src: &Path) -> Result<Self, io::Error>
    where
        T: for<'de> Deserialize<'de>,
    {
        let mut file: File = File::open(src)?;
        let mut buff: String = String::new();
        let _ = file.read_to_string(&mut buff);
        let dst: Self = serde_json::from_reader(buff.as_bytes())?;
        Ok(dst)
    }
    pub fn to_json(&self, dst: &Path) -> Result<(), io::Error> {
        let mut file: File = File::create(dst)?;
        write!(file, "{}", serde_json::to_string(&self)?)?;
        file.flush()?;
        Ok(())
    }
}

impl<T: 'static> Display for ModelParameters<T>
where
    T: CrateFloat,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut output = String::new();
        output += &format!("model type: {}", self.model_enum);
        output += &format!("size of the input layer: {}", self.input_size);
        output += &format!(
            "sizes of the hidden layers: {}",
            vec_to_string(&self.hidden_sizes)
        );
        output += &format!("output size: {}", self.output_size);
        output += &format!("axis of batch: {}", self.batch_axis);
        output += &format!(
            "types of activators: {}",
            vec_to_string(&self.activator_enums)
        );
        output += &format!("type of optimizer: {}", self.optimizer_enum);
        output += &format!(
            "parameters for optimizer: {}",
            vec_to_string(&self.optimizer_params)
        );
        output += &format!("weight init type: {}", self.weight_init_enum);
        output += &format!("weight init std: {}", self.weight_init_std);
        write!(f, "{}", output)
    }
}

fn vec_to_string<T>(src: &[T]) -> String
where
    T: Display,
{
    let mut dst = String::from("[");
    for v in src.iter() {
        dst += &format!("{},", *v);
    }
    dst + &"]"
}
