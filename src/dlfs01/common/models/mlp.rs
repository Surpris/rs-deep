//! mlp
//!
//! Multi-layer perceptron model
//!
//! This model can be used only with datasets composed of 1D data.

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use serde::Deserialize;
use std::io::{self, ErrorKind};
use std::path::Path;

use crate::dlfs01::common::regularizers::{call_regularizer, RegularizerBase, RegularizerEnum};

use super::super::optimizers::*;
use super::super::param_initializers::weight_init::WeightInitEnum;
use super::super::util::*;
use super::model_base::ModelBase;
use super::model_params::ModelParameters;
use super::{super::layers::*, ModelEnum};

/// MLP classifier
pub struct MLPClassifier<T: 'static + CrateFloat> {
    affine_layers: Vec<Affine<T>>,
    batch_norm_layers: Vec<BatchNormalization<T, Ix2>>,
    activators: Vec<Box<dyn LayerBase<T, A = Array2<T>, B = Array2<T>>>>,
    dropout_layers: Vec<DropOut<T, Ix2>>,
    loss_layer: Box<dyn LossLayerBase<T, A = Array2<T>>>,
    optimizer_weight: Box<dyn OptimizerBase<Src = Array2<T>>>,
    optimizer_bias: Box<dyn OptimizerBase<Src = Array1<T>>>,
    regularizer_enum: RegularizerEnum<T>,
    regularizer: Box<dyn RegularizerBase<T, A = Array2<T>>>,
    current_regularizer_value: T,
    current_loss: T,
    nbr_of_hidden_layers: usize,
    nbr_of_affine_layers: usize,
    params: ModelParameters<T>,
}

impl<T: 'static> MLPClassifier<T>
where
    T: CrateFloat,
{
    pub fn new(
        input_size: usize,
        hidden_sizes: &[usize],
        output_size: usize,
        activator_enums: &[ActivatorEnum],
        optimizer_enum: OptimizerEnum<T>,
        use_batch_norm: UseBatchNormEnum<T>,
        use_dropout: UseDropoutEnum<T>,
        regularizer_enum: RegularizerEnum<T>,
        batch_axis: usize,
        weight_init_enum: WeightInitEnum,
        weight_init_std: T,
    ) -> Self {
        assert_eq!(hidden_sizes.len(), activator_enums.len());
        let params: ModelParameters<T> = ModelParameters::from(
            ModelEnum::MLPClassifier,
            input_size,
            hidden_sizes.to_vec(),
            output_size,
            batch_axis,
            activator_enums.to_vec(),
            optimizer_enum,
            use_batch_norm,
            use_dropout,
            regularizer_enum,
            weight_init_enum,
            weight_init_std,
        );
        match Self::from(params) {
            Ok(x) => x,
            Err(err) => panic!("{}", err.to_string()),
        }
    }
    pub fn from(params: ModelParameters<T>) -> Result<Self, io::Error> {
        if params.model_enum != ModelEnum::MLPClassifier {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "The model type specified by the input is not `MLPClassifier`.",
            ));
        }
        let params_clone = params.clone();
        let nbr_of_hidden_layers: usize = params.hidden_sizes.len();
        let mut affine_layers: Vec<Affine<T>> = Vec::new();
        let mut activators: Vec<Box<dyn LayerBase<T, A = Array2<T>, B = Array2<T>>>> = Vec::new();
        let mut batch_norm_layers: Vec<BatchNormalization<T, Ix2>> = Vec::new();
        let mut dropout_layers: Vec<DropOut<T, Ix2>> = Vec::new();
        for ii in 0..nbr_of_hidden_layers {
            if ii == 0 {
                affine_layers.push(Affine::new(
                    (params.input_size, params.hidden_sizes[ii]),
                    params.weight_init_enum.clone(),
                    params.weight_init_std,
                ));
            } else {
                affine_layers.push(Affine::new(
                    (params.hidden_sizes[ii - 1], params.hidden_sizes[ii]),
                    params.weight_init_enum.clone(),
                    params.weight_init_std,
                ));
            }
            batch_norm_layers.push(call_batch_norm_layer(
                params.use_batch_norm.clone(),
                (params.hidden_sizes[ii], params.hidden_sizes[ii]),
                params.batch_axis,
            ));
            activators.push(call_activator(
                params.activator_enums[ii].clone(),
                (params.hidden_sizes[ii], params.hidden_sizes[ii]),
                params.batch_axis,
            ));
            dropout_layers.push(call_dropout_layer(
                params.use_dropout.clone(),
                (params.hidden_sizes[ii], params.hidden_sizes[ii]),
            ));
        }
        affine_layers.push(Affine::new(
            (
                params.hidden_sizes[nbr_of_hidden_layers - 1],
                params.output_size,
            ),
            params.weight_init_enum,
            params.weight_init_std,
        ));
        // batch_norm_layers.push(call_batch_norm_layer(
        //     params.use_batch_norm.clone(),
        //     (params.output_size, params.output_size),
        //     params.batch_axis,
        // ));
        // dropout_layers.push(call_dropout_layer(
        //     params.use_dropout.clone(),
        //     (params.output_size, params.output_size),
        // ));
        let loss_layer = Box::new(SoftmaxWithLoss2::new(
            (params.output_size, params.output_size),
            params.batch_axis,
        ));
        let optimizer_weight = call_optimizer(
            params.optimizer_enum.clone(),
            (params.output_size, params.output_size),
        );
        let optimizer_bias = call_optimizer(
            params.optimizer_enum,
            params.hidden_sizes[params.hidden_sizes.len() - 1],
        );
        let regularizer = call_regularizer(params.regularizer_enum.clone());
        let nbr_of_affine_layers: usize = affine_layers.len();
        Ok(Self {
            affine_layers,
            batch_norm_layers,
            activators,
            dropout_layers,
            loss_layer,
            optimizer_weight,
            optimizer_bias,
            regularizer_enum: params.regularizer_enum,
            regularizer,
            current_regularizer_value: cast_t2u(0.0),
            current_loss: cast_t2u(0.0),
            nbr_of_hidden_layers,
            nbr_of_affine_layers,
            params: params_clone,
        })
    }
    pub fn read_scheme_from_json(src: &Path) -> Result<Self, io::Error>
    where
        T: for<'de> Deserialize<'de>,
    {
        let params: ModelParameters<T> = ModelParameters::from_json(src)?;
        Self::from(params)
    }
}

impl<T: 'static> ModelBase<T> for MLPClassifier<T>
where
    T: CrateFloat,
{
    type A = Array2<T>;

    type B = Array2<T>;

    fn predict_prob(&mut self, x: &Self::A) -> Self::B {
        // The first layer set
        let mut y: Self::B = self.affine_layers[0].forward(x);
        if self.params.use_batch_norm != UseBatchNormEnum::None {
            y = self.batch_norm_layers[0].forward(&y);
        }
        y = self.activators[0].forward(&y);
        if self.params.use_dropout != UseDropoutEnum::None {
            y = self.dropout_layers[0].forward(&y);
        }
        // Hidden layer sets
        for ii in 1..self.nbr_of_hidden_layers {
            y = self.affine_layers[ii].forward(&y);
            if self.params.use_batch_norm != UseBatchNormEnum::None {
                y = self.batch_norm_layers[ii + 1].forward(&y);
            }
            y = self.activators[ii].forward(&y);
            if self.params.use_dropout != UseDropoutEnum::None {
                y = self.dropout_layers[ii].forward(&y);
            }
        }
        // The last Affine layer
        self.affine_layers[self.nbr_of_affine_layers - 1].forward(&y)
    }

    fn predict(&mut self, x: &Self::A) -> Self::B {
        let one: T = cast_t2u(1.0);
        let y: Self::B = self.predict_prob(&x);
        let mut dst: Self::B = Array2::zeros(y.raw_dim());
        for (view1, mut view2) in y.axis_iter(Axis(0)).zip(dst.axis_iter_mut(Axis(0))) {
            let y_argmax = view1.argmax().unwrap();
            view2[y_argmax] = one;
        }
        dst
    }

    fn loss(&mut self, x: &Self::A, t: &Self::B) -> T {
        let y: Self::B = self.predict_prob(&x);
        self.current_loss = self.loss_layer.forward(&y, &t);
        if self.regularizer_enum != RegularizerEnum::None {
            self.current_regularizer_value = cast_t2u(0.0);
            for layer in self.affine_layers.iter() {
                self.current_regularizer_value += self.regularizer.forward(&layer.weight);
            }
            self.current_loss + self.current_regularizer_value
        } else {
            self.current_loss
        }
    }

    fn accuracy(&mut self, x: &Self::A, t: &Self::B) -> T {
        let y: Self::B = self.predict(&x);
        let mut acc: f32 = 0.0;
        for (view1, view2) in y.axis_iter(Axis(0)).zip(t.axis_iter(Axis(0))) {
            let y_argmax = view1.argmax().unwrap();
            let t_argmax = view2.argmax().unwrap();
            if y_argmax == t_argmax {
                acc += 1.0;
            }
        }
        cast_t2u(acc / t.len_of(Axis(0)) as f32)
    }

    fn gradient(&mut self, x: &Self::A, t: &Self::B) {
        // forward
        let _ = self.loss(&x, &t);

        // backward
        let _dx: T = cast_t2u(1.0);
        let mut _dx: Self::B = self.loss_layer.backward(_dx);
        for ii in 0..self.nbr_of_hidden_layers {
            _dx = self.affine_layers[self.nbr_of_affine_layers - 1 - ii].backward(&_dx);
            if self.params.use_dropout != UseDropoutEnum::None {
                _dx = self.dropout_layers[self.nbr_of_hidden_layers - 1 - ii].backward(&_dx);
            }
            _dx = self.activators[self.nbr_of_hidden_layers - 1 - ii].backward(&_dx);
            if self.params.use_batch_norm != UseBatchNormEnum::None {
                _dx = self.batch_norm_layers[self.nbr_of_hidden_layers - 1 - ii].backward(&_dx);
            }
        }
        _dx = self.affine_layers[0].backward(&_dx);
        if self.regularizer_enum != RegularizerEnum::None {
            for layer in self.affine_layers.iter_mut() {
                layer.dw = self.regularizer.backward(&layer.weight) + &layer.dw;
            }
        }
    }

    fn update(&mut self, x: &Self::A, t: &Self::B) {
        self.gradient(&x, &t);
        for layer in self.affine_layers.iter_mut() {
            self.optimizer_weight
                .update(&mut layer.weight, &mut layer.dw);
            self.optimizer_bias.update(&mut layer.bias, &mut layer.db);
        }
        if self.params.use_batch_norm != UseBatchNormEnum::None {
            for layer in self.batch_norm_layers.iter_mut() {
                self.optimizer_bias
                    .update(&mut layer.gamma, &mut layer.dgamma);
                self.optimizer_bias
                    .update(&mut layer.beta, &mut layer.dbeta);
            }
        }
    }

    fn set_trainable(&mut self, flag: bool) {
        if self.params.use_batch_norm != UseBatchNormEnum::None {
            for layer in self.batch_norm_layers.iter_mut() {
                layer.set_trainable(flag);
            }
        }
        if self.params.use_dropout != UseDropoutEnum::None {
            for layer in self.dropout_layers.iter_mut() {
                layer.set_trainable(flag);
            }
        }
    }

    fn print_detail(&self) {
        println!("MLP classifier.");
        for ii in 0..self.nbr_of_hidden_layers {
            self.affine_layers[ii].print_detail();
            if self.params.use_batch_norm != UseBatchNormEnum::None {
                self.batch_norm_layers[ii].print_detail();
            }
            self.activators[ii].print_detail();
            if self.params.use_dropout != UseDropoutEnum::None {
                self.dropout_layers[ii].print_detail();
            }
        }
        self.affine_layers[self.nbr_of_affine_layers - 1].print_detail();
        self.loss_layer.print_detail();
    }

    fn print_parameters(&self) {
        println!("Affine layers:");
        for ii in 0..self.nbr_of_hidden_layers {
            println!("Layer {}:", ii);
            self.affine_layers[ii].print_parameters();
        }
        println!("Layer {}:", self.affine_layers.len() - 1);
        self.affine_layers[self.affine_layers.len() - 1].print_parameters();
        if self.params.use_batch_norm != UseBatchNormEnum::None {
            println!("BatchNormalization layers:");
            for ii in 0..self.batch_norm_layers.len() - 1 {
                println!("Layer {}:", ii);
                self.batch_norm_layers[ii].print_parameters();
            }
        }
    }

    fn get_current_loss(&self) -> T {
        self.current_loss
    }

    fn get_output(&self) -> Self::B {
        self.loss_layer.get_output()
    }

    fn write_scheme_to_json(&self, dst: &Path) -> Result<(), io::Error> {
        self.params.to_json(dst)?;
        Ok(())
    }
}
