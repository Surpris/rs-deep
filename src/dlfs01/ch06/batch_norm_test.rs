//! batch_norm_test.rs
//!
//! test of batch normalization

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::prelude::*;
use ndarray::prelude::*;
use rand::prelude::*;
use std::time::Instant;

type FF = f64;

const EPOCHS: usize = 17;
const HIDDEN_SIZE: usize = 50;
const NBR_OF_ITERS: usize = 10000;
const BATCH_SIZE: usize = 100;
const NBR_OF_TARGET_IMAGES: usize = 60000;
const NBR_OF_SAMPLES: usize = 500;
const LOG_TEMPORAL_RESULT: bool = false;

const VERBOSE: u8 = 1;

#[derive(Clone, Debug)]
struct TrainResult {
    train_loss_list: Vec<FF>,
    train_acc_list: Vec<FF>,
    test_acc_list: Vec<FF>,
}

impl TrainResult {
    fn new() -> Self {
        TrainResult {
            train_loss_list: Vec::new(),
            train_acc_list: Vec::new(),
            test_acc_list: Vec::new(),
        }
    }
}

pub fn main() {
    println!("< ch06 batch_norm_test sub module >");
    // load MNIST dataset
    println!("load MNIST dataset...");
    let data_set: MNISTDataSet2<FF> = MNISTDataSet2::<u8>::new(VERBOSE).unwrap().to_f64();
    let input_size: usize = data_set.train_images.len_of(Axis(1));
    let output_size: usize = data_set.train_labels.len_of(Axis(1));
    let batch_axis: usize = 0;
    let data_set: MNISTDataSet2<FF> = MNISTDataSet2 {
        train_images: 1.0 * &data_set.train_images.slice(s![..NBR_OF_TARGET_IMAGES, ..]),
        train_labels: 1.0 * &data_set.train_labels.slice(s![..NBR_OF_TARGET_IMAGES, ..]),
        test_images: data_set.test_images,
        test_labels: data_set.test_labels,
    };

    // set a parameter for training
    let nbr_train_images: usize = data_set.train_images.len_of(Axis(0));
    let iter_per_epoch: usize = usize::max(nbr_train_images / BATCH_SIZE, 1);

    // set activators and optimizers
    let hidden_sizes: [usize; 1] = [HIDDEN_SIZE];
    let activator_enums: [ActivatorEnum; 1] = [ActivatorEnum::ReLU];
    let optimizer_enum: OptimizerEnum<FF> = OptimizerEnum::SGD(0.1);
    // let optimizer_enum: OptimizerEnum<FF> = OptimizerEnum::AdaGrad(0.1);
    // let optimizer_enum: OptimizerEnum<FF> = OptimizerEnum::Momentum(0.01, 0.9);
    // let optimizer_enum: OptimizerEnum<FF> = OptimizerEnum::Nesterov(0.01, 0.9);
    // let optimizer_enum: OptimizerEnum<FF> = OptimizerEnum::RMSprop(0.01, 0.99);
    // let optimizer_enum: OptimizerEnum<FF> = OptimizerEnum::Adam(0.001, 0.9, 0.999);
    let use_batch_norm = UseBatchNormEnum::Use(0.9);
    let use_dropout_enum: UseDropoutEnum<FF> = UseDropoutEnum::None;
    let regularizer_enum: RegularizerEnum<FF> = RegularizerEnum::None;

    // initialize a multi-layer model
    println!("initialize a model...");
    let weight_init: WeightInitEnum = WeightInitEnum::Normal;
    let weight_init_params: FF = 0.01;
    let mut network: Box<dyn ModelBase<FF, A = Array2<FF>, B = Array2<FF>>> =
        Box::new(MLPClassifier::new(
            input_size,
            &hidden_sizes,
            output_size,
            &activator_enums,
            optimizer_enum,
            use_batch_norm,
            use_dropout_enum,
            regularizer_enum,
            batch_axis,
            weight_init,
            weight_init_params,
        ));
    network.print_detail();
    println!("Initial parameters:");
    network.print_parameters();

    // initialize a TrainResult instance
    let mut trainer: Trainer<FF, Ix2, Ix2> = Trainer::new(
        data_set.train_images.clone(),
        data_set.train_labels.clone(),
        data_set.test_images.clone(),
        data_set.test_labels.clone(),
        0,
        EPOCHS,
        BATCH_SIZE,
        NBR_OF_SAMPLES,
        LOG_TEMPORAL_RESULT,
        VERBOSE as usize,
    );

    // train loop
    println!("start training...");
    trainer.train(&mut network);
    println!("training finished.");
    println!("{} sec elapsed to training.", trainer.get_elapsed_time());
    println!("Final parameters:");
    network.print_parameters();
    print!("validation... ");
    let train_acc = network.accuracy(&data_set.train_images, &data_set.train_labels);
    let test_acc = network.accuracy(&data_set.test_images, &data_set.test_labels);
    println!("acc : train={}, test={}", train_acc, test_acc);
    println!("< ch06 batch_norm_test sub module > finished.");
}
