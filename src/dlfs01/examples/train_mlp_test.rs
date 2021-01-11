//! train_mlp_test
//!
//! Test to train A MLP classifier.
//! The model to train is generated according to a scheme file if given.

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::prelude::*;
use ndarray::prelude::*;
use rand::prelude::*;
use std::path::Path;
use std::time::Instant;

type FF = f64;

const SAVE_MODEL_PATH: &str = "./data/models/model.json";
const EPOCHS: usize = 17;
const HIDDEN_SIZE: usize = 50;
const NBR_OF_ITERS: usize = 10000;
const BATCH_SIZE: usize = 100;
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
    println!("< examples train_mlp_test sub module >");
    // load MNIST dataset
    println!("load MNIST dataset...");
    let data_set: MNISTDataSet2<FF> = MNISTDataSet2::<u8>::new(VERBOSE).unwrap().to_f64();
    let input_size: usize = data_set.train_images.len_of(Axis(1));
    let output_size: usize = data_set.train_labels.len_of(Axis(1));
    let batch_axis: usize = 0;

    // set a parameter for training
    let nbr_train_images: usize = data_set.train_images.len_of(Axis(0));
    let iter_per_epoch: usize = usize::max(nbr_train_images / BATCH_SIZE, 1);

    let args: Vec<String> = std::env::args().collect();
    let mut model: Box<dyn ModelBase<FF, A = Array2<FF>, B = Array2<FF>>>;
    if args.len() == 1 {
        println!("No model scheme specified. A model is built with the default parameters...");
        // set activators and optimizers
        let hidden_sizes: [usize; 1] = [HIDDEN_SIZE];
        let activator_enums: [ActivatorEnum; 1] = [ActivatorEnum::ReLU];
        let optimizer_enum: OptimizerEnum = OptimizerEnum::SGD;
        let optimizer_params: Vec<FF> = match optimizer_enum {
            OptimizerEnum::SGD | OptimizerEnum::AdaGrad => vec![0.1],
            OptimizerEnum::Momentum | OptimizerEnum::Nesterov => vec![0.01, 0.9],
            OptimizerEnum::RMSprop => vec![0.01, 0.99],
            OptimizerEnum::Adam => vec![0.001, 0.9, 0.999],
            _ => panic!(),
        };
        let use_batch_norm = UseBatchNormEnum::None;
        let weight_init: WeightInitEnum = WeightInitEnum::Normal;
        let weight_init_std: FF = 0.01;

        model = Box::new(MLPClassifier::new(
            input_size,
            &hidden_sizes,
            output_size,
            &activator_enums,
            optimizer_enum,
            &optimizer_params,
            use_batch_norm,
            batch_axis,
            weight_init,
            weight_init_std,
        ));
    } else {
        model = match Path::new(&args[1]).canonicalize() {
            Ok(pb) => {
                println!(
                    "load a scheme of a model from `{}`...",
                    pb.to_str().unwrap()
                );
                match MLPClassifier::<FF>::read_scheme_from_json(&pb) {
                    Ok(x) => Box::new(x),
                    Err(err) => panic!("Failure in loading the model: {}", err.to_string()),
                }
            }
            Err(err) => panic!(
                "Failure in validating the input parameter 1: {}",
                err.to_string()
            ),
        }
    }
    println!("Model:");
    model.print_detail();
    // println!("Initial parameters:");
    // model.print_parameters();

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
    trainer.train(&mut model);
    println!("training finished.");
    println!("{} sec elapsed to training.", trainer.get_elapsed_time());

    // println!("Final parameters:");
    // model.print_parameters();

    print!("validation... ");
    let train_acc = model.accuracy(&data_set.train_images, &data_set.train_labels);
    let test_acc = model.accuracy(&data_set.test_images, &data_set.test_labels);
    println!("acc : train={}, test={}", train_acc, test_acc);

    match model.write_scheme_to_json(Path::new(SAVE_MODEL_PATH)) {
        Ok(()) => println!(
            "The scheme of the model has been saved to `{}`.",
            SAVE_MODEL_PATH
        ),
        Err(err) => println!(
            "Failure in saving the scheme of the model: {}.",
            err.to_string()
        ),
    }
    println!("< examples train_mlp_test sub module > finished.");
}
