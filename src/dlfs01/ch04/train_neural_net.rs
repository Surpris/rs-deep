//! train_neura_lnet
//!
//! train a two-layer model

#![allow(dead_code)]

use super::two_layer_net::{Model, TwoLayerNet};
use crate::dlfs01::common::choice::Choice;
use crate::dlfs01::Operators;
// use crate::dlfs01::LossFunc;
use crate::dlfs01::dataset::{load_mnist, MNISTDataSetFlattened};
use rand::prelude::*;

type Vec2d<T> = Vec<Vec<T>>;

const HIDDEN_SIZE: usize = 50;
const NBR_OF_ITERS: usize = 10_000;
const BATCH_SIZE: usize = 100;
const LEARNING_RATE: f32 = 0.1;

pub struct TrainResult {
    train_loss_list: Vec<f32>,
    train_acc_list: Vec<f32>,
    test_acc_list: Vec<f32>,
}

impl TrainResult {
    pub fn new() -> Self {
        TrainResult {
            train_loss_list: Vec::new(),
            train_acc_list: Vec::new(),
            test_acc_list: Vec::new(),
        }
    }
}

pub fn main() {
    println!("< train_neural_net sub module >");
    // load MNIST dataset
    let data_set: MNISTDataSetFlattened<f32> = load_mnist(0u8).unwrap().flatten();

    // set a parameter for training
    let train_size: usize = data_set.train_images.len();
    let iter_per_epoch: usize = usize::max(train_size / BATCH_SIZE, 1);

    // initialize a two-layer model
    let mut network: TwoLayerNet<f32> = TwoLayerNet::new(
        data_set.train_labels[0].len(),
        HIDDEN_SIZE,
        data_set.train_labels[0].len(),
    );
    // initialize a TrainResult instance
    let mut train_result: TrainResult = TrainResult::new();

    // train loop
    let mut rng = thread_rng();
    let mut indices: Vec<usize> = Vec::new();
    for _ in 0..train_size {
        indices.push(rng.next_u64() as usize);
    }
    for ii in 0..NBR_OF_ITERS {
        // choose batched data set
        let x_batch: Vec2d<f32> = data_set
            .train_images
            .shuffle_copy_by_indices(&indices[ii * BATCH_SIZE..(ii + 1) * BATCH_SIZE]);
        let t_batch: Vec2d<f32> = data_set
            .train_labels
            .shuffle_copy_by_indices(&indices[ii * BATCH_SIZE..(ii + 1) * BATCH_SIZE]);

        // calculate gradient
        network.gradient_by_batch(&x_batch, &t_batch);

        // update parameters of the network
        network.w1 = network.w1.sub(&network.grad_w1.mul_value(LEARNING_RATE));
        network.b1 = network.b1.sub(&network.grad_b1.mul_value(LEARNING_RATE));
        network.w2 = network.w2.sub(&network.grad_w2.mul_value(LEARNING_RATE));
        network.b2 = network.b2.sub(&network.grad_b2.mul_value(LEARNING_RATE));

        // calculate loss
        let loss = network.loss_by_batch(&x_batch, &t_batch);
        train_result.train_loss_list.push(loss);

        // validation
        if ii % iter_per_epoch == 0 {}
    }
}
