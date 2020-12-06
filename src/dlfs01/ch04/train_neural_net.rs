//! train_neural_net
//!
//! train a two-layer model

#![allow(dead_code)]

use super::two_layer_net::{Model, TwoLayerNet};
use crate::dlfs01::common::choice::Choice;
use crate::dlfs01::dataset::{load_mnist, MNISTDataSetFlattened};
use crate::dlfs01::math::arange;
use crate::dlfs01::MathFunc;
use crate::dlfs01::Operators;
use plotters::prelude::*;
use rand::prelude::*;

type Vec2d<T> = Vec<Vec<T>>;

type FF = f64;

const HIDDEN_SIZE: usize = 50;
const NBR_OF_ITERS: usize = 100;
const BATCH_SIZE: usize = 100;
const LEARNING_RATE: FF = 0.1;
const NBR_OF_TARGET_IMAGES: usize = 10000;

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
    println!("< train_neural_net sub module >");
    // load MNIST dataset
    println!("load MNIST dataset...");
    let data_set: MNISTDataSetFlattened<FF> = load_mnist(0u8).unwrap().flatten();
    let data_set: MNISTDataSetFlattened<FF> = MNISTDataSetFlattened {
        train_images: data_set.train_images[..NBR_OF_TARGET_IMAGES].to_vec(),
        train_labels: data_set.train_labels[..NBR_OF_TARGET_IMAGES].to_vec(),
        test_images: data_set.test_images[..NBR_OF_TARGET_IMAGES].to_vec(),
        test_labels: data_set.test_labels[..NBR_OF_TARGET_IMAGES].to_vec(),
    };

    // set a parameter for training
    let nbr_train_images: usize = data_set.train_images.len();
    // let iter_per_epoch: usize = usize::max(nbr_train_images / BATCH_SIZE, 1);
    let iter_per_epoch: usize = 10;

    // initialize a two-layer model
    println!("initialize a model...");
    let mut network: TwoLayerNet<FF> = TwoLayerNet::new(
        data_set.train_images[0].len(),
        HIDDEN_SIZE,
        data_set.train_labels[0].len(),
    );
    network.print_detail();
    network.verbose = false;

    // initialize a TrainResult instance
    let mut train_result: TrainResult = TrainResult::new();

    // train loop
    let mut rng = thread_rng();
    println!("start training...");
    for ii in 0..NBR_OF_ITERS {
        // choose indices
        let mut indices: Vec<usize> = Vec::new();
        for _ in 0..BATCH_SIZE {
            indices.push(rng.gen_range(0, nbr_train_images));
        }
        // choose batched data set
        let x_batch: Vec2d<FF> = data_set.train_images.shuffle_copy_by_indices(&indices);
        let t_batch: Vec2d<FF> = data_set.train_labels.shuffle_copy_by_indices(&indices);

        // calculate gradient
        network.gradient_by_batch(&x_batch, &t_batch);

        // update parameters of the network
        // println!("update parameters...");
        network.w1 = network.w1.sub(&network.grad_w1.mul_value(LEARNING_RATE));
        network.b1 = network.b1.sub(&network.grad_b1.mul_value(LEARNING_RATE));
        network.w2 = network.w2.sub(&network.grad_w2.mul_value(LEARNING_RATE));
        network.b2 = network.b2.sub(&network.grad_b2.mul_value(LEARNING_RATE));

        // calculate loss
        // println!("calculate loss...");
        let loss = network.loss_by_batch(&x_batch, &t_batch);
        println!("train loss at step {}: {}", ii + 1, loss);
        train_result.train_loss_list.push(loss);

        // validation
        if (ii + 1) % iter_per_epoch == 0 {
            print!("validation: ");
            print!("train images... ");
            let train_acc = network.accuracy(&data_set.train_images, &data_set.train_labels);
            train_result.train_acc_list.push(train_acc);
            println!("test images...");
            let test_acc = network.accuracy(&data_set.test_images, &data_set.test_labels);
            train_result.test_acc_list.push(test_acc);
            println!(
                "acc at {} step: train={}, test={}",
                ii + 1,
                train_acc,
                test_acc
            );
        }
    }
    println!("training finished.");

    // plot result
    let x: Vec<FF> = arange(0.0, train_result.train_loss_list.len() as FF, 1.0);
    assert_eq!(x.len(), train_result.train_loss_list.len());
    match plot(
        "./images/train_loss.png",
        480,
        640,
        &x,
        &train_result.train_loss_list,
        "train loss",
        "train loss",
    ) {
        Ok(_) => println!("ok"),
        Err(s) => println!("{}", s),
    }
    let x: Vec<FF> = arange(0.0, train_result.train_acc_list.len() as FF, 1.0);
    assert_eq!(x.len(), train_result.train_acc_list.len());
    match plot(
        "./images/train_acc.png",
        480,
        640,
        &x,
        &train_result.train_acc_list,
        "train accuracy",
        "train accuracy",
    ) {
        Ok(_) => println!("ok"),
        Err(s) => println!("{}", s),
    }
    match plot(
        "./images/test_acc.png",
        480,
        640,
        &x,
        &train_result.test_acc_list,
        "test accuracy",
        "test accuracy",
    ) {
        Ok(_) => println!("ok"),
        Err(s) => println!("{}", s),
    }
}

fn plot(
    file_name: &str,
    height: u32,
    width: u32,
    x: &[FF],
    y: &[FF],
    label: &str,
    caption: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(file_name, (width, height)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x[0]..x[x.len() - 1], -y.to_vec().min()..y.to_vec().max())?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new((0..x.len()).map(|ii| (x[ii], y[ii])), &RED))?
        .label(label)
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;
    Ok(())
}
