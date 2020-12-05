//! train_neural_net
//!
//! train a two-layer model

#![allow(dead_code)]
#![allow(unused_imports)]

use super::two_layer_net::{Model, TwoLayerNet};
use crate::dlfs01::common::choice::Choice;
use crate::dlfs01::dataset::{load_mnist, MNISTDataSetArray2};
use crate::dlfs01::math::arange;
use crate::dlfs01::MathFunc;
use crate::dlfs01::Operators;
use ndarray::{Array, Array2, Axis};
use plotters::prelude::*;
use rand::prelude::*;
use std::time::Instant;

const HIDDEN_SIZE: usize = 50;
const NBR_OF_ITERS: usize = 10000;
const BATCH_SIZE: usize = 100;
const LEARNING_RATE: f32 = 0.1;
const NBR_OF_TARGET_IMAGES: usize = 60000;

struct TrainResult {
    train_loss_list: Vec<f32>,
    train_acc_list: Vec<f32>,
    test_acc_list: Vec<f32>,
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
    println!("< ch05 train_neural_net sub module >");
    // load MNIST dataset
    println!("load MNIST dataset...");
    let data_set: MNISTDataSetArray2<f32> = load_mnist(0u8).unwrap().into_array2();
    let input_size = data_set.train_images.len_of(Axis(1));
    let output_size = data_set.train_labels.len_of(Axis(1));
    let data_set: MNISTDataSetArray2<f32> = MNISTDataSetArray2 {
        train_images: 1.0 * &data_set.train_images.slice(s![..NBR_OF_TARGET_IMAGES, ..]),
        train_labels: 1.0 * &data_set.train_labels.slice(s![..NBR_OF_TARGET_IMAGES, ..]),
        test_images: data_set.test_images,
        test_labels: data_set.test_labels,
    };

    // set a parameter for training
    let nbr_train_images: usize = data_set.train_images.len_of(Axis(0));
    let iter_per_epoch: usize = usize::max(nbr_train_images / BATCH_SIZE, 1);

    // initialize a two-layer model
    println!("initialize a model...");
    let mut network: TwoLayerNet<f32> = TwoLayerNet::new(input_size, HIDDEN_SIZE, output_size);
    network.print_detail();
    network.verbose = false;

    // initialize a TrainResult instance
    let mut train_result: TrainResult = TrainResult::new();

    // train loop
    let mut rng = thread_rng();
    println!("start training...");
    let start = Instant::now();
    for ii in 0..NBR_OF_ITERS {
        // choose indices
        let mut indices: Vec<usize> = vec![0usize; BATCH_SIZE];
        for jj in 0..BATCH_SIZE {
            indices[jj] = rng.gen_range(0, nbr_train_images);
        }
        let x_batch = data_set.train_images.select(Axis(0), &indices);
        let t_batch = data_set.train_labels.select(Axis(0), &indices);

        // update parameters of the network
        network.update(&x_batch, &t_batch, LEARNING_RATE);

        // calculate loss
        train_result.train_loss_list.push(network.current_loss);

        // validation
        if ii % iter_per_epoch == 0 {
            println!("train loss at step {}: {}", ii, network.current_loss);
            // print!("validation: ");
            // print!("train images... ");
            // let train_acc = network.accuracy(&data_set.train_images, &data_set.train_labels);
            // train_result.train_acc_list.push(train_acc);
            // println!("test images...");
            // let test_acc = network.accuracy(&data_set.test_images, &data_set.test_labels);
            // train_result.test_acc_list.push(test_acc);
            // println!(
            //     "acc at {} step: train={}, test={}",
            //     ii,
            //     train_acc,
            //     test_acc
            // );
        }
    }
    print!("validation... ");
    let train_acc = network.accuracy(&data_set.train_images, &data_set.train_labels);
    let test_acc = network.accuracy(&data_set.test_images, &data_set.test_labels);
    println!("acc : train={}, test={}", train_acc, test_acc);
    let end = start.elapsed();
    println!(
        "{}.{:03} sec elapsed to training.",
        end.as_secs(),
        end.subsec_millis()
    );
    println!("training finished.");

    // plot result
    // let x: Vec<f32> = arange(0.0, train_result.train_loss_list.len() as f32, 1.0);
    // assert_eq!(x.len(), train_result.train_loss_list.len());
    // match plot(
    //     "./images/train_loss.png",
    //     480,
    //     640,
    //     &x,
    //     &train_result.train_loss_list,
    //     "train loss",
    //     "train loss",
    // ) {
    //     Ok(_) => println!("ok"),
    //     Err(s) => println!("{}", s),
    // }
    // let x: Vec<f32> = arange(0.0, train_result.train_acc_list.len() as f32, 1.0);
    // assert_eq!(x.len(), train_result.train_acc_list.len());
    // match plot(
    //     "./images/train_acc.png",
    //     480,
    //     640,
    //     &x,
    //     &train_result.train_acc_list,
    //     "train accuracy",
    //     "train accuracy",
    // ) {
    //     Ok(_) => println!("ok"),
    //     Err(s) => println!("{}", s),
    // }
    // match plot(
    //     "./images/test_acc.png",
    //     480,
    //     640,
    //     &x,
    //     &train_result.test_acc_list,
    //     "test accuracy",
    //     "test accuracy",
    // ) {
    //     Ok(_) => println!("ok"),
    //     Err(s) => println!("{}", s),
    // }
}

fn plot(
    file_name: &str,
    height: u32,
    width: u32,
    x: &[f32],
    y: &[f32],
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
