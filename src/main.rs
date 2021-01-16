//! rs-deep

#![allow(dead_code)]

// #[macro_use]
extern crate ndarray;

extern crate rs_deep;
use rs_deep::prelude::*;
use serde_json;

// use std::collections::HashMap;
// use num_traits::Float;
// use ndarray::prelude::*;
// use ndarray_rand::rand_distr::Uniform;
// use ndarray_rand::RandomExt;
// use ndarray_stats::QuantileExt;
// use std::time::Instant;

// macro_rules! measure {
//     ( $x:expr) => {{
//         let start = Instant::now();
//         let result = $x;
//         let end = start.elapsed();
//         println!("{}.{:03} sec elapsed.", end.as_secs(), end.subsec_millis());
//         result
//     }};
// }

fn main() {
    // chapter 01
    // rs_deep::dlfs01::ch01::hungry::main();
    // rs_deep::dlfs01::ch01::man::main();
    // rs_deep::dlfs01::ch01::single_graph::main();

    // chapter 02
    // rs_deep::dlfs01::ch02::gates::main();

    // chapter 03
    // rs_deep::dlfs01::ch03::activation::main();

    // chapter 04
    // rs_deep::dlfs01::ch04::gradient_1d::main();
    // rs_deep::dlfs01::ch04::gradient_2d::main();
    // rs_deep::dlfs01::ch04::gradient_method::main();
    // rs_deep::dlfs01::ch04::gradient_simple_net::main();
    // rs_deep::dlfs01::ch04::two_layer_net::main();
    // rs_deep::dlfs01::ch04::train_neural_net::main();

    // chapter 05
    // rs_deep::dlfs01::ch05::buy_apple::main();
    // rs_deep::dlfs01::ch05::buy_apple_orange::main();
    // rs_deep::dlfs01::ch05::two_layer_net::main();
    // rs_deep::dlfs01::ch05::train_neural_net::main();

    // chapter 06
    // rs_deep::dlfs01::ch06::optimizer_compare_mnist::main();
    // rs_deep::dlfs01::ch06::weight_init_compare::main();
    // rs_deep::dlfs01::ch06::batch_norm_test::main();

    // common
    // rs_deep::dlfs01::common::models::mlp::main();

    // dataset
    // rs_deep::dlfs01::dataset::mnist_vec::main();

    // examples
    rs_deep::dlfs01::examples::train_mlp_test::main();

    // common
    // rs_deep::dlfs01::common::models::mlp::main();

fn test() {
    let a: ModelParameters<f32> = ModelParameters::new();
    println!("{:?}", a);
    let b = serde_json::to_string(&a).unwrap();
    println!("{}", b);
    let c: ModelParameters<f32> = serde_json::from_str(&b).unwrap();
    println!("{:?}", c);
}
