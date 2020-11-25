//! rs-deep
#![allow(dead_code)]

extern crate rs_deep;

use ndarray::{ArrayD, Axis, IxDyn};

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
    rs_deep::dlfs01::ch04::two_layer_net::main();
    // rs_deep::dlfs01::ch04::train_neural_net::main();

    // chapter 05
    // rs_deep::dlfs01::ch05::buy_apple::main();
    // rs_deep::dlfs01::ch05::buy_apple_orange::main();
    rs_deep::dlfs01::ch05::two_layer_net::main();

    // commom
    // rs_deep::dlfs01::common::loss_function::main();
    // rs_deep::dlfs01::common::layers::activation::main();
    rs_deep::dlfs01::common::layers::affine::main();
    // rs_deep::dlfs01::common::layers::softmax_with_loss::main();

    // dataset
    // rs_deep::dlfs01::dataset::main();

    // test
    // test();
}

fn test() {
    println!("test code");
    let a = ArrayD::<f32>::zeros(IxDyn(&[2, 3]));
    let b = a.map(|&v| if v == 0.0 { 1.0 } else { 0.0 });
    for v in a.indexed_iter() {
        println!("{:?}, {}, {}", v.0, v.1, b[v.0.clone()]);
    }

    let a = ArrayD::<f32>::ones(IxDyn(&[2, 3, 4]));
    for ax in a.axis_iter(Axis(1)) {
        println!("{:?}, {}", ax.shape(), ax.sum());
    }
}
