//! rs-deep
#![allow(dead_code)]

#[macro_use]
extern crate ndarray;

extern crate rs_deep;
// use rs_deep::dlfs01::common::math_ndarray::MathFunc;

use ndarray::{Array, Array2, ArrayD, Axis, IxDyn};
use ndarray_stats::QuantileExt;

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

    // commom
    // rs_deep::dlfs01::common::loss_function::main();
    // rs_deep::dlfs01::common::layers::activation::main();
    // rs_deep::dlfs01::common::layers::affine::main();
    // rs_deep::dlfs01::common::layers::softmax_with_loss::main();

    // dataset
    // rs_deep::dlfs01::dataset::main();

    // test
    test();
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
    let mut ii: usize = 0;
    for view in a.axis_iter(Axis(0)) {
        let y = view.slice(s![ii..(ii + 1), ..]);
        println!("{}, {}", ii, y);
        ii += 1;
    }
    let a: Array2<f32> = Array::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let a = a.into_dimensionality::<IxDyn>().unwrap();
    let argmax = a.argmax().unwrap();
    println!("{}", a);
    println!("{:?}, {}", argmax.clone(), a[argmax]);
    // let mut dst: Vec<f32> = Vec::new();
    // let mut ii: usize = 0;
    // let indices = [1usize, 0usize];
    // for index in indices.iter() {
    //     dst.append(&mut (1.0 * &a.index_axis(Axis(0), *index)).into_vec());
    // }
}
