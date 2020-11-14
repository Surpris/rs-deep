//! rs-deep

extern crate rs_deep;
use rs_deep::dlfs01::common::loss_function::LossFunc;

fn main() {
    // chapter 01
    rs_deep::dlfs01::ch01::hungry::main();
    rs_deep::dlfs01::ch01::man::main();
    // rs_deep::dlfs01::ch01::single_graph::main();

    // chapter 02
    rs_deep::dlfs01::ch02::gates::main();

    // chapter 03
    // rs_deep::dlfs01::ch03::activation::main();

    // chapter 04
    // rs_deep::dlfs01::ch04::gradient_1d::main();
    rs_deep::dlfs01::ch04::gradient_2d::main();
    // rs_deep::dlfs01::ch04::gradient_method::main();
    rs_deep::dlfs01::ch04::gradient_simple_net::main();

    // commom
    let v: Vec<f32> = vec![1.0; 10];
    let v2: Vec<f32> = vec![0.5; 10];
    println!("{:?}", v.sum_squared_error(&v2));
}
