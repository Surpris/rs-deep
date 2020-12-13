//! rs-deep

#![allow(dead_code)]

// #[macro_use]
extern crate ndarray;

extern crate rs_deep;
use rs_deep::prelude::*;

use std::collections::HashMap;
// use num_traits::Float;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ndarray_stats::QuantileExt;
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
    rs_deep::dlfs01::ch05::train_neural_net::main();

    // dataset
    // rs_deep::dlfs01::dataset::mnist_vec::main();

    // test
    // test();
    // test2();
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
}

fn test2() {
    let mut hash_map: HashMap<usize, ArrayEnum<f64>> = HashMap::new();
    hash_map.insert(0, ArrayEnum::Array1(Array1::zeros(2)));
    hash_map.insert(1, ArrayEnum::Array2(Array2::zeros((2, 2))));
    hash_map.insert(2, ArrayEnum::Array3(Array3::zeros((2, 2, 2))));

    for (k, v) in hash_map.iter() {
        match v {
            ArrayEnum::Array1(x) => println!("{}, {}", k, x),
            ArrayEnum::Array2(x) => println!("{}, {}", k, x),
            ArrayEnum::Array3(x) => println!("{}, {}", k, x),
            ArrayEnum::Array4(x) => println!("{}, {}", k, x),
            ArrayEnum::Array5(x) => println!("{}, {}", k, x),
            ArrayEnum::Array6(x) => println!("{}, {}", k, x),
            ArrayEnum::ArrayD(x) => println!("{}, {}", k, x),
        }
    }
}

fn test3() {
    let a = Array::random((2, 5), Uniform::new(0., 10.));
    println!("{:8.4}", a);
}
