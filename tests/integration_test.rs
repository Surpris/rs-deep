//! integration_test
//!
//! test functions

extern crate rs_deep;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_stats::QuantileExt;
use rand::distributions::Uniform;
use rand::prelude::*;
use rs_deep::prelude::*;

#[test]
fn test_activation() {
    println!("< activation sub module> ");
    let mut rng = rand::thread_rng();
    let gen = Uniform::new(-1.0f32, 1.0f32);

    let a = ArrayD::<f32>::zeros(IxDyn(&[2, 3]));
    let a = a.map(|_| gen.sample(&mut rng));

    println!("ReLU layer");
    let mut layer = ReLUD::<f32>::new(a.shape());
    let b = layer.forward(&a);
    let da = ArrayD::<f32>::zeros(IxDyn(a.shape())).map(|_| gen.sample(&mut rng));
    println!("a: {}", a);
    println!("forward: {}", b);
    println!("da: {}", da);
    println!("backward: {}", layer.backward(&da));

    println!("sigmoid layer");
    let mut layer = SigmoidD::<f32>::new(a.shape());
    let b = layer.forward(&a);
    let da = ArrayD::<f32>::zeros(IxDyn(a.shape())).map(|_| gen.sample(&mut rng));
    println!("a: {}", a);
    println!("forward: {}", b);
    println!("da: {}", da);
    println!("backward: {}", layer.backward(&da));

    println!("softmax layer");
    let mut layer = SoftmaxD::<f32>::new(a.shape(), 0);
    let b = layer.forward(&a);
    let da = ArrayD::<f32>::zeros(IxDyn(a.shape())).map(|_| gen.sample(&mut rng));
    println!("a: {}", a);
    println!("forward: {}, {}", b, b.sum_axis(Axis(1)));
    println!("da: {}", da);
    println!("backward: {}", layer.backward(&da));
}

#[test]
pub fn test_affine() {
    println!("< affine sub module> ");
    let mut rng = rand::thread_rng();
    let gen = Uniform::new(-1.0f32, 1.0f32);

    let input_shape: (usize, usize) = (2, 3);
    let a: Array2<f32> =
        Array::from_shape_vec(input_shape, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]).unwrap();

    let affine_shape = (input_shape.1, 2);
    let w: Array2<f32> =
        Array::from_shape_vec(affine_shape, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b: Array1<f32> = Array::linspace(1.0, 3.0, affine_shape.1);
    let mut layer = Affine::<f32>::from(&w, &b);
    layer.print_detail();

    let y = layer.forward(&a);

    let output_shape = (input_shape.0, affine_shape.1);
    let da = Array2::<f32>::zeros(output_shape).map(|_| gen.sample(&mut rng));
    println!("a: {}", a);
    println!("affine forward: {}", y);
    println!("da: {}", da);
    println!("affine backward: {}", layer.backward(&da));
}

#[test]
pub fn test_softmax_with_loss() {
    println!("< softmax-with-loss sub module >");
    let mut rng = rand::thread_rng();
    let gen = Uniform::new(-1.0f32, 1.0f32);

    let a = ArrayD::<f32>::zeros(IxDyn(&[2, 3]));
    let a = a.map(|_| gen.sample(&mut rng));

    let mut t = ArrayD::<f32>::zeros(IxDyn(a.shape()));
    t[[0, 0]] = 1.0;
    t[[1, 1]] = 1.0;

    println!("softmax-with-loss layer");
    let mut layer: SoftmaxWithLossD<f32> = SoftmaxWithLossD::<f32>::new(a.shape(), 0);
    let b = layer.forward(&a, &t);
    println!("a: {}", a);
    println!("t: {}", t);
    println!("output: {}", layer.get_output());
    println!("forward (loss): {}", b);
    println!("backward: {}", layer.backward(1.0));
}

#[test]
pub fn test_mlp() {
    println!("< ch06 mlp sub module >");
    let mut net: MLPClassifier<f32> = MLPClassifier::new(
        2,
        &[10],
        3,
        &[ActivatorEnum::ReLU],
        OptimizerEnum::SGD,
        &[1.0],
        0,
    );
    net.print_detail();
    let x: Array2<f32> = Array::from_shape_vec((1, 2), vec![0.6, 0.9]).unwrap();
    let t: Array2<f32> = Array::from_shape_vec((1, 3), vec![0.0, 0.0, 1.0]).unwrap();
    println!("features: {}", x);
    println!("target: {}", t);
    println!("predict_prob: {:?}", net.predict_prob(&x));
    println!("predict: {:?}", net.predict(&x));
    println!("accuracy: {}", net.accuracy(&x, &t));
    println!("loss: {}", net.loss(&x, &t));
    println!("output: {}", net.get_output());

    net.gradient(&x, &t);
    println!("numerical calculation of gradients.");
}

#[test]
pub fn test_ndarray_random() {
    let a: Array2<f64> = Array::random((2, 5), Uniform::new(0., 10.));
    println!("{:8.4}", a);
}

#[test]
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
