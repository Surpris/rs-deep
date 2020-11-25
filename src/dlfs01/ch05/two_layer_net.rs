//! two_layer_net
//!
//! two-layer net

use crate::dlfs01::cast_t2u;
use crate::dlfs01::common::layers::*;
use ndarray::{Array, Array2, Ix2, IxDyn};
use num_traits::Float;

pub trait Model<T> {
    // fn train(&self);
    fn predict_prob(&mut self, x: &Array2<T>) -> Array2<T>;
    fn predict(&mut self, x: &Array2<T>) -> Array2<T>;
    fn loss(&mut self, x: &Array2<T>, t: &Array2<T>) -> T;
    fn accuracy(&mut self, x: &Array2<T>, t: &Array2<T>) -> T;
    fn gradient(&mut self, x: &Array2<T>, t: &Array2<T>);
    fn print_detail(&self);
}

pub struct TwoLayerNet<T> {
    pub affine1: Affine<T>,
    pub activator: ReLU<T>,
    pub affine2: Affine<T>,
    pub loss_layer: SoftmaxWithLoss<T>,
    pub verbose: bool,
}

impl<T: 'static> TwoLayerNet<T>
where
    T: Float,
{
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        TwoLayerNet {
            affine1: Affine::new(&(input_size, hidden_size)),
            activator: ReLU::new(&[hidden_size]),
            affine2: Affine::new(&(hidden_size, output_size)),
            loss_layer: SoftmaxWithLoss::new(&[output_size]),
            verbose: false,
        }
    }
}

impl<T: 'static> Model<T> for TwoLayerNet<T>
where
    T: Float + std::fmt::Display,
{
    fn predict_prob(&mut self, x: &Array2<T>) -> Array2<T> {
        let y = self.affine1.forward(&x);
        let y = self
            .activator
            .forward(&y.into_dimensionality::<IxDyn>().unwrap());
        let y = self
            .affine2
            .forward(&y.into_dimensionality::<Ix2>().unwrap());
        y
    }
    fn predict(&mut self, x: &Array2<T>) -> Array2<T> {
        x.clone()
    }
    fn loss(&mut self, x: &Array2<T>, t: &Array2<T>) -> T {
        let y = self.predict_prob(&x);
        self.loss_layer.forward(
            &y.into_dimensionality::<IxDyn>().unwrap(),
            &t.clone().into_dimensionality::<IxDyn>().unwrap(),
        )
    }
    fn accuracy(&mut self, x: &Array2<T>, t: &Array2<T>) -> T {
        let y = self.predict(&x);
        (y - t).sum()
    }
    fn gradient(&mut self, x: &Array2<T>, t: &Array2<T>) {
        // forward
        self.loss(&x, &t);

        // backward
        let dx: T = cast_t2u(1.0);
        let dx = self.loss_layer.backward(dx);
        println!("1: {}", dx);
        let dx = self
            .affine2
            .backward(&dx.into_dimensionality::<Ix2>().unwrap());
        println!("2: {}", dx);
        let dx = self
            .activator
            .backward(&dx.into_dimensionality::<IxDyn>().unwrap());
        println!("3: {}", dx);
        let _dx = self
            .affine1
            .backward(&dx.into_dimensionality::<Ix2>().unwrap());
    }
    fn print_detail(&self) {
        println!("Two-layer net.");
        self.affine1.print_detail();
        self.activator.print_detail();
        self.affine2.print_detail();
        self.loss_layer.print_detail();
    }
}

pub fn main() {
    println!("< two_layer_net sub module >");
    let mut net: TwoLayerNet<f32> = TwoLayerNet::new(2, 10, 3);
    net.print_detail();
    let x: Array2<f32> = Array::from_shape_vec((1, 2), vec![0.6, 0.9]).unwrap();
    let t: Array2<f32> = Array::from_shape_vec((1, 3), vec![0.0, 0.0, 1.0]).unwrap();
    println!("predict_prob: {:?}", net.predict_prob(&x));
    println!("predict: {:?}", net.predict(&x));
    println!("loss: {}", net.loss(&x, &t));
    println!("output: {}", net.loss_layer.output);

    net.gradient(&x, &t);
    println!("numerical calculation of gradients.");
    println!("w1: {:?}", net.affine1.weight);
    println!("b1: {:?}", net.affine1.bias);
    println!("w2: {:?}", net.affine2.weight);
    println!("b2: {:?}", net.affine2.bias);
    println!("grad_w1: {:?}", net.affine1.dw);
    println!("grad_b1: {:?}", net.affine1.db);
    println!("grad_w2: {:?}", net.affine2.dw);
    println!("grad_b2: {:?}", net.affine2.db);
}
