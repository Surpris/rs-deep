//! two_layer_net
//!
//! two-layer net

use crate::dlfs01::cast_t2u;
use crate::dlfs01::common::layers::*;
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use num_traits::Float;

pub trait Model<T> {
    // fn train(&self);
    fn predict_prob(&mut self, x: &Array2<T>) -> Array2<T>;
    fn predict(&mut self, x: &Array2<T>) -> Array2<T>;
    fn loss(&mut self, x: &Array2<T>, t: &Array2<T>) -> T;
    fn accuracy(&mut self, x: &Array2<T>, t: &Array2<T>) -> T;
    fn gradient(&mut self, x: &Array2<T>, t: &Array2<T>);
    fn update(&mut self, x: &Array2<T>, t: &Array2<T>, lr: T);
    fn print_detail(&self);
}

pub struct TwoLayerNet<T> {
    pub affine1: Affine<T>,
    pub activator: ReLU2<T>,
    pub affine2: Affine<T>,
    pub loss_layer: SoftmaxWithLoss2<T>,
    pub verbose: bool,
    pub current_loss: T,
}

impl<T: 'static> TwoLayerNet<T>
where
    T: Float,
{
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        batch_axis: usize,
    ) -> Self {
        TwoLayerNet {
            affine1: Affine::new((input_size, hidden_size)),
            activator: ReLU2::new((hidden_size, hidden_size)),
            affine2: Affine::new((hidden_size, output_size)),
            loss_layer: SoftmaxWithLoss2::new((output_size, output_size), batch_axis),
            verbose: false,
            current_loss: cast_t2u(0.0),
        }
    }
}

impl<T: 'static> Model<T> for TwoLayerNet<T>
where
    T: Float + std::fmt::Display,
{
    fn predict_prob(&mut self, x: &Array2<T>) -> Array2<T> {
        let y = self.affine1.forward(&x);
        let y = self.activator.forward(&y);
        let y = self.affine2.forward(&y);
        y
    }
    fn predict(&mut self, x: &Array2<T>) -> Array2<T> {
        let one: T = cast_t2u(1.0);
        let y: Array2<T> = self.predict_prob(&x);
        let mut dst: Array2<T> = Array2::zeros(y.raw_dim());
        for (view1, mut view2) in y.axis_iter(Axis(0)).zip(dst.axis_iter_mut(Axis(0))) {
            let y_argmax = view1.argmax().unwrap();
            view2[y_argmax] = one;
        }
        dst
    }
    fn loss(&mut self, x: &Array2<T>, t: &Array2<T>) -> T {
        let y = self.predict_prob(&x);
        self.current_loss = self.loss_layer.forward(&y, &t);
        self.current_loss
    }
    fn accuracy(&mut self, x: &Array2<T>, t: &Array2<T>) -> T {
        let y: Array2<T> = self.predict(&x);
        let mut acc: f32 = 0.0;
        for (view1, view2) in y.axis_iter(Axis(0)).zip(t.axis_iter(Axis(0))) {
            let y_argmax = view1.argmax().unwrap();
            let t_argmax = view2.argmax().unwrap();
            if y_argmax == t_argmax {
                acc += 1.0;
            }
        }
        cast_t2u(acc / t.len_of(Axis(0)) as f32)
    }
    fn gradient(&mut self, x: &Array2<T>, t: &Array2<T>) {
        // forward
        let _ = self.loss(&x, &t);

        // backward
        let dx: T = cast_t2u(1.0);
        let dx = self.loss_layer.backward(dx);
        let dx = self.affine2.backward(&dx);
        let dx = self.activator.backward(&dx);
        let _dx = self.affine1.backward(&dx);
    }
    fn update(&mut self, x: &Array2<T>, t: &Array2<T>, lr: T) {
        self.gradient(&x, &t);
        self.affine2.update(lr);
        self.activator.update(lr);
        self.affine1.update(lr);
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
    println!("< ch05 two_layer_net sub module >");
    let mut net: TwoLayerNet<f32> = TwoLayerNet::new(2, 10, 3, 0);
    net.print_detail();
    let x: Array2<f32> = Array::from_shape_vec((1, 2), vec![0.6, 0.9]).unwrap();
    let t: Array2<f32> = Array::from_shape_vec((1, 3), vec![0.0, 0.0, 1.0]).unwrap();
    println!("features: {}", x);
    println!("target: {}", t);
    println!("predict_prob: {:?}", net.predict_prob(&x));
    println!("predict: {:?}", net.predict(&x));
    println!("accuracy: {}", net.accuracy(&x, &t));
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
