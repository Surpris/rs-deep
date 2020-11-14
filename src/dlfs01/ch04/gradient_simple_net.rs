//! gradient_simple_net
//!
//! test calculating gradient of simple network

use crate::dlfs01::cast_t2u;
use crate::dlfs01::gradient::numerical_gradient_2d;
use crate::dlfs01::LossFunc;
use crate::dlfs01::MathFunc;
use num_traits::Float;
use rand::distributions::Uniform;
use rand::prelude::*;

pub trait Model<T> {
    fn train(&self);
    fn predict(&self, x: &Vec<T>) -> Vec<T>;
    fn loss(&self, x: &Vec<T>, t: &Vec<T>) -> T;
}

struct SimpleNet<T> {
    weight: Vec<Vec<T>>,
}

impl<T> SimpleNet<T>
where
    T: Float,
{
    fn new() -> SimpleNet<T> {
        let mut rng = rand::thread_rng();
        let gen = Uniform::new(-10.0, 10.0);
        let mut a: Vec<Vec<T>> = vec![vec![cast_t2u(0.0); 3]; 2];
        for ii in 0..a.len() {
            for jj in 0..a[ii].len() {
                a[ii][jj] = cast_t2u::<f64, T>(gen.sample(&mut rng));
            }
        }
        SimpleNet { weight: a }
    }
}

impl<T> Model<T> for SimpleNet<T>
where
    T: Float,
{
    fn train(&self) {}
    fn predict(&self, x: &Vec<T>) -> Vec<T> {
        let mut dst: Vec<T> = Vec::new();
        for jj in 0..self.weight[0].len() {
            dst.push(
                (0..self.weight.len())
                    .map(|ii| self.weight[ii][jj] * x[ii])
                    .collect::<Vec<T>>()
                    .sum(),
            );
        }
        dst
    }
    fn loss(&self, x: &Vec<T>, t: &Vec<T>) -> T {
        self.predict(x).softmax_loss(t)
    }
}

pub fn main() {
    println!("< gradient_simple_net sub module >");
    let x: [f32; 2] = [0.6, 0.9];
    let t: [f32; 3] = [0.0, 0.0, 1.0];
    let net: SimpleNet<f32> = SimpleNet::new();
    let f = |_: &Vec<f32>| net.loss(&x.to_vec(), &t.to_vec());
    let mut tt: Vec<Vec<f32>> = net.weight.clone();
    let dw = numerical_gradient_2d(&f, &mut tt);
    println!("{:?}", dw);
}
