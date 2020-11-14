//! gradient_simple_net
//!
//! test calculating gradient of simple network

use crate::dlfs01::cast_t2u;
use crate::dlfs01::math;
use crate::dlfs01::LossFunc;
use crate::dlfs01::MathFunc;
use num_traits::Float;

type Vec2d<T> = Vec<Vec<T>>;
const EPS: f64 = 1E-4;
const LOW: f64 = -10.0;
const HIGH: f64 = 10.0;

pub trait Model<T> {
    fn train(&self);
    fn predict(&self, x: &Vec<T>) -> Vec<T>;
    fn loss(&self, x: &Vec<T>, t: &Vec<T>) -> T;
    fn numerical_gradient(&mut self, x: &Vec<T>, t: &Vec<T>) -> Vec2d<T>;
}

#[derive(Clone)]
struct SimpleNet<T> {
    weight: Vec2d<T>,
}

impl<T> SimpleNet<T>
where
    T: Float,
{
    fn new() -> SimpleNet<T> {
        let weight: Vec2d<T> = math::rand_uniform_2d(2, 3, cast_t2u(LOW), cast_t2u(HIGH));
        SimpleNet { weight }
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
    fn numerical_gradient(&mut self, x: &Vec<T>, t: &Vec<T>) -> Vec2d<T> {
        let eps: T = cast_t2u(EPS);
        let eps2: T = cast_t2u(2.0 * EPS);
        let mut grad: Vec2d<T> = vec![vec![cast_t2u(0.0); self.weight[0].len()]; self.weight.len()];
        for ii in 0..self.weight.len() {
            for jj in 0..self.weight[0].len() {
                self.weight[ii][jj] = self.weight[ii][jj] + eps;
                let fxh1 = self.loss(x, t);
                self.weight[ii][jj] = self.weight[ii][jj] - eps2;
                let fxh2 = self.loss(x, t);
                grad[ii][jj] = (fxh1 - fxh2) / eps2;
                self.weight[ii][jj] = self.weight[ii][jj] + eps;
            }
        }
        grad
    }
}

pub fn main() {
    println!("< gradient_simple_net sub module >");
    let x: Vec<f32> = vec![0.6, 0.9];
    let t: Vec<f32> = vec![0.0, 0.0, 1.0];
    let mut net: SimpleNet<f32> = SimpleNet::new();
    let dw = net.numerical_gradient(&x, &t);
    println!("{:?}", dw);
}
