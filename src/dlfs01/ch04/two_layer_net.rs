//! two_layer_net
//!
//! two-layer net

use crate::dlfs01::cast_t2u;
use crate::dlfs01::math;
use crate::dlfs01::operators;
use crate::dlfs01::LossFunc;
use crate::dlfs01::MathFunc;
use crate::dlfs01::Operators;
use num_traits::Float;

type Vec2d<T> = Vec<Vec<T>>;
type FF = f32;

const EPS: f64 = 1E-8;
const LOW: f64 = -1.0;
const HIGH: f64 = 1.0;

pub trait Model<T> {
    fn train(&self);
    fn predict_prob(&self, x: &Vec<T>) -> Vec<T>;
    fn predict(&self, x: &Vec<T>) -> Vec<T>;
    fn loss(&self, x: &Vec<T>, t: &Vec<T>) -> T;
    fn accuracy(&self, x: &Vec2d<T>, t: &Vec2d<T>) -> T;
    fn numerical_gradient(&mut self, x: &Vec<T>, t: &Vec<T>);
    // fn gradient(&mut self, x: &Vec<T>, t: &Vec<T>);
    fn gradient_by_batch(&mut self, x: &Vec2d<T>, t: &Vec2d<T>);
}

pub struct TwoLayerNet<T> {
    pub w1: Vec2d<T>,
    pub b1: Vec<T>,
    pub w2: Vec2d<T>,
    pub b2: Vec<T>,
    pub grad_w1: Vec2d<T>,
    pub grad_b1: Vec<T>,
    pub grad_w2: Vec2d<T>,
    pub grad_b2: Vec<T>,
}

impl<T> TwoLayerNet<T>
where
    T: Float,
{
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> TwoLayerNet<T> {
        let w1 = math::rand_uniform_2d(input_size, hidden_size, cast_t2u(LOW), cast_t2u(HIGH));
        let b1 = vec![cast_t2u(0.0); hidden_size];
        let w2 = math::rand_uniform_2d(hidden_size, output_size, cast_t2u(LOW), cast_t2u(HIGH));
        let b2 = vec![cast_t2u(0.0); output_size];
        let grad_w1 = vec![vec![cast_t2u(0.0); hidden_size]; input_size];
        let grad_b1 = vec![cast_t2u(0.0); hidden_size];
        let grad_w2 = vec![vec![cast_t2u(0.0); output_size]; hidden_size];
        let grad_b2 = vec![cast_t2u(0.0); output_size];
        TwoLayerNet {
            w1,
            b1,
            w2,
            b2,
            grad_w1,
            grad_b1,
            grad_w2,
            grad_b2,
        }
    }
}

impl<T> Model<T> for TwoLayerNet<T>
where
    T: Float,
{
    fn train(&self) {}
    fn predict_prob(&self, x: &Vec<T>) -> Vec<T> {
        let a = operators::dot_1d_2d(x, &self.w1).add(&self.b1);
        let a = operators::dot_1d_2d(&a.sigmoid(), &self.w2).add(&self.b2);
        a.softmax()
    }
    fn predict(&self, x: &Vec<T>) -> Vec<T> {
        let a = self.predict_prob(x);
        let a_max_ind = a.argmax();
        let zero: T = cast_t2u(0.0);
        let one: T = cast_t2u(1.0);
        (0..a.len())
            .map(|ii| if ii == a_max_ind { one } else { zero })
            .collect::<Vec<T>>()
    }
    fn loss(&self, x: &Vec<T>, t: &Vec<T>) -> T {
        self.predict_prob(x).cross_entropy_error(t)
    }
    fn accuracy(&self, x: &Vec2d<T>, t: &Vec2d<T>) -> T {
        let tol: T = cast_t2u(EPS);
        let mut dst: Vec<T> = Vec::new();
        let zero: T = cast_t2u(0.0);
        let one: T = cast_t2u(1.0);
        for ii in 0..x.len() {
            let pred = self.predict(&x[ii]);
            dst.push(if math::is_all_tol_1d(&pred, &t[ii], tol) {
                one
            } else {
                zero
            });
        }
        dst.mean()
    }
    fn numerical_gradient(&mut self, x: &Vec<T>, t: &Vec<T>) {
        let eps: T = cast_t2u(EPS);
        let eps2: T = cast_t2u(2.0 * EPS);
        for ii in 0..self.w1.len() {
            for jj in 0..self.w1[0].len() {
                self.w1[ii][jj] = self.w1[ii][jj] + eps;
                let fxh1 = self.loss(x, t);
                self.w1[ii][jj] = self.w1[ii][jj] - eps2;
                let fxh2 = self.loss(x, t);
                self.grad_w1[ii][jj] = (fxh1 - fxh2) / eps2;
                self.w1[ii][jj] = self.w1[ii][jj] + eps;
            }
        }
        for ii in 0..self.b1.len() {
            self.b1[ii] = self.b1[ii] + eps;
            let fxh1 = self.loss(x, t);
            self.b1[ii] = self.b1[ii] - eps2;
            let fxh2 = self.loss(x, t);
            self.grad_b1[ii] = (fxh1 - fxh2) / eps2;
            self.b1[ii] = self.b1[ii] + eps;
        }
        for ii in 0..self.w2.len() {
            for jj in 0..self.w2[0].len() {
                self.w2[ii][jj] = self.w2[ii][jj] + eps;
                let fxh1 = self.loss(x, t);
                self.w2[ii][jj] = self.w2[ii][jj] - eps2;
                let fxh2 = self.loss(x, t);
                self.grad_w2[ii][jj] = (fxh1 - fxh2) / eps2;
                self.w2[ii][jj] = self.w2[ii][jj] + eps;
            }
        }
        for ii in 0..self.b2.len() {
            self.b2[ii] = self.b2[ii] + eps;
            let fxh1 = self.loss(x, t);
            self.b2[ii] = self.b2[ii] - eps2;
            let fxh2 = self.loss(x, t);
            self.grad_b2[ii] = (fxh1 - fxh2) / eps2;
            self.b2[ii] = self.b2[ii] + eps;
        }
    }
    // fn gradient(&mut self, x: &Vec<T>, t: &Vec<T>) {
    //     // forward
    //     let a1 = operators::dot_2d_1d(&self.w1, x).add(&self.b1);
    //     let a2 = operators::dot_2d_1d(&self.w2, &a1.softmax()).add(&self.b2);
    //     let y = a2.softmax();
    // }
    fn gradient_by_batch(&mut self, x: &Vec2d<T>, t: &Vec2d<T>) {
        // forward
        let mut y: Vec2d<T> = Vec::new();
        let mut a1: Vec2d<T> = Vec::new();
        let mut z1: Vec2d<T> = Vec::new();
        for ii in 0..x.len() {
            let a1_ = operators::dot_1d_2d(&x[ii], &self.w1).add(&self.b1);
            let z1_ = a1_.sigmoid();
            let a2_ = operators::dot_1d_2d(&z1_, &self.w2).add(&self.b2);
            let y_ = a2_.softmax();
            a1.push(a1_);
            z1.push(z1_);
            y.push(y_);
        }

        // backward
        let norm_val: T = cast_t2u(x.len());
        let mut dy: Vec2d<T> = Vec::new();
        for ii in 0..x.len() {
            dy.push(y[ii].sub(&t[ii]).div_value(norm_val));
        }

        self.grad_w2 = operators::dot_2d_2d(&z1.transpose(), &dy);
        self.grad_b2 = dy.transpose().iter().map(|v| v.sum()).collect::<Vec<T>>();

        let dz1 = operators::dot_2d_2d(&dy, &self.w2.transpose());
        let da1 = a1.sigmoid_grad().mul(&dz1);
        self.grad_w1 = operators::dot_2d_2d(&x.transpose(), &da1);
        self.grad_b1 = da1.transpose().iter().map(|v| v.sum()).collect::<Vec<T>>();
    }
}

pub fn main() {
    println!("< two_layer_net sub module >");
    let mut net: TwoLayerNet<FF> = TwoLayerNet::new(2, 10, 3);
    let x: Vec<FF> = vec![0.6, 0.9];
    let t: Vec<FF> = vec![0.0, 0.0, 1.0];
    println!("predict_prob: {:?}", net.predict_prob(&x));
    println!("predict: {:?}", net.predict(&x));

    net.numerical_gradient(&x, &t);
    println!("numerical calculation of gradients.");
    println!("grad_w1: {:?}", net.grad_w1);
    println!("grad_b1: {:?}", net.grad_b1);
    println!("grad_w2: {:?}", net.grad_w2);
    println!("grad_b2: {:?}", net.grad_b2);

    net.gradient_by_batch(&vec![x; 1], &vec![t; 1]);
    println!("numerical calculation of gradients.");
    println!("grad_w1: {:?}", net.grad_w1);
    println!("grad_b1: {:?}", net.grad_b1);
    println!("grad_w2: {:?}", net.grad_w2);
    println!("grad_b2: {:?}", net.grad_b2);
}
