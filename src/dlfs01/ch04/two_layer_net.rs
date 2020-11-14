//! two_layer_net
//!
//! two-layer net

use crate::dlfs01::cast_t2u;
// use crate::dlfs01::gradient::numerical_gradient_2d;
// use crate::dlfs01::LossFunc;
// use crate::dlfs01::MathFunc;
use crate::dlfs01::math;
use num_traits::Float;
use rand::distributions::Uniform;
use rand::prelude::*;

type Vec2d<T> = Vec<Vec<T>>;

const LOW: f64 = -10.0;
const HIGH: f64 = 10.0;

pub trait Model<T> {
    fn train(&self);
    fn predict(&self, x: &Vec<T>) -> Vec<T>;
    fn loss(&self, x: &Vec<T>, t: &Vec<T>) -> T;
    fn accuracy(&self, x: &Vec<T>, t: &Vec<T>) -> T;
    fn numerical_gradient(&self, x: &Vec<T>, t: &Vec<T>) -> Vec2d<T>;
    fn gradient(&self, x: &Vec<T>, t: &Vec<T>) -> Vec2d<T>;
}

struct TwoLayerNet<T> {
    w1: Vec2d<T>,
    b1: Vec<T>,
    w2: Vec2d<T>,
    b2: Vec<T>,
}

impl<T> TwoLayerNet<T>
where
    T: Float,
{
    fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        weight_init_std: T,
    ) -> TwoLayerNet<T> {
        let w1 = math::rand_uniform_2d(input_size, hidden_size, cast_t2u(LOW), cast_t2u(HIGH));
        let w2 = math::rand_uniform_2d(input_size, hidden_size, cast_t2u(LOW), cast_t2u(HIGH));
        let b1 = vec![cast_t2u(0.0); hidden_size];
        let b2 = vec![cast_t2u(0.0); output_size];
        TwoLayerNet { w1, b1, w2, b2 }
    }
}

// impl<T> Model<T> for TwoLayerNet<T> where T: Float {
//     fn train(&self) {

//     }
//     fn predict(&self, x: &Vec<T>) -> Vec<T> {}
//     fn loss(&self, x: &Vec<T>, t: &Vec<T>) -> T {}
// }
