//! softmax_with_loss
//!
//! custom layer: combination of softmax and loss

use super::super::util::cast_t2u;
// use super::activation::Softmax;
use super::layer_base::*;
use itertools::multizip;
use ndarray::{prelude::*, RemoveAxis};
use num_traits::Float;
// use rand::distributions::Uniform;
// use rand::prelude::*;
use std::f64::consts::E;

const EPS: f64 = 1E-8;

/// Arbitrary-D softmax-with-loss layer
pub struct SoftmaxWithLoss<T, D> {
    pub output: Array<T, D>,
    axis: usize,
    target: Array<T, D>,
    loss: T,
}

impl<T: 'static, D> SoftmaxWithLoss<T, D>
where
    T: Float,
    D: Dimension,
{
    pub fn new<Sh>(shape: Sh, axis: usize) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        let zeros = Array::<T, D>::zeros(shape);
        Self {
            output: zeros.clone(),
            axis,
            target: zeros,
            loss: cast_t2u(0.0),
        }
    }
}

impl<T: 'static, D> LossLayerBase<T, D> for SoftmaxWithLoss<T, D>
where
    T: Float,
    D: Dimension + RemoveAxis,
{
    fn forward(&mut self, x: &Array<T, D>, t: &Array<T, D>) -> T {
        let zero: T = cast_t2u(0.0);
        let eps: T = cast_t2u(EPS);
        let e: T = cast_t2u(E);
        let batch_size: T = cast_t2u(x.len_of(Axis(0)));

        self.output = x.clone();
        for mut view in self.output.axis_iter_mut(Axis(self.axis)) {
            let v_max = view.fold(zero / zero, |m, &v| v.max(m));
            view.mapv_inplace(|v| T::exp(v - v_max));
            let v_exp_sum = view.sum();
            view.mapv_inplace(|v| v / v_exp_sum);
        }
        // self.output = dst.clone();
        self.target = t.clone();
        self.loss = self
            .output
            .iter()
            .zip(t.iter())
            .fold(zero, |m, (o_, t_)| m - *t_ * (*o_ + eps).log(e));
        self.loss / batch_size
    }
    fn backward(&mut self, _dx: T) -> Array<T, D> {
        let batch_size: T = cast_t2u(self.target.len_of(Axis(0)));
        let mut dst = Array::<T, D>::zeros(self.target.raw_dim());
        for (t, d, o) in multizip((self.target.iter(), dst.iter_mut(), self.output.iter())) {
            *d = (*o - *t) / batch_size;
        }
        dst
    }
    fn print_detail(&self) {
        println!("softmax-with-loss layer.");
    }
}

pub type SoftmaxWithLoss2<T> = SoftmaxWithLoss<T, Ix2>;
pub type SoftmaxWithLoss3<T> = SoftmaxWithLoss<T, Ix3>;
pub type SoftmaxWithLoss4<T> = SoftmaxWithLoss<T, Ix4>;
pub type SoftmaxWithLoss5<T> = SoftmaxWithLoss<T, Ix5>;
pub type SoftmaxWithLoss6<T> = SoftmaxWithLoss<T, Ix6>;
pub type SoftmaxWithLossD<T> = SoftmaxWithLoss<T, IxDyn>;
