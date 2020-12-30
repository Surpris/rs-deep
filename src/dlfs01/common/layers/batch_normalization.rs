//! batch_normalization
//!
//! BatchNormalization layer

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use super::super::util::*;
use super::layer_base::LayerBase;
use itertools::multizip;
use ndarray::{prelude::*, RemoveAxis};
use std::fmt::{Debug, Display};

const EPS: f64 = 1E-8;

/// BatchNormalization
///
/// See http://arxiv.org/abs/1502.03167 in detail
pub struct BatchNormalization<T: CrateFloat, D: Dimension + RemoveAxis> {
    momentum: T,
    batch_axis: usize,
    batch_size: usize,
    gamma: Array<T, D::Smaller>,
    beta: Array<T, D::Smaller>,
    dgamma: Array<T, D::Smaller>,
    dbeta: Array<T, D::Smaller>,
    xc: Array<T, D>,
    xn: Array<T, D>,
    std: Array<T, D::Smaller>,
    trained: bool,
    eps: T,
    batch_size_t: T,
    one: T,
    two: T,
    half: T,
    one_minus_m: T,
    running_mean: Array<T, D::Smaller>,
    running_var: Array<T, D::Smaller>,
}

impl<T: 'static, D> BatchNormalization<T, D>
where
    T: CrateFloat,
    D: Dimension + RemoveAxis,
{
    pub fn new<Sh>(momentum: T, batch_axis: usize, shape: Sh) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        let one: T = cast_t2u(1.0);
        let two: T = cast_t2u(2.0);
        let half: T = cast_t2u(0.5);
        let xn: Array<T, D> = Array::<T, D>::zeros(shape);
        // let zeros: Array<T, D::Smaller> = xn.clone().index_axis_move(Axis(batch_axis), 0);
        // let ones: Array<T, D::Smaller> =
        //     Array::<T, D>::ones(xn.raw_dim()).index_axis_move(Axis(batch_axis), 0);
        let zeros: Array<T, D::Smaller> =
            Array::<T, D::Smaller>::zeros(xn.raw_dim().try_remove_axis(Axis(batch_axis)));
        let ones: Array<T, D::Smaller> =
            Array::<T, D::Smaller>::ones(xn.raw_dim().try_remove_axis(Axis(batch_axis)));
        Self {
            momentum,
            batch_axis,
            batch_size: 1,
            gamma: ones.clone(),
            beta: zeros.clone(),
            xc: xn.clone(),
            xn,
            std: zeros.clone(),
            dgamma: ones.clone(),
            dbeta: zeros.clone(),
            trained: false,
            eps: cast_t2u(EPS),
            batch_size_t: one,
            one,
            two,
            half,
            one_minus_m: one - momentum,
            running_mean: zeros.clone(),
            running_var: zeros,
        }
    }
}

impl<T: 'static, D: 'static> LayerBase<T> for BatchNormalization<T, D>
where
    T: CrateFloat,
    D: Dimension + RemoveAxis,
{
    type A = Array<T, D>;

    type B = Array<T, D>;

    fn forward(&mut self, x: &Self::A) -> Self::B {
        if self.trained {
            let mu: Array<T, D::Smaller> = x.mean_axis(Axis(self.batch_axis)).unwrap();
            let xc: Self::B = x.clone() - &mu;
            // let mut xc: Self::A = x.clone();
            // for mut view in xc.axis_iter_mut(Axis(self.batch_size)) {
            //     for (v, m) in view.iter_mut().zip(mu.into_iter()) {
            //         *v = *v - *m;
            //     }
            // }
            let var: Array<T, D::Smaller> =
                xc.map(|&v| v * v).mean_axis(Axis(self.batch_axis)).unwrap();
            let std: Array<T, D::Smaller> = var.map(|&v| (v + self.eps).sqrt());
            let xn: Self::B = xc.clone() / &std;
            // for mut view in xn.axis_iter_mut(Axis(self.batch_size)) {
            //     for (v, m) in view.iter_mut().zip(std.iter()) {
            //         *v = *v / *m;
            //     }
            // }

            self.batch_size = x.len_of(Axis(self.batch_size));
            self.batch_size_t = cast_t2u(x.len_of(Axis(self.batch_size)));
            self.xc = xc;
            self.xn = xn.clone();
            self.std = std;
            self.running_mean *= self.momentum;
            self.running_mean.scaled_add(self.one_minus_m, &mu);
            self.running_var *= self.momentum;
            self.running_var.scaled_add(self.one_minus_m, &mu);
            // for (v, m) in multizip((self.running_mean.iter_mut(), mu.iter())) {
            //     *v = self.momentum * *v + self.one_minus_m * *m;
            // }
            // for mut view in xn.axis_iter_mut(Axis(self.batch_size)) {
            //     for (v, g, b) in multizip((view.iter_mut(), self.gamma.iter(), self.beta.iter())) {
            //         *v = *g * *v + *b;
            //     }
            // }
            // xn
            xn * &self.gamma + &self.beta
        } else {
            let xn: Self::B =
                (x.clone() - &self.running_mean) / self.running_var.map(|&v| (v + self.eps).sqrt());
            xn * &self.gamma + &self.beta
            // for mut view in dst.axis_iter_mut(Axis(self.batch_size)) {
            //     for (v, m) in view.iter_mut().zip(self.running_mean.iter()) {
            //         *v = *v - *m;
            //     }
            // }
            // for mut view in dst.axis_iter_mut(Axis(self.batch_size)) {
            //     for (v, m) in view.iter_mut().zip(self.running_var.iter()) {
            //         *v = *v / (*m + self.eps).sqrt();
            //     }
            // }
            // for mut view in dst.axis_iter_mut(Axis(self.batch_size)) {
            //     for (v, g, b) in multizip((view.iter_mut(), self.gamma.iter(), self.beta.iter())) {
            //         *v = *g * *v + *b;
            //     }
            // }
            // dst
        }
    }

    fn backward(&mut self, dx: &Self::B) -> Self::A {
        self.dbeta = dx.sum_axis(Axis(self.batch_axis));
        self.dgamma = (dx.clone() + &self.xn).sum_axis(Axis(self.batch_axis));
        let dxn: Self::A = dx.clone() * &self.gamma;
        let mut dxc: Self::A = dxn.clone() / &self.std;
        let dstd: Array<T, D::Smaller> =
            -((dxn * &self.xc) / self.std.map(|&v| v * v)).sum_axis(Axis(self.batch_axis));
        let dvar: Array<T, D::Smaller> = dstd * self.half / &self.std;
        dxc.scaled_add(self.two / self.batch_size_t, &(self.xc.clone() * &dvar));
        let dmu: Array<T, D::Smaller> = dxc.sum_axis(Axis(self.batch_axis));
        dxc - &(dmu / self.batch_size_t)
        // for mut view in dxn.axis_iter_mut(Axis(self.batch_size)) {
        //     for (v, g) in view.iter_mut().zip(self.gamma.iter()) {
        //         *v = *v * *g;
        //     }
        // }
        // let mut dxc: Self::B = dxn.clone();
        // for mut view in dxc.axis_iter_mut(Axis(self.batch_size)) {
        //     for (v, s) in view.iter_mut().zip(self.std.iter()) {
        //         *v = *v / *s;
        //     }
        // }
        // todo!()
    }
}
