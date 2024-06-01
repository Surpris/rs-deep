//! batch_normalization
//!
//! BatchNormalization layer

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use super::super::util::*;
use super::layer_base::LayerBase;
// use itertools::multizip;
use ndarray::{prelude::*, RemoveAxis};
use std::fmt::{Debug, Display};

const EPS: f64 = 1E-8;

/// Enum for BatchNormalization layer
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum UseBatchNormEnum<T: CrateFloat> {
    Use(T),
    None,
}

impl<T> Display for UseBatchNormEnum<T>
where
    T: CrateFloat,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UseBatchNormEnum::Use(x) => write!(f, "Use (momentum: {})", x),
            UseBatchNormEnum::None => write!(f, "None"),
        }
    }
}

/// validate UseBatchNormEum and generate an appropriate layer
pub fn call_batch_norm_layer<T: 'static, D: 'static, Sh>(
    use_batch_norm: UseBatchNormEnum<T>,
    shape: Sh,
    batch_axis: usize,
) -> BatchNormalization<T, D>
where
    T: CrateFloat,
    D: Dimension + RemoveAxis,
    Sh: ShapeBuilder<Dim = D>,
{
    if let UseBatchNormEnum::Use(momentum) = use_batch_norm {
        BatchNormalization::new(momentum, batch_axis, shape)
    } else {
        BatchNormalization::new(cast_t2u(0.0), batch_axis, shape)
    }
}

/// BatchNormalization
///
/// See http://arxiv.org/abs/1502.03167 in detail
#[derive(Clone, Debug)]
pub struct BatchNormalization<T: CrateFloat, D: Dimension + RemoveAxis> {
    momentum: T,
    batch_axis: usize,
    batch_size: usize,
    pub gamma: Array<T, D::Smaller>,
    pub beta: Array<T, D::Smaller>,
    pub dgamma: Array<T, D::Smaller>,
    pub dbeta: Array<T, D::Smaller>,
    xc: Array<T, D>,
    xn: Array<T, D>,
    std: Array<T, D::Smaller>,
    trainable: bool,
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
            trainable: true,
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
        if self.trainable {
            let mu: Array<T, D::Smaller> = x.mean_axis(Axis(self.batch_axis)).unwrap();
            let xc: Self::B = x.clone() - &mu;
            let var: Array<T, D::Smaller> =
                xc.map(|&v| v * v).mean_axis(Axis(self.batch_axis)).unwrap();
            let std: Array<T, D::Smaller> = var.map(|&v| (v + self.eps).sqrt());
            let xn: Self::B = xc.clone() / &std;

            self.batch_size = x.len_of(Axis(self.batch_axis));
            self.batch_size_t = cast_t2u(x.len_of(Axis(self.batch_axis)));
            self.xc = xc;
            self.xn = xn.clone();
            self.std = std;
            self.running_mean *= self.momentum;
            self.running_mean.scaled_add(self.one_minus_m, &mu);
            self.running_var *= self.momentum;
            self.running_var.scaled_add(self.one_minus_m, &var);
            xn * &self.gamma + &self.beta
        } else {
            let xn: Self::B =
                (x.clone() - &self.running_mean) / self.running_var.map(|&v| (v + self.eps).sqrt());
            xn * &self.gamma + &self.beta
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
    }

    fn set_trainable(&mut self, flag: bool) {
        self.trainable = flag;
    }

    fn print_detail(&self) {
        println!("batch normalization layer.");
        println!("gamma shape: {:?}", self.gamma.shape());
        println!("beta shape: {:?}", self.beta.shape());
    }
    fn print_parameters(&self) {
        println!("gamma: {:?}", self.gamma);
        println!("beta: {:?}", self.beta);
        println!("dgamma: {:?}", self.dgamma);
        println!("dbeta: {:?}", self.dbeta);
    }
}

pub type BatchNormalization2<T> = BatchNormalization<T, Ix2>;
pub type BatchNormalization3<T> = BatchNormalization<T, Ix3>;
pub type BatchNormalization4<T> = BatchNormalization<T, Ix4>;
pub type BatchNormalization5<T> = BatchNormalization<T, Ix5>;
pub type BatchNormalization6<T> = BatchNormalization<T, Ix6>;
pub type BatchNormalizationD<T> = BatchNormalization<T, IxDyn>;
