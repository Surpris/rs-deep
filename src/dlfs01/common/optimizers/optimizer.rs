//! optimizer
//!
//! optimizer struct

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::marker::PhantomData;

use crate::prelude::*;

use super::optimizer_base::OptimizerBase;
use itertools::multizip;
use ndarray::prelude::*;

const EPS: f64 = 1E-8;

/// stochastic gradient descent
pub struct SGD<T: CrateFloat, D> {
    lr: T,
    _phantom: PhantomData<D>,
}

impl<T, D> SGD<T, D>
where
    T: CrateFloat,
    D: Dimension,
{
    pub fn new(lr: T) -> Self {
        Self {
            lr,
            _phantom: PhantomData,
        }
    }
}

impl<T, D> OptimizerBase for SGD<T, D>
where
    T: CrateFloat,
    D: Dimension,
{
    type Src = Array<T, D>;
    fn update(&mut self, param: &mut Self::Src, grads: &Self::Src) {
        param.scaled_add(-self.lr, grads);
        // for (v, g) in param.iter_mut().zip(grads.iter()) {
        //     *v = *v - self.lr * *g;
        // }
    }
}

/// momentum
pub struct Momentum<T: CrateFloat, D> {
    lr: T,
    momentum: T,
    param: Array<T, D>,
}

impl<T, D> Momentum<T, D>
where
    T: CrateFloat,
    D: Dimension,
{
    pub fn new<Sh>(lr: T, momentum: T, shape: Sh) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        Self {
            lr,
            momentum,
            param: Array::<T, D>::zeros(shape),
        }
    }
}

impl<T, D> OptimizerBase for Momentum<T, D>
where
    T: CrateFloat,
    D: Dimension,
{
    type Src = Array<T, D>;
    fn update(&mut self, param: &mut Self::Src, grads: &Self::Src) {
        if self.param.shape() != param.shape() {
            self.param = Array::<T, D>::zeros(param.raw_dim());
        }
        // self.param *= self.momentum;
        // self.param.scaled_add(-self.lr, grads);
        // for (v, p) in multizip((self.param.iter(), param.iter_mut())) {
        //     *p = *p + *v;
        // }
        for (v, p, g) in multizip((self.param.iter_mut(), param.iter_mut(), grads.iter())) {
            *v = self.momentum * *v - self.lr * *g;
            *p = *p + *v;
        }
    }
}

/// Nesterov's Accelerated Gradient
///
/// See http://arxiv.org/abs/1212.0901 in detail.
pub struct Nesterov<T, D> {
    lr: T,
    momentum: T,
    one_plus_mom: T,
    mom2: T,
    param: Array<T, D>,
}

impl<T, D> Nesterov<T, D>
where
    T: CrateFloat,
    D: Dimension,
{
    pub fn new<Sh>(lr: T, momentum: T, shape: Sh) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        Self {
            lr,
            momentum,
            one_plus_mom: cast_t2u::<f32, T>(1.0) + momentum,
            mom2: momentum * momentum,
            param: Array::<T, D>::zeros(shape),
        }
    }
}

impl<T, D> OptimizerBase for Nesterov<T, D>
where
    T: CrateFloat,
    D: Dimension,
{
    type Src = Array<T, D>;
    fn update(&mut self, param: &mut Self::Src, grads: &Self::Src) {
        if self.param.shape() != param.shape() {
            self.param = Array::<T, D>::zeros(param.raw_dim());
        }
        // self.param *= self.momentum;
        // self.param.scaled_add(-self.lr, grads);
        // param.scaled_add(self.mom2, &self.param);
        // param.scaled_add(-self.one_plus_mom * self.lr, grads);
        for (v, p, g) in multizip((self.param.iter_mut(), param.iter_mut(), grads.iter())) {
            *v = self.momentum * *v - self.lr * *g;
            *p = *p + self.mom2 * *v - self.one_plus_mom * self.lr * *g;
        }
    }
}

/// AdaGrad
///
/// See https://jmlr.org/papers/v12/duchi11a.html in detail
pub struct AdaGrad<T, D> {
    lr: T,
    param: Array<T, D>,
    eps: T,
}

impl<T, D> AdaGrad<T, D>
where
    T: CrateFloat,
    D: Dimension,
{
    pub fn new<Sh>(lr: T, shape: Sh) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        Self {
            lr,
            param: Array::<T, D>::zeros(shape),
            eps: cast_t2u(EPS),
        }
    }
}

impl<T, D> OptimizerBase for AdaGrad<T, D>
where
    T: CrateFloat,
    D: Dimension,
{
    type Src = Array<T, D>;
    fn update(&mut self, param: &mut Self::Src, grads: &Self::Src) {
        if self.param.shape() != grads.shape() {
            self.param = Array::<T, D>::zeros(grads.raw_dim());
        }
        for (v, p, g) in multizip((self.param.iter_mut(), param.iter_mut(), grads.iter())) {
            *v = *v + *g * *g;
            *p = *p - self.lr * *g / ((*v).sqrt() + self.eps);
        }
    }
}

/// RMSprop
///
/// See "Tijmen Tieleman; G. Hinton (2012). Lecture 6.5 - rmsprop,
/// COURSERA: Neural Networks for Machine Learning." in detail
pub struct RMSprop<T, D> {
    lr: T,
    decay_rate: T,
    param: Array<T, D>,
    eps: T,
    one_minus_rate: T,
}

impl<T, D> RMSprop<T, D>
where
    T: CrateFloat,
    D: Dimension,
{
    pub fn new<Sh>(lr: T, decay_rate: T, shape: Sh) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        Self {
            lr,
            decay_rate,
            param: Array::<T, D>::zeros(shape),
            eps: cast_t2u(EPS),
            one_minus_rate: cast_t2u::<f32, T>(1.0) - decay_rate,
        }
    }
}

impl<T, D> OptimizerBase for RMSprop<T, D>
where
    T: CrateFloat,
    D: Dimension,
{
    type Src = Array<T, D>;
    fn update(&mut self, param: &mut Self::Src, grads: &Self::Src) {
        if self.param.shape() != grads.shape() {
            self.param = Array::<T, D>::zeros(grads.raw_dim());
        }
        for (v, p, g) in multizip((self.param.iter_mut(), param.iter_mut(), grads.iter())) {
            *v = self.decay_rate * self.one_minus_rate * *g * *g;
            *p = *p - self.lr * *g / ((*v).sqrt() + self.eps);
        }
    }
}

/// AdaDelta
///
/// See https://arxiv.org/abs/1212.5701 in detail
pub struct AdaDelta<T: CrateFloat, D> {
    lr: T,
    param: Array<T, D>,
    eps: T,
}

/// Adam
///
/// See https://arxiv.org/abs/1412.6980 in detail
pub struct Adam<T, D> {
    lr: T,
    beta1: T,
    beta2: T,
    iter: T,
    param: Array<T, D>,
    momentum: Array<T, D>,
    eps: T,
    one: T,
    one_minus_beta1: T,
    one_minus_beta2: T,
}

impl<T, D> Adam<T, D>
where
    T: CrateFloat,
    D: Dimension,
{
    pub fn new<Sh>(lr: T, beta1: T, beta2: T, shape: Sh) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        let one: T = cast_t2u(1.0);
        let zeros = Array::<T, D>::zeros(shape);
        Self {
            lr,
            beta1,
            beta2,
            iter: cast_t2u(1.0),
            param: zeros.clone(),
            momentum: zeros,
            eps: cast_t2u(EPS),
            one,
            one_minus_beta1: one - beta1,
            one_minus_beta2: one - beta2,
        }
    }
}

impl<T, D> OptimizerBase for Adam<T, D>
where
    T: CrateFloat,
    D: Dimension,
{
    type Src = Array<T, D>;
    fn update(&mut self, param: &mut Self::Src, grads: &Self::Src) {
        if self.param.shape() != param.shape() {
            self.param = Array::<T, D>::zeros(param.raw_dim());
            self.momentum = Array::<T, D>::zeros(param.raw_dim());
        }
        self.iter = self.iter + self.one;
        let lr_t: T = self.lr * (self.one - self.beta2.powf(self.iter))
            / (self.one - self.beta1.powf(self.iter));
        for (v, m, p, g) in multizip((
            self.param.iter_mut(),
            self.momentum.iter_mut(),
            param.iter_mut(),
            grads.iter(),
        )) {
            *m = *m + self.one_minus_beta1 * (*g - *m);
            *v = *v + self.one_minus_beta2 * (*g * *g - *v);
            *p = *p - lr_t * *m / ((*v).sqrt() + self.eps);
        }
    }
}

/// RMSpropGraves
///
/// See https://arxiv.org/abs/1308.0850 in detail
pub struct RMSpropGraves<T: CrateFloat, D> {
    lr: T,
    decay_rate: T,
    param: Array<T, D>,
    eps: T,
    one_minus_rate: T,
}

/// SMORMS3
///
/// See https://sifter.org/~simon/journal/20150420.html in detail
pub struct SMORMS3<T: CrateFloat, D> {
    lr: T,
    decay_rate: T,
    param: Array<T, D>,
    eps: T,
    one_minus_rate: T,
}

/// AdaMax
///
/// See https://arxiv.org/abs/1412.6980 in detail
pub struct AdaMax<T: CrateFloat, D> {
    lr: T,
    decay_rate: T,
    param: Array<T, D>,
    eps: T,
    one_minus_rate: T,
}

/// Nadam
///
/// See https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ in detail
pub struct Nadam<T: CrateFloat, D> {
    lr: T,
    decay_rate: T,
    param: Array<T, D>,
    eps: T,
    one_minus_rate: T,
}

/// Eve
///
/// See https://arxiv.org/abs/1611.01505 in detail
pub struct Eve<T: CrateFloat, D> {
    lr: T,
    beta1: T,
    beta2: T,
    iter: T,
    param: Array<T, D>,
    momentum: Array<T, D>,
    eps: T,
    one: T,
    one_minus_beta1: T,
    one_minus_beta2: T,
}

/// Santa
///
/// See http://proceedings.mlr.press/v51/chen16c.pdf in detail
pub struct Santa<T: CrateFloat, D> {
    lr: T,
    decay_rate: T,
    param: Array<T, D>,
    eps: T,
    one_minus_rate: T,
}

/// GD by GD
///
/// See https://proceedings.neurips.cc/paper/2016/file/fb87582825f9d28a8d42c5e5e5e8b23d-Paper.pdf
/// in detail
pub struct GDByGD<T: CrateFloat, D> {
    lr: T,
    decay_rate: T,
    param: Array<T, D>,
    eps: T,
    one_minus_rate: T,
}

/// AdaSecant
///
/// See https://arxiv.org/abs/1412.7419 in detail
pub struct AdaSecant<T: CrateFloat, D> {
    lr: T,
    decay_rate: T,
    param: Array<T, D>,
    eps: T,
    one_minus_rate: T,
}

/// AMSGrad
///
/// See http://www.satyenkale.com/papers/amsgrad.pdf in detail
pub struct AMSGrad<T: CrateFloat, D> {
    lr: T,
    decay_rate: T,
    param: Array<T, D>,
    eps: T,
    one_minus_rate: T,
}

/// AdaBound
///
/// See https://openreview.net/pdf?id=Bkg3g2R9FX in detail
pub struct AdaBound<T: CrateFloat, D> {
    lr: T,
    decay_rate: T,
    param: Array<T, D>,
    eps: T,
    one_minus_rate: T,
}

/// AMSBound
///
/// See https://openreview.net/pdf?id=Bkg3g2R9FX in detail
pub struct AMSBound<T: CrateFloat, D> {
    lr: T,
    decay_rate: T,
    param: Array<T, D>,
    eps: T,
    one_minus_rate: T,
}

/// AdaBelief
///
/// See https://arxiv.org/abs/2010.07468 in detail
pub struct AdaBelief<T: CrateFloat, D> {
    lr: T,
    decay_rate: T,
    param: Array<T, D>,
    eps: T,
    one_minus_rate: T,
}
