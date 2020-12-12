//! optimizer
//!
//! optimizer struct

use crate::prelude::cast_t2u;

use super::optimizer_base::OptimizerBase;
use itertools::multizip;
use ndarray::prelude::*;
use num_traits::Float;

const EPS: f64 = 1E-8;

/// stochastic gradient descent
pub struct SGD<T> {
    lr: T,
}

impl<T> SGD<T>
where
    T: Float,
{
    pub fn new(lr: T) -> Self {
        Self { lr }
    }
}

impl<T, D> OptimizerBase<T, D> for SGD<T>
where
    T: Float,
    D: Dimension,
{
    fn update(&mut self, param: &mut Array<T, D>, grads: &Array<T, D>) {
        for (v, g) in param.iter_mut().zip(grads.iter()) {
            *v = *v - self.lr * *g;
        }
    }
}

/// momentum
pub struct Momentum<T, D> {
    lr: T,
    momentum: T,
    param: Array<T, D>,
}

impl<T, D> Momentum<T, D>
where
    T: Float,
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

impl<T, D> OptimizerBase<T, D> for Momentum<T, D>
where
    T: Float,
    D: Dimension,
{
    fn update(&mut self, param: &mut Array<T, D>, grads: &Array<T, D>) {
        if self.param.shape() != param.shape() {
            self.param = Array::<T, D>::zeros(param.raw_dim());
        }
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
    T: Float,
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

impl<T, D> OptimizerBase<T, D> for Nesterov<T, D>
where
    T: Float,
    D: Dimension,
{
    fn update(&mut self, param: &mut Array<T, D>, grads: &Array<T, D>) {
        if self.param.shape() != param.shape() {
            self.param = Array::<T, D>::zeros(param.raw_dim());
        }
        for (v, p, g) in multizip((self.param.iter_mut(), param.iter_mut(), grads.iter())) {
            *v = self.momentum * *v - self.lr * *g;
            *p = *p + self.mom2 * *v;
            *p = *p - self.one_plus_mom * self.lr * *g;
        }
    }
}

/// AdaGrad
pub struct AdaGrad<T, D> {
    lr: T,
    param: Array<T, D>,
    eps: T,
}

impl<T, D> AdaGrad<T, D>
where
    T: Float,
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

impl<T, D> OptimizerBase<T, D> for AdaGrad<T, D>
where
    T: Float,
    D: Dimension,
{
    fn update(&mut self, param: &mut Array<T, D>, grads: &Array<T, D>) {
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
pub struct RMSprop<T, D> {
    lr: T,
    decay_rate: T,
    param: Array<T, D>,
    eps: T,
    one_minus_rate: T,
}

impl<T, D> RMSprop<T, D>
where
    T: Float,
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

impl<T, D> OptimizerBase<T, D> for RMSprop<T, D>
where
    T: Float,
    D: Dimension,
{
    fn update(&mut self, param: &mut Array<T, D>, grads: &Array<T, D>) {
        if self.param.shape() != grads.shape() {
            self.param = Array::<T, D>::zeros(grads.raw_dim());
        }
        for (v, p, g) in multizip((self.param.iter_mut(), param.iter_mut(), grads.iter())) {
            *v = self.decay_rate * self.one_minus_rate * *g * *g;
            *p = *p - self.lr * *g / ((*v).sqrt() + self.eps);
        }
    }
}

/// Adam
///
/// See http://arxiv.org/abs/1412.6980v8 in detail
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
    T: Float,
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

impl<T, D> OptimizerBase<T, D> for Adam<T, D>
where
    T: Float,
    D: Dimension,
{
    fn update(&mut self, param: &mut Array<T, D>, grads: &Array<T, D>) {
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
