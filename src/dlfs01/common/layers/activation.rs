//! activation
//!
//! Activation layers

#![allow(unused_variables)]

use super::super::util::*;
use super::layer_base::LayerBase;
use ndarray::{prelude::*, RemoveAxis};
use std::marker::PhantomData;

// >>>>>>>>>>>>> Idwntity layer >>>>>>>>>>>>>
/// Arbitrary-D Identity layer
pub struct Identity<T: CrateFloat, D> {
    output: Array<T, D>,
}

impl<T: 'static, D> Identity<T, D>
where
    T: CrateFloat,
    D: Dimension,
{
    pub fn new<Sh>(shape: Sh) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        Self {
            output: Array::<T, D>::zeros(shape),
        }
    }
}

impl<T: 'static, D> LayerBase<T> for Identity<T, D>
where
    T: CrateFloat,
    D: Dimension,
{
    type A = Array<T, D>;
    type B = Array<T, D>;
    fn forward(&mut self, x: &Self::A) -> Self::B {
        self.output = x.clone();
        self.output.clone()
    }
    fn backward(&mut self, dx: &Self::B) -> Self::A {
        dx.clone()
    }
    fn print_detail(&self) {
        println!("Identity activation layer.");
    }
}

pub type Identity1<T> = Identity<T, Ix1>;
pub type Identity2<T> = Identity<T, Ix2>;
pub type Identity3<T> = Identity<T, Ix3>;
pub type Identity4<T> = Identity<T, Ix4>;
pub type Identity5<T> = Identity<T, Ix5>;
pub type Identity6<T> = Identity<T, Ix6>;
pub type IdentityD<T> = Identity<T, IxDyn>;

// <<<<<<<<<<<<< Identity layer <<<<<<<<<<<<<

// >>>>>>>>>>>>> ReLU layer >>>>>>>>>>>>>

/// Arbitrary-D ReLU layer
pub struct ReLU<T: CrateFloat, D> {
    mask: Array<u8, D>,
    phantom: PhantomData<T>,
}

impl<T: 'static, D> ReLU<T, D>
where
    T: CrateFloat,
    D: Dimension,
{
    pub fn new<Sh>(shape: Sh) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        Self {
            mask: Array::<u8, D>::zeros(shape),
            phantom: PhantomData,
        }
    }
}

impl<T: 'static, D> LayerBase<T> for ReLU<T, D>
where
    T: CrateFloat,
    D: Dimension,
{
    type A = Array<T, D>;
    type B = Array<T, D>;
    fn forward(&mut self, x: &Self::A) -> Self::B {
        if self.mask.shape() != x.shape() {
            self.mask = Array::<u8, D>::zeros(x.raw_dim());
        }
        let zero: T = cast_t2u(0.0);
        self.mask = x.map(|&v| if v <= zero { 1 } else { 0 });
        x.map(|&v| if v <= zero { zero } else { v })
    }
    fn backward(&mut self, dx: &Self::B) -> Self::A {
        let zero: T = cast_t2u(0.0);
        let mut dst = dx.clone();
        for (v, d) in self.mask.iter().zip(dst.iter_mut()) {
            if *v != 0u8 {
                *d = zero;
            }
        }
        dst
    }
    fn print_detail(&self) {
        println!("ReLU activation layer.");
    }
}

pub type ReLU1<T> = ReLU<T, Ix1>;
pub type ReLU2<T> = ReLU<T, Ix2>;
pub type ReLU3<T> = ReLU<T, Ix3>;
pub type ReLU4<T> = ReLU<T, Ix4>;
pub type ReLU5<T> = ReLU<T, Ix5>;
pub type ReLU6<T> = ReLU<T, Ix6>;
pub type ReLUD<T> = ReLU<T, IxDyn>;

// <<<<<<<<<<<<< ReLU layer <<<<<<<<<<<<<

// >>>>>>>>>>>>> Sigmoid layer >>>>>>>>>>>>>

/// Arbitrary-D sigmoid layer
pub struct Sigmoid<T: CrateFloat, D> {
    output: Array<T, D>,
}

impl<T: 'static, D> Sigmoid<T, D>
where
    T: CrateFloat,
    D: Dimension,
{
    pub fn new<Sh>(shape: Sh) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        Self {
            output: Array::<T, D>::zeros(shape),
        }
    }
}

impl<T: 'static, D> LayerBase<T> for Sigmoid<T, D>
where
    T: CrateFloat,
    D: Dimension,
{
    type A = Array<T, D>;
    type B = Array<T, D>;
    fn forward(&mut self, x: &Self::A) -> Self::B {
        let one: T = cast_t2u(1.0);
        self.output = x.map(|&v| one / (one + T::exp(-v)));
        self.output.clone()
    }
    fn backward(&mut self, dx: &Self::B) -> Self::A {
        let one: T = cast_t2u(1.0);
        let mut dst = dx.clone();
        for (v, d) in self.output.iter().zip(dst.iter_mut()) {
            *d = *v * (one - *v) * *v;
        }
        dst
    }
    fn print_detail(&self) {
        println!("sigmoid activation layer.");
    }
}

pub type Sigmoid1<T> = Sigmoid<T, Ix1>;
pub type Sigmoid2<T> = Sigmoid<T, Ix2>;
pub type Sigmoid3<T> = Sigmoid<T, Ix3>;
pub type Sigmoid4<T> = Sigmoid<T, Ix4>;
pub type Sigmoid5<T> = Sigmoid<T, Ix5>;
pub type Sigmoid6<T> = Sigmoid<T, Ix6>;
pub type SigmoidD<T> = Sigmoid<T, IxDyn>;

// <<<<<<<<<<<<< Sigmoid layer <<<<<<<<<<<<<

// >>>>>>>>>>>>> Softmax layer >>>>>>>>>>>>>

/// Arbitrary-D sigmoid layer
pub struct Softmax<T: CrateFloat, D> {
    pub output: Array<T, D>,
    axis: usize,
}

impl<T: 'static, D> Softmax<T, D>
where
    T: CrateFloat,
    D: Dimension,
{
    pub fn new<Sh>(shape: Sh, axis: usize) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        Self {
            output: Array::<T, D>::zeros(shape),
            axis,
        }
    }
}

impl<T: 'static, D> LayerBase<T> for Softmax<T, D>
where
    T: CrateFloat,
    D: Dimension + RemoveAxis,
{
    type A = Array<T, D>;
    type B = Array<T, D>;
    fn forward(&mut self, x: &Self::A) -> Self::B {
        let zero: T = cast_t2u(0.0);
        let batch_size: T = cast_t2u(x.len_of(Axis(0)));

        self.output = x.clone();
        for mut view in self.output.axis_iter_mut(Axis(self.axis)) {
            let v_max = view.fold(zero / zero, |m, &v| v.max(m));
            view.mapv_inplace(|v| T::exp(v - v_max));
            let v_exp_sum = view.sum();
            view.mapv_inplace(|v| v / v_exp_sum);
        }
        self.output.clone()
    }
    fn backward(&mut self, dx: &Self::B) -> Self::A {
        // TODO: correct implementation
        dx.clone()
    }
    fn print_detail(&self) {
        println!("softmax activation layer.");
    }
}

pub type Softmax2<T> = Softmax<T, Ix2>;
pub type Softmax3<T> = Softmax<T, Ix3>;
pub type Softmax4<T> = Softmax<T, Ix4>;
pub type Softmax5<T> = Softmax<T, Ix5>;
pub type Softmax6<T> = Softmax<T, Ix6>;
pub type SoftmaxD<T> = Softmax<T, IxDyn>;

// <<<<<<<<<<<<< Softmax layer <<<<<<<<<<<<<
