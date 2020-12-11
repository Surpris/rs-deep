//! activation
//!
//! Activation layers

#![allow(unused_variables)]

use super::super::math::sigmoid;
use super::super::util::cast_t2u;
use super::layer_base::LayerBase;
use ndarray::prelude::*;
use num_traits::Float;
use rand::distributions::Uniform;
use rand::prelude::*;
use std::marker::PhantomData;

// >>>>>>>>>>>>> ReLU layer >>>>>>>>>>>>>

/// Arbitrary-D ReLU layer
pub struct ReLU<T, D> {
    mask: Array<u8, D>,
    phantom: PhantomData<T>,
}

impl<T: 'static, D> ReLU<T, D>
where
    T: Float,
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

impl<T: 'static, D> LayerBase<T, D> for ReLU<T, D>
where
    T: Float,
    D: Dimension,
{
    fn forward(&mut self, x: &Array<T, D>) -> Array<T, D> {
        if self.mask.shape() != x.shape() {
            self.mask = Array::<u8, D>::zeros(x.raw_dim());
        }
        let zero: T = cast_t2u(0.0);
        self.mask = x.map(|&v| if v <= zero { 1 } else { 0 });
        x.map(|&v| if v <= zero { zero } else { v })
    }
    fn backward(&mut self, dx: &Array<T, D>) -> Array<T, D> {
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

pub type ReLU2<T> = ReLU<T, Ix2>;
pub type ReLU3<T> = ReLU<T, Ix3>;
pub type ReLU4<T> = ReLU<T, Ix4>;
pub type ReLU5<T> = ReLU<T, Ix5>;
pub type ReLU6<T> = ReLU<T, Ix6>;
pub type ReLUD<T> = ReLU<T, IxDyn>;

// <<<<<<<<<<<<< ReLU layer <<<<<<<<<<<<<

// >>>>>>>>>>>>> Sigmoid layer >>>>>>>>>>>>>

/// Arbitrary-D sigmoid layer
pub struct Sigmoid<T, D> {
    output: Array<T, D>,
}

impl<T: 'static, D> Sigmoid<T, D>
where
    T: Float,
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

pub type Sigmoid2<T> = Sigmoid<T, Ix2>;
pub type Sigmoid3<T> = Sigmoid<T, Ix3>;
pub type Sigmoid4<T> = Sigmoid<T, Ix4>;
pub type Sigmoid5<T> = Sigmoid<T, Ix5>;
// pub type Sigmoid6<T> = Sigmoid<T, Ix6>;
pub type SigmoidD<T> = Sigmoid<T, IxDyn>;

impl<T: 'static> LayerBase<T, Ix2> for Sigmoid2<T>
where
    T: Float,
{
    fn forward(&mut self, x: &Array2<T>) -> Array2<T> {
        self.output = x.map(|&v| sigmoid(v));
        self.output.clone()
    }
    fn backward(&mut self, dx: &Array2<T>) -> Array2<T> {
        let one: T = cast_t2u(1.0);
        let mut dst = dx.clone();
        for v in self.output.indexed_iter() {
            dst[v.0] = *v.1 * (one - *v.1) * *v.1;
        }
        dst
    }
    fn print_detail(&self) {
        println!("sigmoid activation layer.");
    }
}

impl<T: 'static> LayerBase<T, Ix3> for Sigmoid3<T>
where
    T: Float,
{
    fn forward(&mut self, x: &Array3<T>) -> Array3<T> {
        self.output = x.map(|&v| sigmoid(v));
        self.output.clone()
    }
    fn backward(&mut self, dx: &Array3<T>) -> Array3<T> {
        let one: T = cast_t2u(1.0);
        let mut dst = dx.clone();
        for v in self.output.indexed_iter() {
            dst[v.0] = *v.1 * (one - *v.1) * *v.1;
        }
        dst
    }
    fn print_detail(&self) {
        println!("sigmoid activation layer.");
    }
}

impl<T: 'static> LayerBase<T, Ix4> for Sigmoid4<T>
where
    T: Float,
{
    fn forward(&mut self, x: &Array4<T>) -> Array4<T> {
        self.output = x.map(|&v| sigmoid(v));
        self.output.clone()
    }
    fn backward(&mut self, dx: &Array4<T>) -> Array4<T> {
        let one: T = cast_t2u(1.0);
        let mut dst = dx.clone();
        for v in self.output.indexed_iter() {
            dst[v.0] = *v.1 * (one - *v.1) * *v.1;
        }
        dst
    }
    fn print_detail(&self) {
        println!("sigmoid activation layer.");
    }
}

impl<T: 'static> LayerBase<T, Ix5> for Sigmoid5<T>
where
    T: Float,
{
    fn forward(&mut self, x: &Array5<T>) -> Array5<T> {
        self.output = x.map(|&v| sigmoid(v));
        self.output.clone()
    }
    fn backward(&mut self, dx: &Array5<T>) -> Array5<T> {
        let one: T = cast_t2u(1.0);
        let mut dst = dx.clone();
        for v in self.output.indexed_iter() {
            dst[v.0] = *v.1 * (one - *v.1) * *v.1;
        }
        dst
    }
    fn print_detail(&self) {
        println!("sigmoid activation layer.");
    }
}

impl<T: 'static> LayerBase<T, IxDyn> for SigmoidD<T>
where
    T: Float,
{
    fn forward(&mut self, x: &ArrayD<T>) -> ArrayD<T> {
        self.output = x.map(|&v| sigmoid(v));
        self.output.clone()
    }
    fn backward(&mut self, dx: &ArrayD<T>) -> ArrayD<T> {
        let one: T = cast_t2u(1.0);
        let mut dst = dx.clone();
        for v in self.output.indexed_iter() {
            dst[v.0] = *v.1 * (one - *v.1) * *v.1;
        }
        dst
    }
    fn print_detail(&self) {
        println!("sigmoid activation layer.");
    }
}

// <<<<<<<<<<<<< Sigmoid layer <<<<<<<<<<<<<

// >>>>>>>>>>>>> Softmax layer >>>>>>>>>>>>>

/// Arbitrary-D sigmoid layer
pub struct Softmax<T, D> {
    pub output: Array<T, D>,
    axis: usize,
}

impl<T: 'static, D> Softmax<T, D>
where
    T: Float,
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

pub type Softmax2<T> = Softmax<T, Ix2>;
pub type Softmax3<T> = Softmax<T, Ix3>;
pub type Softmax4<T> = Softmax<T, Ix4>;
pub type Softmax5<T> = Softmax<T, Ix5>;
pub type Softmax6<T> = Softmax<T, Ix6>;
pub type SoftmaxD<T> = Softmax<T, IxDyn>;

impl<T: 'static> LayerBase<T, Ix2> for Softmax2<T>
where
    T: Float,
{
    fn forward(&mut self, x: &Array2<T>) -> Array2<T> {
        let zero: T = cast_t2u(0.0);
        let mut dst = x.clone();
        for mut view in dst.axis_iter_mut(Axis(self.axis)) {
            let v_max = view.fold(zero / zero, |m, &v| v.max(m));
            view.mapv_inplace(|v| T::exp(v - v_max));
            let v_exp_sum = view.sum();
            view.mapv_inplace(|v| v / v_exp_sum);
        }
        self.output = dst;
        self.output.clone()
    }
    fn backward(&mut self, dx: &Array2<T>) -> Array2<T> {
        // TODO: correct implementation
        dx.clone()
    }
    fn print_detail(&self) {
        println!("softmax activation layer.");
    }
}

impl<T: 'static> LayerBase<T, Ix3> for Softmax3<T>
where
    T: Float,
{
    fn forward(&mut self, x: &Array3<T>) -> Array3<T> {
        let zero: T = cast_t2u(0.0);
        let mut dst = x.clone();
        for mut view in dst.axis_iter_mut(Axis(self.axis)) {
            let v_max = view.fold(zero / zero, |m, &v| v.max(m));
            view.mapv_inplace(|v| T::exp(v - v_max));
            let v_exp_sum = view.sum();
            view.mapv_inplace(|v| v / v_exp_sum);
        }
        self.output = dst;
        self.output.clone()
    }
    fn backward(&mut self, dx: &Array3<T>) -> Array3<T> {
        // TODO: correct implementation
        dx.clone()
    }
    fn print_detail(&self) {
        println!("softmax activation layer.");
    }
}

impl<T: 'static> LayerBase<T, Ix4> for Softmax4<T>
where
    T: Float,
{
    fn forward(&mut self, x: &Array4<T>) -> Array4<T> {
        let zero: T = cast_t2u(0.0);
        let mut dst = x.clone();
        for mut view in dst.axis_iter_mut(Axis(self.axis)) {
            let v_max = view.fold(zero / zero, |m, &v| v.max(m));
            view.mapv_inplace(|v| T::exp(v - v_max));
            let v_exp_sum = view.sum();
            view.mapv_inplace(|v| v / v_exp_sum);
        }
        self.output = dst;
        self.output.clone()
    }
    fn backward(&mut self, dx: &Array4<T>) -> Array4<T> {
        // TODO: correct implementation
        dx.clone()
    }
    fn print_detail(&self) {
        println!("softmax activation layer.");
    }
}

impl<T: 'static> LayerBase<T, Ix5> for Softmax5<T>
where
    T: Float,
{
    fn forward(&mut self, x: &Array5<T>) -> Array5<T> {
        let zero: T = cast_t2u(0.0);
        let mut dst = x.clone();
        for mut view in dst.axis_iter_mut(Axis(self.axis)) {
            let v_max = view.fold(zero / zero, |m, &v| v.max(m));
            view.mapv_inplace(|v| T::exp(v - v_max));
            let v_exp_sum = view.sum();
            view.mapv_inplace(|v| v / v_exp_sum);
        }
        self.output = dst;
        self.output.clone()
    }
    fn backward(&mut self, dx: &Array5<T>) -> Array5<T> {
        // TODO: correct implementation
        dx.clone()
    }
    fn print_detail(&self) {
        println!("softmax activation layer.");
    }
}

impl<T: 'static> LayerBase<T, Ix6> for Softmax6<T>
where
    T: Float,
{
    fn forward(&mut self, x: &Array6<T>) -> Array6<T> {
        let zero: T = cast_t2u(0.0);
        let mut dst = x.clone();
        for mut view in dst.axis_iter_mut(Axis(self.axis)) {
            let v_max = view.fold(zero / zero, |m, &v| v.max(m));
            view.mapv_inplace(|v| T::exp(v - v_max));
            let v_exp_sum = view.sum();
            view.mapv_inplace(|v| v / v_exp_sum);
        }
        self.output = dst;
        self.output.clone()
    }
    fn backward(&mut self, dx: &Array6<T>) -> Array6<T> {
        // TODO: correct implementation
        dx.clone()
    }
    fn print_detail(&self) {
        println!("softmax activation layer.");
    }
}

impl<T: 'static> LayerBase<T, IxDyn> for SoftmaxD<T>
where
    T: Float,
{
    fn forward(&mut self, x: &ArrayD<T>) -> ArrayD<T> {
        let zero: T = cast_t2u(0.0);
        let mut dst = x.clone();
        for mut view in dst.axis_iter_mut(Axis(self.axis)) {
            let v_max = view.fold(zero / zero, |m, &v| v.max(m));
            view.mapv_inplace(|v| T::exp(v - v_max));
            let v_exp_sum = view.sum();
            view.mapv_inplace(|v| v / v_exp_sum);
        }
        self.output = dst;
        self.output.clone()
    }
    fn backward(&mut self, dx: &ArrayD<T>) -> ArrayD<T> {
        // TODO: correct implementation
        dx.clone()
    }
    fn print_detail(&self) {
        println!("softmax activation layer.");
    }
}

// <<<<<<<<<<<<< Softmax layer <<<<<<<<<<<<<

pub fn main() {
    println!("< activation sub module> ");
    let mut rng = rand::thread_rng();
    let gen = Uniform::new(-1.0f32, 1.0f32);

    let a = ArrayD::<f32>::zeros(IxDyn(&[2, 3]));
    let a = a.map(|_| gen.sample(&mut rng));

    println!("ReLU layer");
    let mut layer = ReLUD::<f32>::new(a.shape());
    let b = layer.forward(&a);
    let da = ArrayD::<f32>::zeros(IxDyn(a.shape())).map(|_| gen.sample(&mut rng));
    println!("a: {}", a);
    println!("forward: {}", b);
    println!("da: {}", da);
    println!("backward: {}", layer.backward(&da));

    println!("sigmoid layer");
    let mut layer = SigmoidD::<f32>::new(a.shape());
    let b = layer.forward(&a);
    let da = ArrayD::<f32>::zeros(IxDyn(a.shape())).map(|_| gen.sample(&mut rng));
    println!("a: {}", a);
    println!("forward: {}", b);
    println!("da: {}", da);
    println!("backward: {}", layer.backward(&da));

    println!("softmax layer");
    let mut layer = SoftmaxD::<f32>::new(a.shape(), 0);
    let b = layer.forward(&a);
    let da = ArrayD::<f32>::zeros(IxDyn(a.shape())).map(|_| gen.sample(&mut rng));
    println!("a: {}", a);
    println!("forward: {}, {}", b, b.sum_axis(Axis(1)));
    println!("da: {}", da);
    println!("backward: {}", layer.backward(&da));
}
