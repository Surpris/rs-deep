//! relu
//!
//! ReLU function

use num_traits::Float;
use num_traits::{Num, NumCast};
use plotters::prelude::*;
// use std::ops::AddAssign;
// use std::cmp::Ord;

pub fn main() {
    println!("< activation sub module >");
    let x: Vec<f32> = arange(-6.0, 6.0, 0.01);
    let y: Vec<f32> = relu(&x);
    let y2: Vec<f32> = sigmoid(&x);
    let y3: Vec<f32> = step(&x);
    match plot("./images/activation.png", 480, 640, &x, &y, &y2, &y3) {
        Ok(_) => println!("ok"),
        Err(s) => println!("{}", s),
    }
}

/// cast a numeric value with type T to one with U
fn cast_t2u<T, U>(x: T) -> U
where
    T: Num + NumCast + Copy,
    U: Num + NumCast + Copy,
{
    U::from(x).unwrap()
}

/// calculate a [`a`, `b`) vector with a step of `step`.
/// This function is similar to numpy.arange(a, b, step).
fn arange<T>(a: T, b: T, step: T) -> Vec<T>
where
    T: Float,
{
    let size = cast::usize(cast_t2u::<T, f32>((b - a) / step).floor()).unwrap();
    (0..size)
        .map(|i| a + cast_t2u::<usize, T>(i) * step)
        .collect()
}

/// ReLU function
fn relu<T>(x: &[T]) -> Vec<T>
where
    T: Float,
{
    let zero: T = cast_t2u(0.0);
    x.iter().map(|&v| T::max(zero, v)).collect()
}

/// sigmoid function
fn sigmoid<T>(x: &[T]) -> Vec<T>
where
    T: Float,
{
    let one: T = cast_t2u(1.0);
    x.iter().map(|&v| one / (one + T::exp(v))).collect()
}

/// step function
fn step<T>(x: &[T]) -> Vec<T>
where
    T: Float,
{
    let zero: T = cast_t2u(0.0);
    let one: T = cast_t2u(1.0);
    x.iter().map(|&v| if v <= zero { zero } else { one }).collect()
}

#[allow(dead_code)]
/// max function
fn max<T>(x: &[T]) -> T
where
    T: Float,
{
    let zero: T = cast_t2u(0.0);
    x.into_iter().fold(zero / zero, |m, &v| v.max(m))
}

#[allow(dead_code)]
/// min function
fn min<T>(x: &[T]) -> T
where
    T: Float,
{
    let zero: T = cast_t2u(0.0);
    x.into_iter().fold(zero / zero, |m, &v| v.min(m))
}

fn plot(
    file_name: &str,
    height: u32,
    width: u32,
    x: &[f32],
    y: &[f32],
    y2: &[f32],
    y3: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(file_name, (width, height)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("y=f(x)", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x[0]..x[x.len() - 1], -max(y)..max(y))?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new((0..x.len()).map(|ii| (x[ii], y[ii])), &RED))?
        .label("y = relu(x)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .draw_series(LineSeries::new(
            (0..x.len()).map(|ii| (x[ii], y2[ii])),
            &BLUE,
        ))?
        .label("y = sigmoid(x)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .draw_series(LineSeries::new(
            (0..x.len()).map(|ii| (x[ii], y3[ii])),
            &GREEN,
        ))?
        .label("y = step(x)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;
    Ok(())
}
