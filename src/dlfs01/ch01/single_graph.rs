//! # single_graph
//!
//!

use num_traits::float::Float;
use num_traits::{Num, NumCast};
use plotters::prelude::*;
use std::f64::consts::PI;

pub fn main() {
    println!("< single_graph sub module >");
    let x: Vec<f32> = arange(0.0, 6.0, 0.01);
    let y: Vec<f32> = sin(&x, 1.0);
    match plot("./images/single_graph.png", 480, 640, &x, &y) {
        Ok(_) => println!("ok"),
        Err(s) => println!("{}", s),
    }
}

/// cast a numeric value with type T to one with U
fn cast_t2u<T, U>(x: T) -> U
where
    T: Num + NumCast + Copy,
    U: Num + NumCast,
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

/// calculate a sine curve
fn sin<T>(t: &[T], f0: T) -> Vec<T>
where
    T: Float,
{
    t.iter()
        .map(|t_| (cast_t2u::<f32, T>(2.0) * cast_t2u::<f64, T>(PI) * f0 * *t_).sin())
        .collect()
}

fn plot(
    file_name: &str,
    height: u32,
    width: u32,
    x: &[f32],
    y: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(file_name, (width, height)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("y=x^2", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x[0]..x[x.len() - 1], -1f32..1f32)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new((0..x.len()).map(|ii| (x[ii], y[ii])), &RED))?
        .label("y = sin(x)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;
    Ok(())
}
