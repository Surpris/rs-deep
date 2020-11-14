//! gradient_method
//!
//! gradient method test

use super::gradient_2d::numerical_gradient_1d;
use crate::dlfs01::cast_t2u;
use crate::dlfs01::MathFunc;
use crate::dlfs01::Operators;
use num_traits::Float;
use plotters::prelude::*;

type Vec2d<T> = Vec<Vec<T>>;
// type Vec3d<T> = Vec<Vec2d<T>>;

fn gradient_descent<T, F: Fn(&Vec<T>) -> T>(
    f: &F,
    x: Vec<T>,
    lr: T,
    step_num: usize,
) -> (Vec<T>, Vec2d<T>)
where
    T: Float,
{
    let mut x_ = x.clone();
    let mut history: Vec2d<T> = Vec2d::new();
    for _ in 0..step_num {
        history.push(x_.clone());
        let grad = numerical_gradient_1d(f, &mut x_);
        x_ = x_.sub(&grad.mul_value(lr));
    }
    (x_, history)
}

fn function_2<T>(x: &Vec<T>) -> T
where
    T: Float,
{
    x.powf(cast_t2u(2.0)).sum()
}

pub fn main() {
    println!("< gradient_method sub module >");
    let init_x: Vec<f32> = vec![-3.0, 4.0];
    let lr: f32 = 0.1;
    let step_num: usize = 20;
    let (x, history): (Vec<f32>, Vec2d<f32>) =
        gradient_descent(&function_2, init_x.clone(), lr, step_num);
    println!("init_x, final_x: {:?}, {:?}", init_x, x);
    match plot(
        "./images/gradient_method.png",
        480,
        640,
        &history.transpose()[0],
        &history.transpose()[1],
    ) {
        Ok(_) => println!("ok"),
        Err(s) => println!("{}", s),
    }
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
        .caption("Gradient method", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x[0]..x[x.len() - 1], -y.to_vec().min()..y.to_vec().max())?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new((0..x.len()).map(|ii| (x[ii], y[ii])), &RED))?
        .label("x^2 + y^2 = 25")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;
    Ok(())
}
