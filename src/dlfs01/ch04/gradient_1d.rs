//! gradient_1d
//!
//! 1D gradient test

use super::super::common::math::*;
use super::super::common::util::cast_t2u;
use num_traits::Float;
use plotters::prelude::*;
const EPS: f64 = 1E-4;

fn numerical_diff<T, F: Fn(T) -> T>(f: &F, x: T) -> T
where
    T: Float,
{
    let eps: T = cast_t2u(EPS);
    let eps2: T = cast_t2u(2.0 * EPS);
    (f(x + eps) - f(x - eps)) / eps2
}

fn function_1<T>(x: T) -> T
where
    T: Float,
{
    let x_f64: f64 = cast_t2u(x);
    cast_t2u(0.01 * x_f64.powf(2.0) + 0.1 * x_f64)
}

fn tangent_line<T, F: Fn(T) -> T>(f: &F, x: T, t: &[T]) -> Vec<T>
where
    T: Float,
{
    let d = numerical_diff(f, x);
    let y = f(x) - d * x;
    (0..t.len()).map(|ii| d * t[ii] + y).collect()
}

pub fn main() {
    println!("< gradient_1d sub module >");
    let x: Vec<f32> = arange(0.0, 20.0, 0.1);
    let y: Vec<f32> = x.iter().map(|&v| function_1(v)).collect();
    let y2: Vec<f32> = tangent_line(&function_1, 5.0, &x);
    match plot("./images/gradient_1d.png", 480, 640, &x, &y, &y2) {
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
    y2: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(file_name, (width, height)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("y=f(x)", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x[0]..x[x.len() - 1], -y.to_vec().min()..y.to_vec().max())?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new((0..x.len()).map(|ii| (x[ii], y[ii])), &RED))?
        .label("y = function_1(x)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .draw_series(LineSeries::new(
            (0..x.len()).map(|ii| (x[ii], y2[ii])),
            &BLUE,
        ))?
        .label("y = tangent_line(x)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;
    Ok(())
}
