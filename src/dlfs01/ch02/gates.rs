//! gates
//!
//!

use num_traits::{Num, NumCast};
use std::ops::AddAssign;

/// cast a numeric value with type T to one with U
fn cast_t2u<T, U>(x: T) -> U
where
    T: Num + NumCast + Copy,
    U: Num + NumCast + Copy,
{
    U::from(x).unwrap()
}

/// calculate the inner product of two arrays
fn inner_prod<T>(x: &[T], y: &[T]) -> T
where
    T: Num + NumCast + Copy + Default + AddAssign,
{
    let mut sum: T = cast_t2u(0.0);
    for ii in 0..x.len() {
        sum += x[ii] * y[ii];
    }
    cast_t2u(sum)
}

/// AND gate
fn and<T>(x1: T, x2: T) -> bool
where
    T: Num + NumCast + Copy + Default + AddAssign,
{
    let x: [T; 2] = [x1, x2];
    let weight: [T; 2] = [cast_t2u(0.5), cast_t2u(0.5)];
    let b: T = cast_t2u(-0.7);
    let tmp: T = inner_prod(&x, &weight) + b;
    if cast_t2u::<T, f64>(tmp) > 0.0 {
        true
    } else {
        false
    }
}

/// NAND gate
fn nand<T>(x1: T, x2: T) -> bool
where
    T: Num + NumCast + Copy + Default + AddAssign,
{
    let x: [T; 2] = [x1, x2];
    let weight: [T; 2] = [cast_t2u(-0.5), cast_t2u(-0.5)];
    let b: T = cast_t2u(0.7);
    let tmp: T = inner_prod(&x, &weight) + b;
    if cast_t2u::<T, f64>(tmp) > 0.0 {
        true
    } else {
        false
    }
}

/// OR gate
fn or<T>(x1: T, x2: T) -> bool
where
    T: Num + NumCast + Copy + Default + AddAssign,
{
    let x: [T; 2] = [x1, x2];
    let weight: [T; 2] = [cast_t2u(0.5), cast_t2u(0.5)];
    let b: T = cast_t2u(-0.2);
    let tmp: T = inner_prod(&x, &weight) + b;
    if cast_t2u::<T, f64>(tmp) > 0.0 {
        true
    } else {
        false
    }
}

/// XOR gate
fn xor<T>(x1: T, x2: T) -> bool
where
    T: Num + NumCast + Copy + Default + AddAssign,
{
    nand(x1, x2) & or(x1, x2)
}

pub fn main() {
    println!("< gates >");
    let pairs: [(f32, f32); 4] = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
    for xs in pairs.iter() {
        println!(
            "({}, {}) -> and: {}, nand: {}, or: {}, xor: {}",
            xs.0,
            xs.1,
            and(xs.0, xs.1),
            nand(xs.0, xs.1),
            or(xs.0, xs.1),
            xor(xs.0, xs.1)
        );
    }
}
