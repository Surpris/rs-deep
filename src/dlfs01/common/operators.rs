//! ops
//!
//! operators

use super::math::MathFunc;
use super::util::cast_t2u;
use num_traits::{Float, Num, NumCast};

type Vec2d<T> = Vec<Vec<T>>;
type Vec3d<T> = Vec<Vec2d<T>>;
type Vec4d<T> = Vec<Vec3d<T>>;

// >>>>>>>>>>>>> Operators >>>>>>>>>>>>>

/// Operators trait
pub trait Operators<T> {
    /// addition
    fn add_value(&self, other: T) -> Self;
    /// subtraction
    fn sub_value(&self, other: T) -> Self;
    /// multiplication
    fn mul_value(&self, other: T) -> Self;
    /// division
    fn div_value(&self, other: T) -> Self;
    /// addition
    fn add(&self, other: &Self) -> Self;
    /// subtraction
    fn sub(&self, other: &Self) -> Self;
    /// multiplication
    fn mul(&self, other: &Self) -> Self;
    /// division
    fn div(&self, other: &Self) -> Self;
    /// zeros_like
    fn zeros_like(&self) -> Self;
    /// ones_like
    fn ones_like(&self) -> Self;
    /// flatten
    fn flatten(&self) -> Vec<T>;
    /// transpose
    fn transpose(&self) -> Self;
}

impl<T> Operators<T> for Vec<T>
where
    T: Num + Copy + NumCast + PartialOrd,
{
    fn add_value(&self, other: T) -> Self {
        self.iter().map(|&v| v + other).collect()
    }
    fn sub_value(&self, other: T) -> Self {
        self.iter().map(|&v| v - other).collect()
    }
    fn mul_value(&self, other: T) -> Self {
        self.iter().map(|&v| v * other).collect()
    }
    fn div_value(&self, other: T) -> Self {
        self.iter().map(|&v| v / other).collect()
    }
    fn add(&self, other: &Self) -> Self {
        (0..self.len()).map(|ii| self[ii] + other[ii]).collect()
    }
    fn sub(&self, other: &Self) -> Self {
        (0..self.len()).map(|ii| self[ii] - other[ii]).collect()
    }
    fn mul(&self, other: &Self) -> Self {
        (0..self.len()).map(|ii| self[ii] * other[ii]).collect()
    }
    fn div(&self, other: &Self) -> Self {
        (0..self.len()).map(|ii| self[ii] / other[ii]).collect()
    }
    fn zeros_like(&self) -> Self {
        let zero: T = cast_t2u(0.0);
        (0..self.len()).map(|_| zero).collect()
    }
    fn ones_like(&self) -> Self {
        let one: T = cast_t2u(1.0);
        (0..self.len()).map(|_| one).collect()
    }
    fn flatten(&self) -> Vec<T> {
        self.clone()
    }
    fn transpose(&self) -> Self {
        self.clone()
    }
}

impl<T> Operators<T> for Vec2d<T>
where
    T: Num + Copy + NumCast + PartialOrd,
{
    fn add_value(&self, other: T) -> Self {
        (0..self.len())
            .map(|ii| self[ii].clone().add_value(other))
            .collect()
    }
    fn sub_value(&self, other: T) -> Self {
        (0..self.len())
            .map(|ii| self[ii].clone().sub_value(other))
            .collect()
    }
    fn mul_value(&self, other: T) -> Self {
        (0..self.len())
            .map(|ii| self[ii].clone().mul_value(other))
            .collect()
    }
    fn div_value(&self, other: T) -> Self {
        (0..self.len())
            .map(|ii| self[ii].clone().div_value(other))
            .collect()
    }
    fn add(&self, other: &Self) -> Self {
        (0..self.len())
            .map(|ii| self[ii].clone().add(&other[ii]))
            .collect()
    }
    fn sub(&self, other: &Self) -> Self {
        (0..self.len())
            .map(|ii| self[ii].clone().sub(&other[ii]))
            .collect()
    }
    fn mul(&self, other: &Self) -> Self {
        (0..self.len())
            .map(|ii| self[ii].clone().mul(&other[ii]))
            .collect()
    }
    fn div(&self, other: &Self) -> Self {
        (0..self.len())
            .map(|ii| self[ii].clone().div(&other[ii]))
            .collect()
    }
    fn zeros_like(&self) -> Self {
        let zero: T = cast_t2u(0.0);
        let size1d: usize = self[0].len();
        (0..self.len()).map(|_| vec![zero; size1d]).collect()
    }
    fn ones_like(&self) -> Self {
        let one: T = cast_t2u(1.0);
        let size1d: usize = self[0].len();
        (0..self.len()).map(|_| vec![one; size1d]).collect()
    }
    fn flatten(&self) -> Vec<T> {
        let mut v: Vec<T> = Vec::new();
        for ii in 0..self.len() {
            v.append(&mut self[ii].clone());
        }
        v
    }
    fn transpose(&self) -> Self {
        let mut dst: Vec2d<T> = Vec::new();
        for ii in 0..self[0].len() {
            dst.push((0..self.len()).map(|jj| self[jj][ii]).collect());
        }
        dst
    }
}

impl<T> Operators<T> for Vec3d<T>
where
    T: Num + Copy + NumCast + PartialOrd,
{
    fn add_value(&self, other: T) -> Self {
        (0..self.len())
            .map(|ii| self[ii].clone().add_value(other))
            .collect()
    }
    fn sub_value(&self, other: T) -> Self {
        (0..self.len())
            .map(|ii| self[ii].clone().sub_value(other))
            .collect()
    }
    fn mul_value(&self, other: T) -> Self {
        (0..self.len())
            .map(|ii| self[ii].clone().mul_value(other))
            .collect()
    }
    fn div_value(&self, other: T) -> Self {
        (0..self.len())
            .map(|ii| self[ii].clone().div_value(other))
            .collect()
    }
    fn add(&self, other: &Self) -> Self {
        (0..self.len())
            .map(|ii| self[ii].clone().add(&other[ii]))
            .collect()
    }
    fn sub(&self, other: &Self) -> Self {
        (0..self.len())
            .map(|ii| self[ii].clone().sub(&other[ii]))
            .collect()
    }
    fn mul(&self, other: &Self) -> Self {
        (0..self.len())
            .map(|ii| self[ii].clone().mul(&other[ii]))
            .collect()
    }
    fn div(&self, other: &Self) -> Self {
        (0..self.len())
            .map(|ii| self[ii].clone().div(&other[ii]))
            .collect()
    }
    fn zeros_like(&self) -> Self {
        let zero: T = cast_t2u(0.0);
        let size1d: usize = self[0].len();
        let size2d: usize = self[0][0].len();
        (0..self.len())
            .map(|_| vec![vec![zero; size2d]; size1d])
            .collect()
    }
    fn ones_like(&self) -> Self {
        let one: T = cast_t2u(1.0);
        let size1d: usize = self[0].len();
        let size2d: usize = self[0][0].len();
        (0..self.len())
            .map(|_| vec![vec![one; size2d]; size1d])
            .collect()
    }
    fn flatten(&self) -> Vec<T> {
        let mut v: Vec<T> = Vec::new();
        for ii in 0..self.len() {
            v.append(&mut self[ii].clone().flatten());
        }
        v
    }
    fn transpose(&self) -> Self {
        //TODO: implement correctly
        self.clone()
    }
}

impl<T> Operators<T> for Vec4d<T>
where
    T: Num + Copy + NumCast + PartialOrd,
{
    fn add_value(&self, other: T) -> Self {
        (0..self.len())
            .map(|ii| self[ii].clone().add_value(other))
            .collect()
    }
    fn sub_value(&self, other: T) -> Self {
        (0..self.len())
            .map(|ii| self[ii].clone().sub_value(other))
            .collect()
    }
    fn mul_value(&self, other: T) -> Self {
        (0..self.len())
            .map(|ii| self[ii].clone().mul_value(other))
            .collect()
    }
    fn div_value(&self, other: T) -> Self {
        (0..self.len())
            .map(|ii| self[ii].clone().div_value(other))
            .collect()
    }
    fn add(&self, other: &Self) -> Self {
        (0..self.len())
            .map(|ii| self[ii].clone().add(&other[ii]))
            .collect()
    }
    fn sub(&self, other: &Self) -> Self {
        (0..self.len())
            .map(|ii| self[ii].clone().sub(&other[ii]))
            .collect()
    }
    fn mul(&self, other: &Self) -> Self {
        (0..self.len())
            .map(|ii| self[ii].clone().mul(&other[ii]))
            .collect()
    }
    fn div(&self, other: &Self) -> Self {
        (0..self.len())
            .map(|ii| self[ii].clone().div(&other[ii]))
            .collect()
    }
    fn zeros_like(&self) -> Self {
        let zero: T = cast_t2u(0.0);
        let size1d: usize = self[0].len();
        let size2d: usize = self[0][0].len();
        let size3d: usize = self[0][0][0].len();
        (0..self.len())
            .map(|_| vec![vec![vec![zero; size3d]; size2d]; size1d])
            .collect()
    }
    fn ones_like(&self) -> Self {
        let one: T = cast_t2u(1.0);
        let size1d: usize = self[0].len();
        let size2d: usize = self[0][0].len();
        let size3d: usize = self[0][0][0].len();
        (0..self.len())
            .map(|_| vec![vec![vec![one; size3d]; size2d]; size1d])
            .collect()
    }
    fn flatten(&self) -> Vec<T> {
        let mut v: Vec<T> = Vec::new();
        for ii in 0..self.len() {
            v.append(&mut self[ii].clone().flatten());
        }
        v
    }
    fn transpose(&self) -> Self {
        //TODO: implement correctly
        self.clone()
    }
}

// <<<<<<<<<<<<< Operators <<<<<<<<<<<<<

// >>>>>>>>>>>>> Fundamental operators >>>>>>>>>>>>>

/// inner product of 1D vectors
pub fn dot_1d<T>(x: &Vec<T>, y: &Vec<T>) -> T
where
    T: Float,
{
    assert_eq!(x.len(), y.len());
    (0..x.len())
        .map(|ii| x[ii] * y[ii])
        .collect::<Vec<T>>()
        .sum()
}

/// inner product of 1D and 2D vectors
pub fn dot_1d_2d<T>(x: &Vec<T>, y: &Vec2d<T>) -> Vec<T>
where
    T: Float,
{
    assert_eq!(x.len(), y.len());
    let mut dst: Vec<T> = Vec::new();
    for jj in 0..y[0].len() {
        dst.push(
            (0..y.len())
                .map(|ii| y[ii][jj] * x[ii])
                .collect::<Vec<T>>()
                .sum(),
        );
    }
    dst
}

/// inner product of 2D and 1D vectors
pub fn dot_2d_1d<T>(x: &Vec2d<T>, y: &Vec<T>) -> Vec<T>
where
    T: Float,
{
    assert_eq!(x[0].len(), y.len());
    let mut dst: Vec<T> = Vec::new();
    for jj in 0..x.len() {
        dst.push(
            (0..x[0].len())
                .map(|ii| x[jj][ii] * y[ii])
                .collect::<Vec<T>>()
                .sum(),
        );
    }
    dst
}

/// inner product of 2D vectors
pub fn dot_2d_2d<T>(x: &Vec2d<T>, y: &Vec2d<T>) -> Vec2d<T>
where
    T: Float,
{
    assert_eq!(x[0].len(), y.len());
    let mut dst: Vec2d<T> = vec![vec![cast_t2u(0.0); y[0].len()]; x.len()];
    for ii in 0..x.len() {
        for jj in 0..y[0].len() {
            dst[ii][jj] = (0..x[0].len())
                .map(|kk| x[ii][kk] * y[kk][jj])
                .collect::<Vec<T>>()
                .sum();
        }
    }
    dst
}

// <<<<<<<<<<<<< Fundamental operators <<<<<<<<<<<<<
