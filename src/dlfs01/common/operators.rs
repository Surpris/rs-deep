//! ops
//!
//! operators

use super::util::cast_t2u;
use num_traits::Float;

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
    T: Float,
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

impl<T> Operators<T> for Vec<Vec<T>>
where
    T: Float,
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
        let mut dst: Vec<Vec<T>> = Vec::new();
        for ii in 0..self[0].len() {
            dst.push((0..self.len()).map(|jj| self[jj][ii]).collect());
        }
        dst
    }
}
