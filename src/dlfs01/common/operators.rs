//! ops
//!
//! operators

use num_traits::Float;

pub trait Operators<T> {
    /// addition
    fn add(self, other: &Self) -> Self;
    /// subtraction
    fn sub(self, other: &Self) -> Self;
    /// multiplication
    fn mul(self, other: &Self) -> Self;
    /// division
    fn div(self, other: &Self) -> Self;
}

impl<T> Operators<T> for Vec<T>
where
    T: Float,
{
    fn add(self, other: &Self) -> Self {
        (0..self.len()).map(|ii| self[ii] + other[ii]).collect()
    }
    fn sub(self, other: &Self) -> Self {
        (0..self.len()).map(|ii| self[ii] - other[ii]).collect()
    }
    fn mul(self, other: &Self) -> Self {
        (0..self.len()).map(|ii| self[ii] * other[ii]).collect()
    }
    fn div(self, other: &Self) -> Self {
        (0..self.len()).map(|ii| self[ii] / other[ii]).collect()
    }
}
