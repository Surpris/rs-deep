//! choice
//!
//! choice operator
//!
//! # TODO
//! implementation of arbitrary-axis mode

use num_traits::{Num, NumCast};

type Vec2d<T> = Vec<Vec<T>>;
type Vec3d<T> = Vec<Vec2d<T>>;
type Vec4d<T> = Vec<Vec3d<T>>;

pub trait Choice<T> {
    fn shuffle_by_indices(self, indices: &[usize]) -> Self;
    fn shuffle_copy_by_indices(&self, indices: &[usize]) -> Self;
}

impl<T> Choice<T> for Vec<T>
where
    T: Num + NumCast + Copy,
{
    fn shuffle_by_indices(self, indices: &[usize]) -> Self {
        let mut dst: Vec<T> = Vec::new();
        for v in indices {
            dst.push(self[*v]);
        }
        dst
    }
    fn shuffle_copy_by_indices(&self, indices: &[usize]) -> Self {
        let mut dst: Vec<T> = Vec::new();
        for v in indices {
            dst.push(self[*v]);
        }
        dst
    }
}

impl<T> Choice<T> for Vec2d<T>
where
    T: Num + NumCast + Copy,
{
    fn shuffle_by_indices(self, indices: &[usize]) -> Self {
        let mut dst: Vec2d<T> = Vec::new();
        for v in indices {
            dst.push(self[*v].clone());
        }
        dst
    }
    fn shuffle_copy_by_indices(&self, indices: &[usize]) -> Self {
        let mut dst: Vec2d<T> = Vec::new();
        for v in indices {
            dst.push(self[*v].clone());
        }
        dst
    }
}

impl<T> Choice<T> for Vec3d<T>
where
    T: Num + NumCast + Copy,
{
    fn shuffle_by_indices(self, indices: &[usize]) -> Self {
        let mut dst: Vec3d<T> = Vec::new();
        for v in indices {
            dst.push(self[*v].clone());
        }
        dst
    }
    fn shuffle_copy_by_indices(&self, indices: &[usize]) -> Self {
        let mut dst: Vec3d<T> = Vec::new();
        for v in indices {
            dst.push(self[*v].clone());
        }
        dst
    }
}

impl<T> Choice<T> for Vec4d<T>
where
    T: Num + NumCast + Copy,
{
    fn shuffle_by_indices(self, indices: &[usize]) -> Self {
        let mut dst: Vec4d<T> = Vec::new();
        for v in indices {
            dst.push(self[*v].clone());
        }
        dst
    }
    fn shuffle_copy_by_indices(&self, indices: &[usize]) -> Self {
        let mut dst: Vec4d<T> = Vec::new();
        for v in indices {
            dst.push(self[*v].clone());
        }
        dst
    }
}
