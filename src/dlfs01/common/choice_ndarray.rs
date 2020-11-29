//! choice_ndarray
//!
//! choice trait for ndarray

use crate::dlfs01::cast_t2u;
use ndarray::{Array1, ArrayD, Axis};
use num_traits::{Num, NumCast};

pub trait Choice<T> {
    fn shuffle_by_indices(self, indices: &Array1<usize>, ax: usize) -> Self;
    fn shuffle_copy_by_indices(&self, indices: &Array1<usize>, ax: usize) -> Self;
}

impl<T> Choice<T> for ArrayD<T>
where
    T: Num + NumCast + Copy,
{
    fn shuffle_by_indices(self, indices: &Array1<usize>, ax: usize) -> Self {
        let one: T = cast_t2u(1.0);
        let mut dst: ArrayD<T> = ArrayD::zeros(self.raw_dim());
        // let mut ii: usize = 0;
        // for mut view in dst.axis_iter_mut(Axis(ax)) {
        //     let y = self.slice(s![indices[ii]..(indices[ii] + 1)]);
        //     view = self.slice(s![indices[ii]..indices[ii]+1]);
        //     ii += 1;
        // }
        dst
    }
    fn shuffle_copy_by_indices(&self, indices: &Array1<usize>, ax: usize) -> Self {
        self.clone()
    }
}
