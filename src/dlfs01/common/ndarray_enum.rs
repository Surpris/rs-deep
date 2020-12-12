//! ndarray_enum
//! 
//! Enum consisting of Array types in ndarray crate

use ndarray::prelude::*;

pub enum ArrayEnum<T> {
    Array1(Array1<T>),
    Array2(Array2<T>),
    Array3(Array3<T>),
    Array4(Array4<T>),
    Array5(Array5<T>),
    Array6(Array6<T>),
    ArrayD(ArrayD<T>),
}
