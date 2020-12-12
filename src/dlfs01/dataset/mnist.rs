//! mnist
//!
//! MNIST dataset

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::dlfs01::cast_t2u;
use ndarray::prelude::*;
use num_traits::{Num, NumCast};
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::path::Path;
use thiserror::Error as ThisError;

const MNIST_DIR: &str = "./data/mnist/";
const LENA_DIR: &str = "./images/lena/";

const URL_BASE: &str = "http://yann.lecun.com/exdb/mnist/";
const KEY_FILE_GZ: [(&str, &str); 4] = [
    ("train_img", "train-images-idx3-ubyte.gz"),
    ("train_label", "train-labels-idx1-ubyte.gz"),
    ("test_img", "t10k-images-idx3-ubyte.gz"),
    ("test_label", "t10k-labels-idx1-ubyte.gz"),
];
const KEY_FILE: [(&str, &str); 4] = [
    ("train_img", "train-images-idx3-ubyte"),
    ("train_label", "train-labels-idx1-ubyte"),
    ("test_img", "t10k-images-idx3-ubyte"),
    ("test_label", "t10k-labels-idx1-ubyte"),
];

const NBR_SKIP_BYTES_IMAGE: usize = 16;
const NBR_SKIP_BYTES_LABEL: usize = 8;

const NBR_TRAIN: usize = 60000;
const NBR_TEST: usize = 10000;
const IMG_DIM: (usize, usize, usize) = (1, 28, 28);
const IMG_SIZE: usize = 784;
const IMG_MAX: u8 = 255;
const NBR_CLASS: usize = 10;

#[derive(ThisError, Debug)]
pub enum DataSetError {
    #[error("Wrong dimension.")]
    DimensionError,
    #[error("Index out of bound.")]
    IndexError,
    #[error("File IO error.")]
    FileIOError(#[from] io::Error),
    #[error("failure in reshaping array.")]
    ShapeError(#[from] ndarray::ShapeError),
}

#[derive(Clone)]
pub struct MNISTDataSet<T, D> {
    pub train_images: Array<T, D>,
    pub train_labels: Array2<T>,
    pub test_images: Array<T, D>,
    pub test_labels: Array2<T>,
}

pub type MNISTDataSet2<T> = MNISTDataSet<T, Ix2>;
pub type MNISTDataSet4<T> = MNISTDataSet<T, Ix4>;

impl<D> MNISTDataSet<u8, D>
where
    D: Dimension,
{
    pub fn to_f64(self) -> MNISTDataSet<f64, D> {
        MNISTDataSet::<f64, D> {
            train_images: self.train_images.map(|&v| v as f64 / IMG_MAX as f64),
            train_labels: self.train_labels.map(|&v| v as f64 / 1.0),
            test_images: self.test_images.map(|&v| v as f64 / IMG_MAX as f64),
            test_labels: self.test_labels.map(|&v| v as f64 / 1.0),
        }
    }
}

impl MNISTDataSet2<u8> {
    pub fn new(verbose: u8) -> Result<Self, DataSetError> {
        if verbose > 0u8 {
            println!("load images for training...");
        }
        let file_path = Path::new(MNIST_DIR).join(KEY_FILE[0].1);
        let train_images = load_images_2d(&file_path)?;

        if verbose > 0u8 {
            println!("load labels for training...");
        }
        let file_path = Path::new(MNIST_DIR).join(KEY_FILE[1].1);
        let train_labels = load_labels(&file_path)?;

        if verbose > 0u8 {
            println!("load images for test...");
        }
        let file_path = Path::new(MNIST_DIR).join(KEY_FILE[2].1);
        let test_images = load_images_2d(&file_path)?;

        if verbose > 0u8 {
            println!("load labels for test...");
        }
        let file_path = Path::new(MNIST_DIR).join(KEY_FILE[3].1);
        let test_labels = load_labels(&file_path)?;
        Ok(Self {
            train_images,
            train_labels,
            test_images,
            test_labels,
        })
    }
}

impl MNISTDataSet4<u8> {
    pub fn new(verbose: u8) -> Result<Self, DataSetError> {
        if verbose > 0u8 {
            println!("load images for training...");
        }
        let file_path = Path::new(MNIST_DIR).join(KEY_FILE[0].1);
        let train_images = load_images_4d(&file_path)?;

        if verbose > 0u8 {
            println!("load labels for training...");
        }
        let file_path = Path::new(MNIST_DIR).join(KEY_FILE[1].1);
        let train_labels = load_labels(&file_path)?;

        if verbose > 0u8 {
            println!("load images for test...");
        }
        let file_path = Path::new(MNIST_DIR).join(KEY_FILE[2].1);
        let test_images = load_images_4d(&file_path)?;

        if verbose > 0u8 {
            println!("load labels for test...");
        }
        let file_path = Path::new(MNIST_DIR).join(KEY_FILE[3].1);
        let test_labels = load_labels(&file_path)?;
        Ok(Self {
            train_images,
            train_labels,
            test_images,
            test_labels,
        })
    }
}

pub fn load_images_4d(file_path: &Path) -> Result<Array4<u8>, DataSetError> {
    let v = read_file(file_path)?;
    if v.len() <= NBR_SKIP_BYTES_IMAGE {
        return Err(DataSetError::IndexError);
    }
    let v = v[NBR_SKIP_BYTES_IMAGE..].to_vec();
    let shape = (v.len() / IMG_SIZE, IMG_DIM.0, IMG_DIM.1, IMG_DIM.2);
    let dst = Array::from_shape_vec(shape, v)?;
    Ok(dst)
}

pub fn load_images_2d(file_path: &Path) -> Result<Array2<u8>, DataSetError> {
    let v = read_file(file_path)?;
    if v.len() <= NBR_SKIP_BYTES_IMAGE {
        return Err(DataSetError::IndexError);
    }
    let v = v[NBR_SKIP_BYTES_IMAGE..].to_vec();
    let shape = (v.len() / IMG_SIZE, IMG_SIZE);
    let dst = Array::from_shape_vec(shape, v)?;
    Ok(dst)
}

pub fn load_labels(file_path: &Path) -> Result<Array2<u8>, DataSetError> {
    let v = read_file(file_path)?;
    assert!(v.len() > NBR_SKIP_BYTES_LABEL);
    let v = v[NBR_SKIP_BYTES_LABEL..].to_vec();
    let shape = (v.len(), NBR_CLASS);
    let mut dst: Vec<u8> = Vec::new();
    for c in v {
        dst.append(&mut one_hot(c, NBR_CLASS)?);
    }
    let x = Array2::from_shape_vec(shape, dst)?;
    Ok(x)
}

pub fn one_hot(c: u8, size: usize) -> Result<Vec<u8>, DataSetError> {
    if c as usize > size {
        return Err(DataSetError::IndexError);
    }
    let mut dst: Vec<u8> = vec![0u8; size];
    dst[c as usize] = 1u8;
    Ok(dst)
}

pub fn read_file(file_path: &Path) -> Result<Vec<u8>, io::Error> {
    let pb = file_path.canonicalize()?;
    let mut file = File::open(pb)?;
    let mut buf: Vec<u8> = Vec::new();
    let _ = file.read_to_end(&mut buf)?;
    Ok(buf)
}

pub fn print_image<T>(v: &Array3<T>)
where
    T: Num + NumCast + Copy + PartialOrd,
{
    let zero: T = cast_t2u(0.0);
    for v1 in v.axis_iter(Axis(0)) {
        let mut s = String::new();
        for v2 in v1.axis_iter(Axis(0)) {
            for c in v2 {
                if *c > zero {
                    s += " ";
                } else {
                    s += "#";
                }
            }
            s += "\n";
        }
        println!("{}", s);
        println!("");
    }
}
