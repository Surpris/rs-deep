//! dataset
//!
//!

#![allow(dead_code)]

// use std::io::{BufReader};
use crate::dlfs01::{cast_t2u, Operators};
use ndarray::{Array, Array2, ArrayD};
use num_traits::{Num, NumCast};
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::path::Path;
use thiserror::Error as ThisError;

type Vec2d<T> = Vec<Vec<T>>;
type Vec3d<T> = Vec<Vec2d<T>>;
type Vec4d<T> = Vec<Vec3d<T>>;

const MNIST_DIR: &str = "/home/user/rust/projects/data/mnist/";
const LENA_DIR: &str = "/home/user/rust/projects/images/lena/";

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

#[derive(Clone)]
pub struct MNISTDataSetArray2<T> {
    pub train_images: Array2<T>,
    pub train_labels: Array2<T>,
    pub test_images: Array2<T>,
    pub test_labels: Array2<T>,
}

#[derive(Clone)]
pub struct MNISTDataSetArrayD<T> {
    pub train_images: ArrayD<T>,
    pub train_labels: ArrayD<T>,
    pub test_images: ArrayD<T>,
    pub test_labels: ArrayD<T>,
}

#[derive(Clone)]
pub struct MNISTDataSetFlattened<T> {
    pub train_images: Vec2d<T>,
    pub train_labels: Vec2d<T>,
    pub test_images: Vec2d<T>,
    pub test_labels: Vec2d<T>,
}

#[derive(Clone)]
pub struct MNISTDataSet<T> {
    pub train_images: Vec4d<T>,
    pub train_labels: Vec2d<T>,
    pub test_images: Vec4d<T>,
    pub test_labels: Vec2d<T>,
}

impl<T> MNISTDataSet<T>
where
    T: Num + Copy + NumCast + PartialOrd,
{
    pub fn flatten(self) -> MNISTDataSetFlattened<T> {
        let train_images: Vec2d<T> = self.train_images.iter().map(|v| v.flatten()).collect();
        let train_labels = self.train_labels;
        let test_images: Vec2d<T> = self.test_images.iter().map(|v| v.flatten()).collect();
        let test_labels = self.test_labels;
        MNISTDataSetFlattened {
            train_images,
            train_labels,
            test_images,
            test_labels,
        }
    }
    pub fn into_array2(self) -> MNISTDataSetArray2<T> {
        let img_size: usize = IMG_DIM.0 * IMG_DIM.1 * IMG_DIM.2;
        let target_size = NBR_CLASS;
        let train_images: Array2<T> = Array::from_shape_vec(
            (self.train_images.len(), img_size),
            self.train_images.flatten(),
        )
        .unwrap();
        let train_labels: Array2<T> = Array::from_shape_vec(
            (self.train_labels.len(), target_size),
            self.train_labels.flatten(),
        )
        .unwrap();
        let test_images: Array2<T> = Array::from_shape_vec(
            (self.test_images.len(), img_size),
            self.test_images.flatten(),
        )
        .unwrap();
        let test_labels: Array2<T> = Array::from_shape_vec(
            (self.test_labels.len(), target_size),
            self.test_labels.flatten(),
        )
        .unwrap();
        MNISTDataSetArray2 {
            train_images,
            train_labels,
            test_images,
            test_labels,
        }
    }
}

impl MNISTDataSet<u8> {
    pub fn to_f32(self) -> MNISTDataSet<f32> {
        MNISTDataSet::<f32> {
            train_images: to_f32_4d(self.train_images, IMG_MAX),
            train_labels: to_f32_2d(self.train_labels, 1),
            test_images: to_f32_4d(self.test_images, IMG_MAX),
            test_labels: to_f32_2d(self.test_labels, 1),
        }
    }
    pub fn to_f64(self) -> MNISTDataSet<f64> {
        MNISTDataSet::<f64> {
            train_images: to_f64_4d(self.train_images, IMG_MAX),
            train_labels: to_f64_2d(self.train_labels, 1),
            test_images: to_f64_4d(self.test_images, IMG_MAX),
            test_labels: to_f64_2d(self.test_labels, 1),
        }
    }
}

fn to_f32_1d(src: Vec<u8>, norm: u8) -> Vec<f32> {
    if norm == 1 {
        src.iter().map(|&v| v as f32).collect()
    } else {
        src.iter().map(|&v| v as f32 / norm as f32).collect()
    }
}

fn to_f32_2d(src: Vec2d<u8>, norm: u8) -> Vec2d<f32> {
    src.iter().map(|v| to_f32_1d(v.clone(), norm)).collect()
}

fn to_f32_3d(src: Vec3d<u8>, norm: u8) -> Vec3d<f32> {
    src.iter().map(|v| to_f32_2d(v.clone(), norm)).collect()
}

fn to_f32_4d(src: Vec4d<u8>, norm: u8) -> Vec4d<f32> {
    src.iter().map(|v| to_f32_3d(v.clone(), norm)).collect()
}

fn to_f64_1d(src: Vec<u8>, norm: u8) -> Vec<f64> {
    if norm == 1 {
        src.iter().map(|&v| v as f64).collect()
    } else {
        src.iter().map(|&v| v as f64 / norm as f64).collect()
    }
}

fn to_f64_2d(src: Vec2d<u8>, norm: u8) -> Vec2d<f64> {
    src.iter().map(|v| to_f64_1d(v.clone(), norm)).collect()
}

fn to_f64_3d(src: Vec3d<u8>, norm: u8) -> Vec3d<f64> {
    src.iter().map(|v| to_f64_2d(v.clone(), norm)).collect()
}

fn to_f64_4d(src: Vec4d<u8>, norm: u8) -> Vec4d<f64> {
    src.iter().map(|v| to_f64_3d(v.clone(), norm)).collect()
}

#[derive(ThisError, Debug)]
pub enum DataSetError {
    #[error("Index out of bound.")]
    IndexError,
    #[error("File IO error.")]
    FileIOError(#[from] io::Error),
}

pub fn load_mnist(verbose: u8) -> Result<MNISTDataSet<f64>, DataSetError> {
    if verbose > 0u8 {
        println!("load images for training...");
    }
    let file_path = Path::new(MNIST_DIR).join(KEY_FILE[0].1);
    let train_images = load_images(&file_path)?;

    if verbose > 0u8 {
        println!("load labels for training...");
    }
    let file_path = Path::new(MNIST_DIR).join(KEY_FILE[1].1);
    let train_labels = load_labels(&file_path)?;

    if verbose > 0u8 {
        println!("load images for test...");
    }
    let file_path = Path::new(MNIST_DIR).join(KEY_FILE[2].1);
    let test_images = load_images(&file_path)?;

    if verbose > 0u8 {
        println!("load labels for test...");
    }
    let file_path = Path::new(MNIST_DIR).join(KEY_FILE[3].1);
    let test_labels = load_labels(&file_path)?;
    let data_set: MNISTDataSet<u8> = MNISTDataSet {
        train_images,
        train_labels,
        test_images,
        test_labels,
    };

    println!("converting into f32...");
    Ok(data_set.to_f64())
}

pub fn load_images(file_path: &Path) -> Result<Vec4d<u8>, DataSetError> {
    let v = read_file(file_path)?;
    assert!(v.len() > NBR_SKIP_BYTES_IMAGE);
    let img_size: usize = IMG_DIM.0 * IMG_DIM.1 * IMG_DIM.2;
    let mut dst: Vec4d<u8> = Vec::new();
    let mut index: usize = NBR_SKIP_BYTES_IMAGE;
    while index < v.len() {
        match reshape_1d3d(&v[index..(index + img_size)].to_vec(), IMG_DIM) {
            Ok(v) => dst.push(v),
            Err(err) => return Err(err),
        }
        index += img_size;
    }
    Ok(dst)
}

pub fn reshape_1d3d(v: &Vec<u8>, shape: (usize, usize, usize)) -> Result<Vec3d<u8>, DataSetError> {
    let mut dst: Vec3d<u8> = vec![vec![vec![0u8; shape.2]; shape.1]; shape.0];
    if shape.0 * shape.1 * shape.2 != v.len() {
        return Err(DataSetError::IndexError);
    }
    for ii in 0..shape.0 {
        let ii_pos = ii * shape.1 * shape.2;
        for jj in 0..shape.1 {
            let jj_pos = jj * shape.1;
            for kk in 0..shape.2 {
                let pos: usize = ii_pos + jj_pos + kk;
                dst[ii][jj][kk] = v[pos];
            }
        }
    }
    Ok(dst)
}

pub fn load_labels(file_path: &Path) -> Result<Vec2d<u8>, DataSetError> {
    let v = read_file(file_path)?;
    assert!(v.len() > NBR_SKIP_BYTES_LABEL);
    let mut dst: Vec2d<u8> = Vec::new();
    for c in v[NBR_SKIP_BYTES_LABEL..].to_vec() {
        let v_: Vec<u8> = one_hot(c, NBR_CLASS)?;
        dst.push(v_);
    }
    Ok(dst)
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

pub fn print_image<T>(v: &Vec3d<T>)
where
    T: Num + NumCast + Copy + PartialOrd,
{
    for v1 in v {
        let mut s = String::new();
        for v2 in v1 {
            for c in v2 {
                if *c > cast_t2u(0.0) {
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

pub fn main() {
    println!("< dataset sub module >");
    let verbose: u8 = 1u8;
    let data_set: MNISTDataSet<f64> = load_mnist(verbose).unwrap();
    println!("train_labels: # = {}", data_set.train_labels.len());
    println!("First 10: {:?}", data_set.train_labels[..10].to_vec());
    println!("train_image: # = {}", data_set.train_images.len());
    println!("First 10:");
    for ii in 0..10 {
        print_image::<f64>(&data_set.train_images[ii]);
    }
    println!("flatten the data set...");
    let data_set = data_set.flatten();
    println!(
        "flattened data size of train and test: {} x {}, {} x {}",
        data_set.train_images.len(),
        data_set.train_images[0].len(),
        data_set.test_images.len(),
        data_set.test_images[0].len()
    );
}
