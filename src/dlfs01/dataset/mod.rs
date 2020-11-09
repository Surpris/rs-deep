//! dataset
//!
//!

#![allow(dead_code)]
const URL_BASE: &str = "http://yann.lecun.com/exdb/mnist/";
const KEY_FILE: [(&str, &str); 4] = [
    ("train_img", "train-images-idx3-ubyte.gz"),
    ("train_label", "train-labels-idx1-ubyte.gz"),
    ("test_img", "t10k-images-idx3-ubyte.gz"),
    ("test_label", "t10k-labels-idx1-ubyte.gz"),
];

const MNIST_DIR: &str = "/home/user/rust/data/mnist/";
const LENA_DIR: &str = "/home/user/rust/images/lena/";

const TRAIN_NUM: usize = 60000;
const TEST_NUM: usize = 10000;
const IMG_DIM: (usize, usize, usize) = (1, 28, 28);
const IMG_SIZE: usize = 784;

pub fn load_mnist() {}
