//! trainers
//!
//! Trainers for models

use super::models::ModelBase;
use super::util::*;
use ndarray::{prelude::*, RemoveAxis};
use rand::prelude::*;
use std::time::Instant;

#[derive(Clone)]
pub struct TrainResult<T: CrateFloat> {
    train_loss_list: Vec<T>,
    train_acc_list: Vec<T>,
    test_acc_list: Vec<T>,
}

impl<T> TrainResult<T>
where
    T: CrateFloat,
{
    pub fn new() -> Self {
        TrainResult {
            train_loss_list: Vec::new(),
            train_acc_list: Vec::new(),
            test_acc_list: Vec::new(),
        }
    }
    pub fn train_result(&self) -> (Vec<T>, Vec<T>, Vec<T>) {
        (
            self.train_loss_list.clone(),
            self.train_acc_list.clone(),
            self.test_acc_list.clone(),
        )
    }
}

pub struct Trainer<T: CrateFloat, D1: Dimension, D2: Dimension> {
    x_train: Array<T, D1>,
    t_train: Array<T, D2>,
    x_test: Array<T, D1>,
    t_test: Array<T, D2>,
    epochs: usize,
    batch_size: usize,
    batch_axis: usize,
    nbr_of_samples_per_epoch: usize,
    log_temporal_result: bool,
    verbose: usize,
    // optimizer: Box<dyn OptimizerBase<Src>>,
    train_loss_list: Vec<T>,
    train_acc_list: Vec<T>,
    test_acc_list: Vec<T>,
    iter_per_epoch: usize,
    train_size: usize,
    max_iter: usize,
    current_iter: usize,
    current_epoch: usize,
    sample_indices: Vec<usize>,
    elapsed_time: f64,
}

impl<T, D1, D2> Trainer<T, D1, D2>
where
    T: CrateFloat,
    D1: Dimension + RemoveAxis,
    D2: Dimension + RemoveAxis,
{
    pub fn new(
        x_train: Array<T, D1>,
        t_train: Array<T, D2>,
        x_test: Array<T, D1>,
        t_test: Array<T, D2>,
        batch_axis: usize,
        epochs: usize,
        batch_size: usize,
        nbr_of_samples_per_epoch: usize,
        log_temporal_result: bool,
        verbose: usize,
    ) -> Self {
        let train_size: usize = x_train.len_of(Axis(batch_axis));
        let iter_per_epoch: usize = usize::max(train_size / batch_size, 1);
        let max_iter: usize = epochs * iter_per_epoch;
        let sample_indices: Vec<usize> = if nbr_of_samples_per_epoch > 0 {
            let mut dst: Vec<usize> = Vec::new();
            for ii in 0..nbr_of_samples_per_epoch {
                dst.push(ii);
            }
            dst
        } else {
            Vec::new()
        };
        Self {
            x_train,
            t_train,
            x_test,
            t_test,
            epochs,
            batch_size,
            batch_axis,
            nbr_of_samples_per_epoch,
            log_temporal_result,
            verbose,
            train_loss_list: Vec::new(),
            train_acc_list: Vec::new(),
            test_acc_list: Vec::new(),
            iter_per_epoch,
            train_size,
            max_iter,
            current_iter: 0,
            current_epoch: 0,
            sample_indices,
            elapsed_time: 0.0f64,
        }
    }
    fn train_step(
        &mut self,
        model: &mut Box<dyn ModelBase<T, A = Array<T, D1>, B = Array<T, D2>>>,
    ) {
        let mut rng = thread_rng();
        // choose indices
        let mut indices: Vec<usize> = vec![0usize; self.batch_size];
        for jj in 0..self.batch_size {
            indices[jj] = rng.gen_range(0, self.x_train.len_of(Axis(self.batch_axis)));
        }
        let x_batch = self.x_train.select(Axis(0), &indices);
        let t_batch = self.t_train.select(Axis(0), &indices);
        model.update(&x_batch, &t_batch);
        if self.current_iter % self.iter_per_epoch == 0 {
            self.current_epoch += 1;
            if self.log_temporal_result {
                let train_acc: T;
                let test_acc: T;
                if self.nbr_of_samples_per_epoch > 0 {
                    train_acc = model.accuracy(
                        &self
                            .x_train
                            .select(Axis(self.batch_axis), &self.sample_indices),
                        &self
                            .t_train
                            .select(Axis(self.batch_axis), &self.sample_indices),
                    );
                    test_acc = model.accuracy(
                        &self
                            .x_test
                            .select(Axis(self.batch_axis), &self.sample_indices),
                        &self
                            .t_test
                            .select(Axis(self.batch_axis), &self.sample_indices),
                    );
                } else {
                    train_acc = model.accuracy(&self.x_train, &self.t_train);
                    test_acc = model.accuracy(&self.x_test, &self.t_test);
                }
                self.train_acc_list.push(train_acc);
                self.test_acc_list.push(test_acc);
            }
            if self.verbose > 0 {
                println!(
                    "train loss at step {}: {}",
                    self.current_iter,
                    model.get_current_loss()
                );
            }
        }
        self.current_iter += 1;
    }
    pub fn train(&mut self, model: &mut Box<dyn ModelBase<T, A = Array<T, D1>, B = Array<T, D2>>>) {
        let start = Instant::now();
        for _ in 0..self.max_iter {
            self.train_step(model);
        }
        let end = start.elapsed();
        self.elapsed_time = end.as_secs() as f64 + end.subsec_micros() as f64 * 1E-6;
    }
    pub fn get_results(&self) -> TrainResult<T> {
        TrainResult {
            train_loss_list: self.train_loss_list.clone(),
            train_acc_list: self.train_acc_list.clone(),
            test_acc_list: self.test_acc_list.clone(),
        }
    }
    pub fn get_elapsed_time(&self) -> f64 {
        self.elapsed_time
    }
    pub fn print_train_parameters(&self) {
        println!("# of train datasets: {}", self.train_size);
        println!("# of epochs: {}", self.epochs);
        println!("# of iterations per epoch: {}", self.iter_per_epoch);
        println!("batch size: {}", self.batch_size);
        println!("# of samples per epoch: {}", self.nbr_of_samples_per_epoch);
        println!("max of iterations: {}", self.max_iter);
        println!("batch axis: {}", self.batch_axis);
        println!("log temporal results: {}", self.log_temporal_result);
        println!("elapsed time: {}", self.elapsed_time);
    }
}
