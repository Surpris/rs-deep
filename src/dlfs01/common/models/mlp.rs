//! mlp
//!
//! Multi-layer perceptron model
//!
//! This model can be used only with datasets composed of 1D data.

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use ndarray::prelude::*;
use ndarray_stats::QuantileExt;

use super::super::layers::*;
use super::super::optimizers::*;
use super::super::param_initializers::weight_init::WeightInitEnum;
use super::super::util::*;
use super::model_base::ModelBase;

/// MLP classifier
pub struct MLPClassifier<T: CrateFloat> {
    affines: Vec<Affine<T>>,
    activators: Vec<Box<dyn LayerBase<T, A = Array2<T>, B = Array2<T>>>>,
    loss_layer: Box<dyn LossLayerBase<T, A = Array2<T>>>,
    optimizer_weight: Box<dyn OptimizerBase<Src = Array2<T>>>,
    optimizer_bias: Box<dyn OptimizerBase<Src = Array1<T>>>,
    current_loss: T,
    nbr_of_hiddens: usize,
    nbr_of_affines: usize,
}

impl<T: 'static> MLPClassifier<T>
where
    T: CrateFloat,
{
    pub fn new(
        input_size: usize,
        hidden_sizes: &[usize],
        output_size: usize,
        activator_enums: &[ActivatorEnum],
        optimizer_enum: OptimizerEnum,
        optimizer_params: &[T],
        batch_axis: usize,
        weight_init: WeightInitEnum,
        weight_init_std: T,
    ) -> Self {
        assert_eq!(hidden_sizes.len(), activator_enums.len());
        let nbr_of_hiddens: usize = hidden_sizes.len();
        let mut affines: Vec<Affine<T>> = Vec::new();
        let mut activators: Vec<Box<dyn LayerBase<T, A = Array<T, Ix2>, B = Array<T, Ix2>>>> =
            Vec::new();
        for ii in 0..nbr_of_hiddens {
            if ii == 0 {
                affines.push(Affine::new(
                    (input_size, hidden_sizes[ii]),
                    weight_init.clone(),
                    weight_init_std,
                ));
            } else {
                affines.push(Affine::new(
                    (hidden_sizes[ii - 1], hidden_sizes[ii]),
                    weight_init.clone(),
                    weight_init_std,
                ));
            }
            activators.push(call_activator(
                activator_enums[ii].clone(),
                (hidden_sizes[ii], hidden_sizes[ii]),
                batch_axis,
            ));
        }
        affines.push(Affine::new(
            (hidden_sizes[nbr_of_hiddens - 1], output_size),
            weight_init,
            weight_init_std,
        ));
        let loss_layer = Box::new(SoftmaxWithLoss2::new(
            (output_size, output_size),
            batch_axis,
        ));
        let optimizer_weight = call_optimizer(
            optimizer_enum.clone(),
            (hidden_sizes[hidden_sizes.len() - 1], output_size),
            optimizer_params,
        );
        let optimizer_bias = call_optimizer(
            optimizer_enum,
            hidden_sizes[hidden_sizes.len() - 1],
            optimizer_params,
        );
        let nbr_of_affines: usize = affines.len();
        Self {
            affines,
            activators,
            loss_layer,
            optimizer_weight,
            optimizer_bias,
            current_loss: cast_t2u(0.0),
            nbr_of_hiddens,
            nbr_of_affines,
        }
    }
    pub fn print_parameters(&self) {
        for ii in 0..self.nbr_of_hiddens {
            println!("w{}: {:?}", ii, self.affines[ii].weight);
            println!("b{}: {:?}", ii, self.affines[ii].bias);
            println!("dw{}: {:?}", ii, self.affines[ii].dw);
            println!("db{}: {:?}", ii, self.affines[ii].db);
        }
        println!(
            "w{}: {:?}",
            self.affines.len() - 1,
            self.affines[self.affines.len() - 1].weight
        );
        println!(
            "b{}: {:?}",
            self.affines.len() - 1,
            self.affines[self.affines.len() - 1].bias
        );
        println!(
            "dw{}: {:?}",
            self.affines.len() - 1,
            self.affines[self.affines.len() - 1].dw
        );
        println!(
            "db{}: {:?}",
            self.affines.len() - 1,
            self.affines[self.affines.len() - 1].db
        );
    }
}

impl<T: 'static> ModelBase<T> for MLPClassifier<T>
where
    T: CrateFloat,
{
    type A = Array2<T>;

    type B = Array2<T>;

    fn predict_prob(&mut self, x: &Self::A) -> Self::B {
        let mut y: Self::B = self.affines[0].forward(x);
        for ii in 0..self.nbr_of_hiddens {
            y = self.activators[ii].forward(&y);
            y = self.affines[ii + 1].forward(&y);
        }
        y
    }

    fn predict(&mut self, x: &Self::A) -> Self::B {
        let one: T = cast_t2u(1.0);
        let y: Self::B = self.predict_prob(&x);
        let mut dst: Self::B = Array2::zeros(y.raw_dim());
        for (view1, mut view2) in y.axis_iter(Axis(0)).zip(dst.axis_iter_mut(Axis(0))) {
            let y_argmax = view1.argmax().unwrap();
            view2[y_argmax] = one;
        }
        dst
    }

    fn loss(&mut self, x: &Self::A, t: &Self::B) -> T {
        let y: Self::B = self.predict_prob(&x);
        self.current_loss = self.loss_layer.forward(&y, &t);
        self.current_loss
    }

    fn accuracy(&mut self, x: &Self::A, t: &Self::B) -> T {
        let y: Self::B = self.predict(&x);
        let mut acc: f32 = 0.0;
        for (view1, view2) in y.axis_iter(Axis(0)).zip(t.axis_iter(Axis(0))) {
            let y_argmax = view1.argmax().unwrap();
            let t_argmax = view2.argmax().unwrap();
            if y_argmax == t_argmax {
                acc += 1.0;
            }
        }
        cast_t2u(acc / t.len_of(Axis(0)) as f32)
    }

    fn gradient(&mut self, x: &Self::A, t: &Self::B) {
        // forward
        let _ = self.loss(&x, &t);

        // backward
        let _dx: T = cast_t2u(1.0);
        let mut _dx: Self::B = self.loss_layer.backward(_dx);
        for ii in 0..self.activators.len() {
            _dx = self.affines[self.nbr_of_affines - 1 - ii].backward(&_dx);
            _dx = self.activators[self.nbr_of_hiddens - 1 - ii].backward(&_dx);
        }
        _dx = self.affines[0].backward(&_dx);
    }

    fn update(&mut self, x: &Self::A, t: &Self::B) {
        self.gradient(&x, &t);
        for layer in self.affines.iter_mut() {
            self.optimizer_weight
                .update(&mut layer.weight, &mut layer.dw);
            self.optimizer_bias.update(&mut layer.bias, &mut layer.db);
        }
    }

    fn print_detail(&self) {
        println!("MLP classifier.");
        for ii in 0..self.activators.len() {
            self.affines[ii].print_detail();
            self.activators[ii].print_detail();
        }
        self.affines[self.nbr_of_affines - 1].print_detail();
        self.loss_layer.print_detail();
    }

    fn get_current_loss(&self) -> T {
        self.current_loss
    }

    fn get_output(&self) -> Self::B {
        self.loss_layer.get_output()
    }
}
