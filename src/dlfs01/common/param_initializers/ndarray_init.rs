//! ndarray_init
//!
//! Initializer of ndarray

use super::super::util::*;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::*;
use std::fmt::Display;

/// Enum of distributions generating random values
#[derive(Clone, Debug)]
pub enum DistributionEnum {
    Bernoulli = 0,
    Beta = 1,
    Binomial = 2,
    Cauchy = 3,
    ChiSquared = 4,
    Dirichlet = 5,
    Exp = 6,
    FisherF = 7,
    Gamma = 8,
    LogNormal = 9,
    Normal = 10,
    Pareto = 11,
    Pert = 12,
    Poisson = 13,
    StudentT = 14,
    Triangular = 15,
    Uniform = 16,
    Weibull = 17,
}

impl Display for DistributionEnum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DistributionEnum::Bernoulli => write!(f, "Bernoulli"),
            DistributionEnum::Beta => write!(f, "Beta"),
            DistributionEnum::Binomial => write!(f, "Binomial"),
            DistributionEnum::Cauchy => write!(f, "Cauchy"),
            DistributionEnum::ChiSquared => write!(f, "ChiSquared"),
            DistributionEnum::Dirichlet => write!(f, "Dirichlet"),
            DistributionEnum::Exp => write!(f, "Exp"),
            DistributionEnum::FisherF => write!(f, "FisherF"),
            DistributionEnum::Gamma => write!(f, "Gamma"),
            DistributionEnum::LogNormal => write!(f, "LogNormal"),
            DistributionEnum::Normal => write!(f, "Normal"),
            DistributionEnum::Pareto => write!(f, "Pareto"),
            DistributionEnum::Pert => write!(f, "Pert"),
            DistributionEnum::Poisson => write!(f, "Poisson"),
            DistributionEnum::StudentT => write!(f, "StudentT"),
            DistributionEnum::Triangular => write!(f, "Triangular"),
            DistributionEnum::Uniform => write!(f, "Uniform"),
            DistributionEnum::Weibull => write!(f, "Weibull"),
        }
    }
}

/// generate ndarray with random values
pub fn initialize_randomized_ndarray<T, D, Sh>(
    name: DistributionEnum,
    shape: Sh,
    params: &[T],
) -> Array<T, D>
where
    T: CrateFloat,
    D: Dimension,
    Sh: ShapeBuilder<Dim = D>,
{
    let mut rng = rand::thread_rng();
    match name {
        // DistributionEnum::Bernoulli => {
        //     let gen = Bernoulli::new(cast_t2u(params[0])).unwrap();
        //     Array::zeros(shape).map(|_| cast_t2u(gen.sample(&mut rng)))
        // },
        DistributionEnum::Beta => {
            let gen =
                Beta::new(cast_t2u::<T, f64>(params[0]), cast_t2u::<T, f64>(params[1])).unwrap();
            Array::<T, D>::zeros(shape).map(|_| cast_t2u(gen.sample(&mut rng)))
        }
        // DistributionEnum::Binomial => Array::random(shape, Binomial::new(params[0], params[1])),
        DistributionEnum::Cauchy => {
            let gen =
                Cauchy::new(cast_t2u::<T, f64>(params[0]), cast_t2u::<T, f64>(params[1])).unwrap();
            Array::<T, D>::zeros(shape).map(|_| cast_t2u(gen.sample(&mut rng)))
        }
        DistributionEnum::ChiSquared => {
            let gen = ChiSquared::new(cast_t2u::<T, f64>(params[0])).unwrap();
            Array::<T, D>::zeros(shape).map(|_| cast_t2u(gen.sample(&mut rng)))
        }
        // DistributionEnum::Dirichlet => {
        //     let gen = Dirichlet::new(cast_t2u::<T, f64>(params[0]), cast_t2u::<T, f64>(params[1]))
        //         .unwrap();
        //     Array::<T, D>::zeros(shape).map(|_| cast_t2u(gen.sample(&mut rng)))
        // },
        DistributionEnum::Exp => {
            let gen = Exp::new(cast_t2u::<T, f64>(params[0])).unwrap();
            Array::<T, D>::zeros(shape).map(|_| cast_t2u(gen.sample(&mut rng)))
        }
        DistributionEnum::FisherF => {
            let gen =
                FisherF::new(cast_t2u::<T, f64>(params[0]), cast_t2u::<T, f64>(params[1])).unwrap();
            Array::<T, D>::zeros(shape).map(|_| cast_t2u(gen.sample(&mut rng)))
        }
        DistributionEnum::Gamma => {
            let gen =
                Gamma::new(cast_t2u::<T, f64>(params[0]), cast_t2u::<T, f64>(params[1])).unwrap();
            Array::<T, D>::zeros(shape).map(|_| cast_t2u(gen.sample(&mut rng)))
        }
        DistributionEnum::LogNormal => {
            let gen = LogNormal::new(cast_t2u::<T, f64>(params[0]), cast_t2u::<T, f64>(params[1]))
                .unwrap();
            Array::<T, D>::zeros(shape).map(|_| cast_t2u(gen.sample(&mut rng)))
        }
        DistributionEnum::Normal => {
            let gen =
                Normal::new(cast_t2u::<T, f64>(params[0]), cast_t2u::<T, f64>(params[1])).unwrap();
            Array::<T, D>::zeros(shape).map(|_| cast_t2u(gen.sample(&mut rng)))
        }
        DistributionEnum::Pareto => {
            let gen =
                Pareto::new(cast_t2u::<T, f64>(params[0]), cast_t2u::<T, f64>(params[1])).unwrap();
            Array::<T, D>::zeros(shape).map(|_| cast_t2u(gen.sample(&mut rng)))
        }
        DistributionEnum::Pert => {
            let gen = Pert::new(
                cast_t2u::<T, f64>(params[0]),
                cast_t2u::<T, f64>(params[1]),
                cast_t2u::<T, f64>(params[2]),
            )
            .unwrap();
            Array::<T, D>::zeros(shape).map(|_| cast_t2u(gen.sample(&mut rng)))
        }
        DistributionEnum::Poisson => {
            let gen = Poisson::new(cast_t2u::<T, f64>(params[0])).unwrap();
            Array::<T, D>::zeros(shape).map(|_| cast_t2u::<f64, T>(gen.sample(&mut rng)))
        }
        DistributionEnum::StudentT => {
            let gen = StudentT::new(cast_t2u::<T, f64>(params[0])).unwrap();
            Array::<T, D>::zeros(shape).map(|_| cast_t2u(gen.sample(&mut rng)))
        }
        DistributionEnum::Triangular => {
            let gen = Triangular::new(
                cast_t2u::<T, f64>(params[0]),
                cast_t2u::<T, f64>(params[1]),
                cast_t2u::<T, f64>(params[2]),
            )
            .unwrap();
            Array::<T, D>::zeros(shape).map(|_| cast_t2u(gen.sample(&mut rng)))
        }
        DistributionEnum::Uniform => {
            let gen = Uniform::new(cast_t2u::<T, f64>(params[0]), cast_t2u::<T, f64>(params[1]));
            Array::<T, D>::zeros(shape).map(|_| cast_t2u(gen.sample(&mut rng)))
        }
        DistributionEnum::Weibull => {
            let gen =
                Weibull::new(cast_t2u::<T, f64>(params[0]), cast_t2u::<T, f64>(params[1])).unwrap();
            Array::<T, D>::zeros(shape).map(|_| cast_t2u(gen.sample(&mut rng)))
        }
        _ => panic!("Invalid distribution name: {}", name),
    }
}
