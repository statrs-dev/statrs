//! This crate aims to be a functional port of the Math.NET Numerics
//! Distribution package and in doing so providing the Rust numerical computing
//! community with a robust, well-tested statistical distribution package. This
//! crate also ports over some of the special statistical functions from
//! Math.NET in so far as they are used in the computation of distribution
//! values. This crate depends on the `rand` crate to provide RNG.
//!
//! # Sampling
//! The common use case is to set up the distributions and sample from them which depends on the `Rand` crate for random number generation.
//! ```
//! use statrs::distribution::Exp;
//! use rand::distributions::Distribution;
//! let mut r = rand::rngs::OsRng;
//! let n = Exp::new(0.5).unwrap();
//! print!("{}", n.sample(&mut r));
//! ```
//!
//! # Introspecting distributions
//! Statrs also comes with a number of useful utility traits for more detailed introspection of distributions.
//! ```
//! use statrs::distribution::{Exp, Continuous, ContinuousCDF}; // `cdf` and `pdf`
//! use statrs::statistics::Distribution; // statistical moments and entropy
//!
//! let n = Exp::new(1.0).unwrap();
//! assert_eq!(n.mean(), Some(1.0));
//! assert_eq!(n.variance(), Some(1.0));
//! assert_eq!(n.entropy(), Some(1.0));
//! assert_eq!(n.skewness(), Some(2.0));
//! assert_eq!(n.cdf(1.0), 0.6321205588285576784045);
//! assert_eq!(n.pdf(1.0), 0.3678794411714423215955);
//! ```
//!
//! # Utility functions
//! as well as utility functions including `erf`, `gamma`, `ln_gamma`, `beta`, etc.
//!
//! ```
//! use statrs::distribution::FisherSnedecor;
//! use statrs::statistics::Distribution;
//!
//! let n = FisherSnedecor::new(1.0, 1.0).unwrap();
//! assert!(n.variance().is_none());
//! ```
//! ## Distributions implemented
//! Statrs comes with a number of commonly used distributions including Normal, Gamma, Student's T, Exponential, Weibull, etc. view all implemented in `distributions` module.

#![crate_type = "lib"]
#![crate_name = "statrs"]
#![allow(clippy::excessive_precision)]
#![allow(clippy::many_single_char_names)]
#![allow(unused_imports)]
#![forbid(unsafe_code)]

#[macro_use]
extern crate approx;

#[macro_export]
macro_rules! assert_almost_eq {
    ($a:expr, $b:expr, $prec:expr $(,)?) => {
        if !$crate::prec::almost_eq($a, $b, $prec) {
            panic!(
                "assertion failed: `abs(left - right) < {:e}`, (left: `{}`, right: `{}`)",
                $prec, $a, $b
            );
        }
    };
}

pub mod consts;
#[macro_use]
pub mod distribution;
pub mod euclid;
pub mod function;
pub mod generate;
pub mod prec;
pub mod statistics;

mod error;

// function to silence clippy on the special case when comparing to zero.
#[inline(always)]
pub(crate) fn is_zero(x: f64) -> bool {
    ulps_eq!(x, 0.0, max_ulps = 0)
}

// #[cfg(test)]
mod testing;

pub use crate::error::StatsError;

/// Result type for the statrs library package that returns
/// either a result type `T` or a `StatsError`
pub type Result<T> = std::result::Result<T, StatsError>;
