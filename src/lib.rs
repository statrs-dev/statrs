//! This crate aims to be a functional port of the Math.NET Numerics
//! Distribution package and in doing so providing the Rust numerical computing
//! community with a robust, well-tested statistical distribution package. This
//! crate also ports over some of the special statistical functions from
//! Math.NET in so far as they are used in the computation of distribution
//! values. This crate depends on the `rand` crate to provide RNG.
//!
//! # Example
//! The following example samples from a standard normal distribution
//!
//! ```
//! # extern crate rand;
//! # extern crate statrs;
//! use rand::distributions::Distribution;
//! use statrs::distribution::Normal;
//!
//! # fn main() {
//! let mut r = rand::thread_rng();
//! let n = Normal::new(0.0, 1.0).unwrap();
//! for _ in 0..10 {
//!     print!("{}", n.sample(&mut r));
//! }
//! # }
//! ```

#![crate_type = "lib"]
#![crate_name = "statrs"]

extern crate rand;

#[macro_export]
macro_rules! assert_almost_eq {
    ($a:expr, $b:expr, $prec:expr) => {
        if !$crate::prec::almost_eq($a, $b, $prec) {
            panic!(format!(
                "assertion failed: `abs(left - right) < {:e}`, (left: `{}`, right: `{}`)",
                $prec, $a, $b
            ));
        }
    };
}

pub mod consts;
pub mod distribution;
pub mod euclid;
pub mod function;
pub mod generate;
pub mod prec;
pub mod statistics;

mod error;

#[cfg(test)]
mod testing;

pub use error::StatsError;

/// Result type for the statrs library package that returns
/// either a result type `T` or a `StatsError`
pub type Result<T> = std::result::Result<T, StatsError>;
