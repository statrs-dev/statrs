//! Defines common interfaces for interacting with statistical distributions
//! and provides
//! concrete implementations for a variety of distributions.

use rand::Rng;
use statistics::{Max, Min};

pub use self::bernoulli::Bernoulli;
pub use self::beta::Beta;
pub use self::binomial::Binomial;
pub use self::categorical::Categorical;
pub use self::cauchy::Cauchy;
pub use self::chi::Chi;
pub use self::chi_squared::ChiSquared;
pub use self::dirichlet::Dirichlet;
pub use self::discrete_uniform::DiscreteUniform;
pub use self::erlang::Erlang;
pub use self::exponential::Exponential;
pub use self::fisher_snedecor::FisherSnedecor;
pub use self::gamma::Gamma;
pub use self::geometric::Geometric;
pub use self::hypergeometric::Hypergeometric;
pub use self::inverse_gamma::InverseGamma;
pub use self::log_normal::LogNormal;
pub use self::multinomial::Multinomial;
pub use self::normal::Normal;
pub use self::pareto::Pareto;
pub use self::poisson::Poisson;
pub use self::students_t::StudentsT;
pub use self::triangular::Triangular;
pub use self::uniform::Uniform;
pub use self::weibull::Weibull;

mod bernoulli;
mod beta;
mod binomial;
mod categorical;
mod cauchy;
mod chi;
mod chi_squared;
mod dirichlet;
mod discrete_uniform;
mod erlang;
mod exponential;
mod fisher_snedecor;
mod gamma;
mod geometric;
mod hypergeometric;
mod internal;
mod inverse_gamma;
mod log_normal;
mod multinomial;
mod normal;
mod pareto;
mod poisson;
mod students_t;
mod triangular;
mod uniform;
mod weibull;
mod ziggurat;
mod ziggurat_tables;

/// The `Distribution` trait is used to specify an interface
/// for sampling statistical distributions
pub trait Distribution<T> {
    /// Draws a random sample using the supplied random number generator
    /// # Examples
    ///
    /// A trivial implementation that just samples from the supplied
    /// random number generator
    ///
    /// ```
    /// # extern crate rand;
    /// # extern crate statrs;
    ///
    /// use rand::Rng;
    /// use statrs::distribution::Distribution;
    ///
    /// struct Foo;
    ///
    /// impl Distribution<f64> for Foo {
    ///     fn sample<R: Rng>(&self, r: &mut R) -> f64 {
    ///         r.next_f64()
    ///     }
    /// }
    ///
    /// # fn main() { }
    /// ```
    fn sample<R: Rng>(&self, r: &mut R) -> T;
}

/// The `Univariate` trait is used to specify an interface for univariate
/// distributions e.g. distributions that have a closed form cumulative
/// distribution
/// function
pub trait Univariate<T, K>: Distribution<K> + Min<T> + Max<T> {
    /// Returns the cumulative distribution function calculated
    /// at `x` for a given distribution. May panic depending
    /// on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{Univariate, Uniform};
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.5, n.cdf(0.5));
    /// ```
    fn cdf(&self, x: K) -> K;
}

/// The `InverseCDF` trait used to specify an interface for distributions
/// with a closed form solution to the inverse cumulative distribution function.
/// This trait will probably be merged into `Univariate` in a future release
/// when already implemented distributions have `InverseCDF` back ported
pub trait InverseCDF<T> {
    /// Returns the inverse cumulative distribution function
    /// calculated at `x` for a given distribution. May panic
    /// depending on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// ```
    fn inverse_cdf(&self, x: T) -> T;
}

/// The `Continuous` trait extends the `Distribution`
/// trait and provides an interface for interacting with continuous
/// statistical distributions
///
/// # Remarks
///
/// All methods provided by the `Continuous` trait are unchecked, meaning
/// they can panic if in an invalid state or encountering invalid input
/// depending on the implementing distribution.
pub trait Continuous<T, K> {
    /// Returns the probability density function calculated at `x` for a given
    /// distribution.
    /// May panic depending on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{Continuous, Uniform};
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(1.0, n.pdf(0.5));
    /// ```
    fn pdf(&self, x: T) -> K;

    /// Returns the log of the probability density function calculated at `x`
    /// for a given distribution.
    /// May panic depending on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{Continuous, Uniform};
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.0, n.ln_pdf(0.5));
    /// ```
    fn ln_pdf(&self, x: T) -> K;
}

/// The `Discrete` trait extends the `Distribution`
/// trait and provides an interface for interacting with discrete
/// statistical distributions
///
/// # Remarks
///
/// All methods provided by the `Discrete` trait are unchecked, meaning
/// they can panic if in an invalid state or encountering invalid input
/// depending on the implementing distribution.
pub trait Discrete<T, K> {
    /// Returns the probability mass function calculated at `x` for a given
    /// distribution.
    /// May panic depending on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{Discrete, Binomial};
    /// use statrs::prec;
    ///
    /// let n = Binomial::new(0.5, 10).unwrap();
    /// assert!(prec::almost_eq(n.pmf(5), 0.24609375, 1e-15));
    /// ```
    fn pmf(&self, x: T) -> K;

    /// Returns the log of the probability mass function calculated at `x` for
    /// a given distribution.
    /// May panic depending on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{Discrete, Binomial};
    /// use statrs::prec;
    ///
    /// let n = Binomial::new(0.5, 10).unwrap();
    /// assert!(prec::almost_eq(n.ln_pmf(5), (0.24609375f64).ln(), 1e-15));
    /// ```
    fn ln_pmf(&self, x: T) -> K;
}
