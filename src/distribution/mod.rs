//! Defines common interfaces for interacting with statistical distributions
//! and provides
//! concrete implementations for a variety of distributions.
use super::statistics::{Max, Min};
use ::num_traits::{float::Float, Bounded, Num};

pub use self::bernoulli::Bernoulli;
pub use self::beta::Beta;
pub use self::binomial::Binomial;
pub use self::categorical::Categorical;
pub use self::cauchy::Cauchy;
pub use self::chi::Chi;
pub use self::chi_squared::ChiSquared;
pub use self::dirac::Dirac;
pub use self::dirichlet::Dirichlet;
pub use self::discrete_uniform::DiscreteUniform;
pub use self::empirical::Empirical;
pub use self::erlang::Erlang;
pub use self::exponential::Exp;
pub use self::fisher_snedecor::FisherSnedecor;
pub use self::gamma::Gamma;
pub use self::geometric::Geometric;
pub use self::hypergeometric::Hypergeometric;
pub use self::inverse_gamma::InverseGamma;
pub use self::laplace::Laplace;
pub use self::log_normal::LogNormal;
pub use self::multinomial::Multinomial;
pub use self::multivariate_normal::MultivariateNormal;
pub use self::negative_binomial::NegativeBinomial;
pub use self::normal::Normal;
pub use self::pareto::Pareto;
pub use self::poisson::Poisson;
pub use self::students_t::StudentsT;
pub use self::triangular::Triangular;
pub use self::uniform::Uniform;
pub use self::weibull::Weibull;
pub use self::wishart::Wishart;
pub use self::inverse_wishart::InverseWishart;

mod bernoulli;
mod beta;
mod binomial;
mod categorical;
mod cauchy;
mod chi;
mod chi_squared;
mod dirac;
mod dirichlet;
mod discrete_uniform;
mod empirical;
mod erlang;
mod exponential;
mod fisher_snedecor;
mod gamma;
mod geometric;
mod hypergeometric;
#[macro_use]
mod internal;
mod inverse_gamma;
mod laplace;
mod log_normal;
mod multinomial;
mod multivariate_normal;
mod negative_binomial;
mod normal;
mod pareto;
mod poisson;
mod students_t;
mod triangular;
mod uniform;
mod weibull;
mod ziggurat;
mod ziggurat_tables;
mod wishart;
mod inverse_wishart;

use crate::Result;

/// The `ContinuousCDF` trait is used to specify an interface for univariate
/// distributions for which cdf float arguments are sensible.
pub trait ContinuousCDF<K: Float, T: Float>: Min<K> + Max<K> {
    /// Returns the cumulative distribution function calculated
    /// at `x` for a given distribution. May panic depending
    /// on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{ContinuousCDF, Uniform};
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.5, n.cdf(0.5));
    /// ```
    fn cdf(&self, x: K) -> T;

    /// Returns the survival function calculated
    /// at `x` for a given distribution. May panic depending
    /// on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{ContinuousCDF, Uniform};
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.5, n.sf(0.5));
    /// ```
    fn sf(&self, x: K) -> T;

    /// Due to issues with rounding and floating-point accuracy the default
    /// implementation may be ill-behaved.
    /// Specialized inverse cdfs should be used whenever possible.
    /// Performs a binary search on the domain of `cdf` to obtain an approximation
    /// of `F^-1(p) := inf { x | F(x) >= p }`. Needless to say, performance may
    /// may be lacking.
    fn inverse_cdf(&self, p: T) -> K {
        if p == T::zero() {
            return self.min();
        };
        if p == T::one() {
            return self.max();
        };
        let two = K::one() + K::one();
        let mut high = two;
        let mut low = -high;
        while self.cdf(low) > p {
            low = low + low;
        }
        while self.cdf(high) < p {
            high = high + high;
        }
        let mut i = 16;
        while i != 0 {
            let mid = (high + low) / two;
            if self.cdf(mid) >= p {
                high = mid;
            } else {
                low = mid;
            }
            i -= 1;
        }
        (high + low) / two
    }
}

/// The `DiscreteCDF` trait is used to specify an interface for univariate
/// discrete distributions.
pub trait DiscreteCDF<K: Bounded + Clone + Num, T: Float>: Min<K> + Max<K> {
    /// Returns the cumulative distribution function calculated
    /// at `x` for a given distribution. May panic depending
    /// on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{DiscreteCDF, DiscreteUniform};
    ///
    /// let n = DiscreteUniform::new(1, 10).unwrap();
    /// assert_eq!(0.6, n.cdf(6));
    /// ```
    fn cdf(&self, x: K) -> T;

    /// Returns the survival function calculated at `x` for
    /// a given distribution. May panic depending on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{DiscreteCDF, DiscreteUniform};
    ///
    /// let n = DiscreteUniform::new(1, 10).unwrap();
    /// assert_eq!(0.4, n.sf(6));
    /// ```
    fn sf(&self, x: K) -> T;

    /// Due to issues with rounding and floating-point accuracy the default implementation may be ill-behaved
    /// Specialized inverse cdfs should be used whenever possible.
    fn inverse_cdf(&self, p: T) -> K {
        // TODO: fix integer implementation
        if p == T::zero() {
            return self.min();
        };
        if p == T::one() {
            return self.max();
        };
        let two = K::one() + K::one();
        let mut high = two.clone();
        let mut low = K::min_value();
        while self.cdf(high.clone()) < p {
            high = high.clone() + high.clone();
        }
        while high != low {
            let mid = (high.clone() + low.clone()) / two.clone();
            if self.cdf(mid.clone()) >= p {
                high = mid;
            } else {
                low = mid;
            }
        }
        high
    }
}

/// The `Continuous` trait  provides an interface for interacting with
/// continuous statistical distributions
///
/// # Remarks
///
/// All methods provided by the `Continuous` trait are unchecked, meaning
/// they can panic if in an invalid state or encountering invalid input
/// depending on the implementing distribution.
pub trait Continuous<K, T> {
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
    fn pdf(&self, x: K) -> T;

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
    fn ln_pdf(&self, x: K) -> T;
}

/// The `Discrete` trait provides an interface for interacting with discrete
/// statistical distributions
///
/// # Remarks
///
/// All methods provided by the `Discrete` trait are unchecked, meaning
/// they can panic if in an invalid state or encountering invalid input
/// depending on the implementing distribution.
pub trait Discrete<K, T> {
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
    fn pmf(&self, x: K) -> T;

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
    fn ln_pmf(&self, x: K) -> T;
}
