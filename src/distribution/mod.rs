//! Defines common interfaces for interacting with statistical distributions
//! and provides
//! concrete implementations for a variety of distributions.
use super::statistics::{Max, Min};
use ::num_traits::{Float, Num};
use num_traits::NumAssignOps;

pub use self::bernoulli::Bernoulli;
pub use self::beta::{Beta, BetaError};
pub use self::binomial::{Binomial, BinomialError};
pub use self::categorical::{Categorical, CategoricalError};
pub use self::cauchy::{Cauchy, CauchyError};
pub use self::chi::{Chi, ChiError};
pub use self::chi_squared::ChiSquared;
pub use self::dirac::{Dirac, DiracError};
#[cfg(feature = "nalgebra")]
pub use self::dirichlet::{Dirichlet, DirichletError};
pub use self::discrete_uniform::{DiscreteUniform, DiscreteUniformError};
pub use self::empirical::Empirical;
pub use self::erlang::Erlang;
pub use self::exponential::{Exp, ExpError};
pub use self::fisher_snedecor::{FisherSnedecor, FisherSnedecorError};
pub use self::gamma::{Gamma, GammaError};
pub use self::geometric::{Geometric, GeometricError};
pub use self::gumbel::{Gumbel, GumbelError};
pub use self::hypergeometric::{Hypergeometric, HypergeometricError};
pub use self::inverse_gamma::{InverseGamma, InverseGammaError};
pub use self::laplace::{Laplace, LaplaceError};
pub use self::levy::{Levy, LevyError};
pub use self::log_normal::{LogNormal, LogNormalError};
#[cfg(feature = "nalgebra")]
pub use self::multinomial::{Multinomial, MultinomialError};
#[cfg(feature = "nalgebra")]
pub use self::multivariate_normal::{MultivariateNormal, MultivariateNormalError};
#[cfg(feature = "nalgebra")]
pub use self::multivariate_students_t::{MultivariateStudent, MultivariateStudentError};
pub use self::negative_binomial::{NegativeBinomial, NegativeBinomialError};
pub use self::normal::{Normal, NormalError};
pub use self::pareto::{Pareto, ParetoError};
pub use self::poisson::{Poisson, PoissonError};
pub use self::students_t::{StudentsT, StudentsTError};
pub use self::triangular::{Triangular, TriangularError};
pub use self::uniform::{Uniform, UniformError};
pub use self::weibull::{Weibull, WeibullError};

mod bernoulli;
mod beta;
mod binomial;
mod categorical;
mod cauchy;
mod chi;
mod chi_squared;
mod dirac;
#[cfg(feature = "nalgebra")]
#[cfg_attr(docsrs, doc(cfg(feature = "nalgebra")))]
mod dirichlet;
mod discrete_uniform;
mod empirical;
mod erlang;
mod exponential;
mod fisher_snedecor;
mod gamma;
mod geometric;
mod gumbel;
mod hypergeometric;
#[macro_use]
mod internal;
mod inverse_gamma;
mod laplace;
mod levy;
mod log_normal;
#[cfg(feature = "nalgebra")]
#[cfg_attr(docsrs, doc(cfg(feature = "nalgebra")))]
mod multinomial;
#[cfg(feature = "nalgebra")]
#[cfg_attr(docsrs, doc(cfg(feature = "nalgebra")))]
mod multivariate_normal;
#[cfg(feature = "nalgebra")]
#[cfg_attr(docsrs, doc(cfg(feature = "nalgebra")))]
mod multivariate_students_t;
mod negative_binomial;
mod normal;
mod pareto;
mod poisson;
mod students_t;
mod triangular;
mod uniform;
mod weibull;
#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
mod ziggurat;
#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
mod ziggurat_tables;

/// Represents the errors that can occur when computing [`ContinuousCDF::try_inverse_cdf`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum InverseCdfError {
    /// The argument `p` is outside the closed interval `[0, 1]`.
    ArgumentOutOfRange,
}

impl core::fmt::Display for InverseCdfError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match self {
            InverseCdfError::ArgumentOutOfRange => write!(f, "argument is outside [0, 1]"),
        }
    }
}

impl core::error::Error for InverseCdfError {}

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
    fn sf(&self, x: K) -> T {
        T::one() - self.cdf(x)
    }

    /// Due to issues with rounding and floating-point accuracy the default
    /// implementation may be ill-behaved.
    /// Specialized inverse cdfs should be used whenever possible.
    /// Performs a binary search on the domain of `cdf` to obtain an approximation
    /// of `F^-1(p) := inf { x | F(x) >= p }`. Needless to say, performance may
    /// be lacking.
    #[doc(alias = "quantile function")]
    #[doc(alias = "quantile")]
    fn inverse_cdf(&self, p: T) -> K {
        if p == T::zero() {
            return self.min();
        };
        if p == T::one() {
            return self.max();
        };
        let two = K::one() + K::one();

        // Bracket the root, preferring the distribution's own domain bounds and
        // only doubling outward from ±2 when a bound is not finite.
        let mut low = self.min();
        if !low.is_finite() {
            low = -two;
            while self.cdf(low) > p {
                low = low + low;
            }
        }
        let mut high = self.max();
        if !high.is_finite() {
            high = two;
            while self.cdf(high) < p {
                high = high + high;
            }
        }

        // In the upper half invert the survival function instead of the cdf:
        // as `cdf` saturates to one it loses the resolution needed to place the
        // quantile, whereas `sf` keeps that tail well conditioned.
        let upper = p > T::one() / (T::one() + T::one());
        let target = if upper { T::one() - p } else { p };

        // Bisect until the bracket agrees to the crate's relative accuracy. A
        // fixed iteration count cannot approach this when the bracket is wide or
        // the quantile lies deep in a tail, where the search would otherwise
        // return the bracket midpoint rather than the true value.
        let accuracy = K::from(crate::prec::DEFAULT_RELATIVE_ACC).unwrap();
        for _ in 0..100 {
            let mid = low + (high - low) / two;
            if mid <= low || mid >= high {
                break;
            }
            let above = if upper {
                self.sf(mid) <= target
            } else {
                self.cdf(mid) >= target
            };
            if above {
                high = mid;
            } else {
                low = mid;
            }
            if (high - low).abs() <= accuracy * low.abs().max(high.abs()) {
                break;
            }
        }
        low + (high - low) / two
    }

    /// Due to issues with rounding and floating-point accuracy the default
    /// implementation may be ill-behaved.
    /// Specialized inverse cdfs should be used whenever possible.
    /// Performs a binary search on the domain of `cdf` to obtain an approximation
    /// of `F^-1(p) := inf { x | F(x) >= p }`. Needless to say, performance may
    /// may be lacking.
    #[doc(alias = "quantile function")]
    #[doc(alias = "quantile")]
    fn try_inverse_cdf(&self, p: T) -> Result<K, InverseCdfError> {
        Ok(self.inverse_cdf(p))
    }
}

/// The `DiscreteCDF` trait is used to specify an interface for univariate
/// discrete distributions.
pub trait DiscreteCDF<K: Sized + Num + Ord + Clone + NumAssignOps, T: Float>:
    Min<K> + Max<K>
{
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
    fn sf(&self, x: K) -> T {
        T::one() - self.cdf(x)
    }

    /// Due to issues with rounding and floating-point accuracy the default implementation may be ill-behaved
    /// Specialized inverse cdfs should be used whenever possible.
    ///
    /// # Panics
    /// this default impl panics if provided `p` not on interval [0.0, 1.0]
    fn inverse_cdf(&self, p: T) -> K {
        if p <= self.cdf(self.min()) {
            return self.min();
        } else if p == T::one() {
            return self.max();
        } else if !(T::zero()..=T::one()).contains(&p) {
            panic!("p must be on [0, 1]")
        }

        let two = K::one() + K::one();
        let mut ub = two.clone();
        let lb = self.min();
        while self.cdf(ub.clone()) < p {
            ub *= two.clone();
        }

        internal::integral_bisection_search(|p| self.cdf(p.clone()), p, lb, ub).unwrap()
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
    /// use approx::assert_abs_diff_eq;
    ///
    /// let n = Binomial::new(0.5, 10).unwrap();
    /// assert_abs_diff_eq!(n.pmf(5), 0.24609375, epsilon = 1e-15);
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
    /// use approx::assert_abs_diff_eq;
    ///
    /// let n = Binomial::new(0.5, 10).unwrap();
    /// assert_abs_diff_eq!(n.ln_pmf(5), (0.24609375f64).ln(), epsilon = 1e-15);
    /// ```
    fn ln_pmf(&self, x: K) -> T;
}

#[cfg(test)]
mod tests {
    use super::{ContinuousCDF, Max, Min};

    // Standard logistic distribution, used to exercise the ContinuousCDF default
    // inverse_cdf on an infinite (two-sided) support with a closed-form quantile.
    struct Logistic;

    impl Min<f64> for Logistic {
        fn min(&self) -> f64 {
            f64::NEG_INFINITY
        }
    }

    impl Max<f64> for Logistic {
        fn max(&self) -> f64 {
            f64::INFINITY
        }
    }

    impl ContinuousCDF<f64, f64> for Logistic {
        fn cdf(&self, x: f64) -> f64 {
            1.0 / (1.0 + (-x).exp())
        }
        fn sf(&self, x: f64) -> f64 {
            1.0 / (1.0 + x.exp())
        }
    }

    #[test]
    fn test_default_inverse_cdf_infinite_support() {
        let d = Logistic;
        // Doubling out from ±2 on both sides, then cdf (lower) and sf (upper) search.
        for &p in &[1e-10f64, 1e-4, 0.1, 0.3, 0.7, 0.9, 1.0 - 1e-4, 1.0 - 1e-10] {
            let expected = (p / (1.0 - p)).ln();
            let q = d.inverse_cdf(p);
            let relerr = ((q - expected) / expected).abs();
            assert!(
                relerr <= 1e-11,
                "logistic inverse_cdf({p}) = {q}, want {expected}"
            );
        }
    }
}
