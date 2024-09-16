use crate::distribution::{Discrete, DiscreteCDF};
use crate::function::{beta, gamma};
use crate::statistics::*;
use std::f64;

/// Implements the
/// [negative binomial](http://en.wikipedia.org/wiki/Negative_binomial_distribution)
/// distribution.
///
/// *Please note carefully the meaning of the parameters.*  As noted in the
/// wikipedia article, there are several different commonly used conventions
/// for the parameters of the negative binomial distribution.
///
/// The negative binomial distribution is a discrete distribution with two
/// parameters, `r` and `p`.  When `r` is an integer, the negative binomial
/// distribution can be interpreted as the distribution of the number of
/// failures in a sequence of Bernoulli trials that continue until `r`
/// successes occur.  `p` is the probability of success in a single Bernoulli
/// trial.
///
/// `NegativeBinomial` accepts non-integer values for `r`.  This is a
/// generalization of the more common case where `r` is an integer.
///
/// # Examples
///
/// ```
/// use statrs::distribution::{NegativeBinomial, Discrete};
/// use statrs::statistics::DiscreteDistribution;
/// use statrs::prec::almost_eq;
///
/// let r = NegativeBinomial::new(4.0, 0.5).unwrap();
/// assert_eq!(r.mean().unwrap(), 4.0);
/// assert!(almost_eq(r.pmf(0), 0.0625, 1e-8));
/// assert!(almost_eq(r.pmf(3), 0.15625, 1e-8));
/// ```
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct NegativeBinomial {
    r: f64,
    p: f64,
}

/// Represents the errors that can occur when creating a [`NegativeBinomial`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum NegativeBinomialError {
    /// `r` is NaN or less than zero.
    RInvalid,

    /// `p` is NaN or not in `[0, 1]`.
    PInvalid,
}

impl std::fmt::Display for NegativeBinomialError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            NegativeBinomialError::RInvalid => write!(f, "r is NaN or less than zero"),
            NegativeBinomialError::PInvalid => write!(f, "p is NaN or not in [0, 1]"),
        }
    }
}

impl std::error::Error for NegativeBinomialError {}

impl NegativeBinomial {
    /// Constructs a new negative binomial distribution with parameters `r`
    /// and `p`.  When `r` is an integer, the negative binomial distribution
    /// can be interpreted as the distribution of the number of failures in
    /// a sequence of Bernoulli trials that continue until `r` successes occur.
    /// `p` is the probability of success in a single Bernoulli trial.
    ///
    /// # Errors
    ///
    /// Returns an error if `p` is `NaN`, less than `0.0`,
    /// greater than `1.0`, or if `r` is `NaN` or less than `0`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::NegativeBinomial;
    ///
    /// let mut result = NegativeBinomial::new(4.0, 0.5);
    /// assert!(result.is_ok());
    ///
    /// result = NegativeBinomial::new(-0.5, 5.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(r: f64, p: f64) -> Result<NegativeBinomial, NegativeBinomialError> {
        if r.is_nan() || r < 0.0 {
            return Err(NegativeBinomialError::RInvalid);
        }

        if p.is_nan() || !(0.0..=1.0).contains(&p) {
            return Err(NegativeBinomialError::PInvalid);
        }

        Ok(NegativeBinomial { r, p })
    }

    /// Returns the probability of success `p` of a single
    /// Bernoulli trial associated with the negative binomial
    /// distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::NegativeBinomial;
    ///
    /// let r = NegativeBinomial::new(5.0, 0.5).unwrap();
    /// assert_eq!(r.p(), 0.5);
    /// ```
    pub fn p(&self) -> f64 {
        self.p
    }

    /// Returns the number `r` of success of this negative
    /// binomial distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::NegativeBinomial;
    ///
    /// let r = NegativeBinomial::new(5.0, 0.5).unwrap();
    /// assert_eq!(r.r(), 5.0);
    /// ```
    pub fn r(&self) -> f64 {
        self.r
    }
}

impl std::fmt::Display for NegativeBinomial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NB({},{})", self.r, self.p)
    }
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl ::rand::distributions::Distribution<u64> for NegativeBinomial {
    fn sample<R: ::rand::Rng + ?Sized>(&self, r: &mut R) -> u64 {
        use crate::distribution::{gamma, poisson};

        let lambda = gamma::sample_unchecked(r, self.r, (1.0 - self.p) / self.p);
        poisson::sample_unchecked(r, lambda).floor() as u64
    }
}

impl DiscreteCDF<u64, f64> for NegativeBinomial {
    /// Calculates the cumulative distribution function for the
    /// negative binomial distribution at `x`.
    ///
    /// # Formula
    ///
    /// ```text
    /// I_(p)(r, x+1)
    /// ```
    ///
    /// where `I_(x)(a, b)` is the regularized incomplete beta function.
    fn cdf(&self, x: u64) -> f64 {
        beta::beta_reg(self.r, x as f64 + 1.0, self.p)
    }

    /// Calculates the survival function for the
    /// negative binomial distribution at `x`
    ///
    /// Note that due to extending the distribution to the reals
    /// (allowing positive real values for `r`), while still technically
    /// a discrete distribution the CDF behaves more like that of a
    /// continuous distribution rather than a discrete distribution
    /// (i.e. a smooth graph rather than a step-ladder)
    ///
    /// # Formula
    ///
    /// ```text
    /// I_(1-p)(x+1, r)
    /// ```
    ///
    /// where `I_(x)(a, b)` is the regularized incomplete beta function
    fn sf(&self, x: u64) -> f64 {
        beta::beta_reg(x as f64 + 1.0, self.r, 1. - self.p)
    }
}

impl Min<u64> for NegativeBinomial {
    /// Returns the minimum value in the domain of the
    /// negative binomial distribution representable by a 64-bit
    /// integer.
    ///
    /// # Formula
    ///
    /// ```text
    /// 0
    /// ```
    fn min(&self) -> u64 {
        0
    }
}

impl Max<u64> for NegativeBinomial {
    /// Returns the maximum value in the domain of the
    /// negative binomial distribution representable by a 64-bit
    /// integer.
    ///
    /// # Formula
    ///
    /// ```text
    /// u64::MAX
    /// ```
    fn max(&self) -> u64 {
        u64::MAX
    }
}

impl DiscreteDistribution<f64> for NegativeBinomial {
    /// Returns the mean of the negative binomial distribution.
    ///
    /// # Formula
    ///
    /// ```text
    /// r * (1-p) / p
    /// ```
    fn mean(&self) -> Option<f64> {
        Some(self.r * (1.0 - self.p) / self.p)
    }

    /// Returns the variance of the negative binomial distribution.
    ///
    /// # Formula
    ///
    /// ```text
    /// r * (1-p) / p^2
    /// ```
    fn variance(&self) -> Option<f64> {
        Some(self.r * (1.0 - self.p) / (self.p * self.p))
    }

    /// Returns the skewness of the negative binomial distribution.
    ///
    /// # Formula
    ///
    /// ```text
    /// (2-p) / sqrt(r * (1-p))
    /// ```
    fn skewness(&self) -> Option<f64> {
        Some((2.0 - self.p) / f64::sqrt(self.r * (1.0 - self.p)))
    }
}

impl Mode<Option<f64>> for NegativeBinomial {
    /// Returns the mode for the negative binomial distribution.
    ///
    /// # Formula
    ///
    /// ```text
    /// if r > 1 then
    ///     floor((r - 1) * (1-p / p))
    /// else
    ///     0
    /// ```
    fn mode(&self) -> Option<f64> {
        let mode = if self.r > 1.0 {
            f64::floor((self.r - 1.0) * (1.0 - self.p) / self.p)
        } else {
            0.0
        };
        Some(mode)
    }
}

impl Discrete<u64, f64> for NegativeBinomial {
    /// Calculates the probability mass function for the negative binomial
    /// distribution at `x`.
    ///
    /// # Formula
    ///
    /// When `r` is an integer, the formula is:
    ///
    /// ```text
    /// (x + r - 1 choose x) * (1 - p)^x * p^r
    /// ```
    ///
    /// The general formula for real `r` is:
    ///
    /// ```text
    /// Γ(r + x)/(Γ(r) * Γ(x + 1)) * (1 - p)^x * p^r
    /// ```
    ///
    /// where Γ(x) is the Gamma function.
    fn pmf(&self, x: u64) -> f64 {
        self.ln_pmf(x).exp()
    }

    /// Calculates the log probability mass function for the negative binomial
    /// distribution at `x`.
    ///
    /// # Formula
    ///
    /// When `r` is an integer, the formula is:
    ///
    /// ```text
    /// ln((x + r - 1 choose x) * (1 - p)^x * p^r)
    /// ```
    ///
    /// The general formula for real `r` is:
    ///
    /// ```text
    /// ln(Γ(r + x)/(Γ(r) * Γ(x + 1)) * (1 - p)^x * p^r)
    /// ```
    ///
    /// where Γ(x) is the Gamma function.
    fn ln_pmf(&self, x: u64) -> f64 {
        let k = x as f64;
        gamma::ln_gamma(self.r + k) - gamma::ln_gamma(self.r) - gamma::ln_gamma(k + 1.0)
            + (self.r * self.p.ln())
            + (k * (-self.p).ln_1p())
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::internal::test;
    use crate::testing_boiler;

    testing_boiler!(r: f64, p: f64; NegativeBinomial; NegativeBinomialError);

    #[test]
    fn test_create() {
        create_ok(0.0, 0.0);
        create_ok(0.3, 0.4);
        create_ok(1.0, 0.3);
    }

    #[test]
    fn test_bad_create() {
        test_create_err(f64::NAN, 1.0, NegativeBinomialError::RInvalid);
        test_create_err(0.0, f64::NAN, NegativeBinomialError::PInvalid);
        create_err(-1.0, 1.0);
        create_err(2.0, 2.0);
    }

    #[test]
    fn test_mean() {
        let mean = |x: NegativeBinomial| x.mean().unwrap();
        test_exact(4.0, 0.0, f64::INFINITY, mean);
        test_absolute(3.0, 0.3, 7.0, 1e-15 , mean);
        test_exact(2.0, 1.0, 0.0, mean);
    }

    #[test]
    fn test_variance() {
        let variance = |x: NegativeBinomial| x.variance().unwrap();
        test_exact(4.0, 0.0, f64::INFINITY, variance);
        test_absolute(3.0, 0.3, 23.333333333333, 1e-12, variance);
        test_exact(2.0, 1.0, 0.0, variance);
    }

    #[test]
    fn test_skewness() {
        let skewness = |x: NegativeBinomial| x.skewness().unwrap();
        test_exact(0.0, 0.0, f64::INFINITY, skewness);
        test_absolute(0.1, 0.3, 6.425396041, 1e-09, skewness);
        test_exact(1.0, 1.0, f64::INFINITY, skewness);
    }

    #[test]
    fn test_mode() {
        let mode = |x: NegativeBinomial| x.mode().unwrap();
        test_exact(0.0, 0.0, 0.0, mode);
        test_exact(0.3, 0.0, 0.0, mode);
        test_exact(1.0, 1.0, 0.0, mode);
        test_exact(10.0, 0.01, 891.0, mode);
    }

    #[test]
    fn test_min_max() {
        let min = |x: NegativeBinomial| x.min();
        let max = |x: NegativeBinomial| x.max();
        test_exact(1.0, 0.5, 0, min);
        test_exact(1.0, 0.3, u64::MAX, max);
    }

    #[test]
    fn test_pmf() {
        let pmf = |arg: u64| move |x: NegativeBinomial| x.pmf(arg);
        test_absolute(4.0, 0.5, 0.0625, 1e-8, pmf(0));
        test_absolute(4.0, 0.5, 0.15625, 1e-8, pmf(3));
        test_exact(1.0, 0.0, 0.0, pmf(0));
        test_exact(1.0, 0.0, 0.0, pmf(1));
        test_absolute(3.0, 0.2, 0.008, 1e-15, pmf(0));
        test_absolute(3.0, 0.2, 0.0192, 1e-15, pmf(1));
        test_absolute(3.0, 0.2, 0.04096, 1e-15, pmf(3));
        test_absolute(10.0, 0.2, 1.024e-07, 1e-07, pmf(0));
        test_absolute(10.0, 0.2, 8.192e-07, 1e-07, pmf(1));
        test_absolute(10.0, 0.2, 0.001015706852, 1e-07, pmf(10));
        test_absolute(1.0, 0.3, 0.3, 1e-15,  pmf(0));
        test_absolute(1.0, 0.3, 0.21, 1e-15, pmf(1));
        test_absolute(3.0, 0.3, 0.027, 1e-15, pmf(0));
        test_exact(0.3, 1.0, 0.0, pmf(1));
        test_exact(0.3, 1.0, 0.0, pmf(3));
        test_is_nan(0.3, 1.0, pmf(0));
        test_exact(0.3, 1.0, 0.0, pmf(1));
        test_exact(0.3, 1.0, 0.0, pmf(10));
        test_is_nan(1.0, 1.0, pmf(0));
        test_exact(1.0, 1.0, 0.0, pmf(1));
        test_is_nan(3.0, 1.0, pmf(0));
        test_exact(3.0, 1.0, 0.0, pmf(1));
        test_exact(3.0, 1.0, 0.0, pmf(3));
        test_is_nan(10.0, 1.0, pmf(0));
        test_exact(10.0, 1.0, 0.0, pmf(1));
        test_exact(10.0, 1.0, 0.0, pmf(10));
    }

    #[test]
    fn test_ln_pmf() {
        let ln_pmf = |arg: u64| move |x: NegativeBinomial| x.ln_pmf(arg);
        test_exact(1.0, 0.0, f64::NEG_INFINITY, ln_pmf(0));
        test_exact(1.0, 0.0, f64::NEG_INFINITY, ln_pmf(1));
        test_absolute(3.0, 0.2, -4.828313737, 1e-08, ln_pmf(0));
        test_absolute(3.0, 0.2, -3.952845, 1e-08, ln_pmf(1));
        test_absolute(3.0, 0.2, -3.195159298, 1e-08, ln_pmf(3));
        test_absolute(10.0, 0.2, -16.09437912, 1e-08, ln_pmf(0));
        test_absolute(10.0, 0.2, -14.01493758, 1e-08, ln_pmf(1));
        test_absolute(10.0, 0.2, -6.892170503, 1e-08, ln_pmf(10));
        test_absolute(1.0, 0.3, -1.203972804, 1e-08,  ln_pmf(0));
        test_absolute(1.0, 0.3, -1.560647748, 1e-08, ln_pmf(1));
        test_absolute(3.0, 0.3, -3.611918413, 1e-08, ln_pmf(0));
        test_exact(0.3, 1.0, f64::NEG_INFINITY, ln_pmf(1));
        test_exact(0.3, 1.0, f64::NEG_INFINITY, ln_pmf(3));
        test_is_nan(0.3, 1.0, ln_pmf(0));
        test_exact(0.3, 1.0, f64::NEG_INFINITY, ln_pmf(1));
        test_exact(0.3, 1.0, f64::NEG_INFINITY, ln_pmf(10));
        test_is_nan(1.0, 1.0, ln_pmf(0));
        test_exact(1.0, 1.0, f64::NEG_INFINITY, ln_pmf(1));
        test_is_nan(3.0, 1.0, ln_pmf(0));
        test_exact(3.0, 1.0, f64::NEG_INFINITY, ln_pmf(1));
        test_exact(3.0, 1.0, f64::NEG_INFINITY, ln_pmf(3));
        test_is_nan(10.0, 1.0, ln_pmf(0));
        test_exact(10.0, 1.0, f64::NEG_INFINITY, ln_pmf(1));
        test_exact(10.0, 1.0, f64::NEG_INFINITY, ln_pmf(10));
    }

    #[test]
    fn test_cdf() {
        let cdf = |arg: u64| move |x: NegativeBinomial| x.cdf(arg);
        test_absolute(1.0, 0.3, 0.3, 1e-08, cdf(0));
        test_absolute(1.0, 0.3, 0.51, 1e-08, cdf(1));
        test_absolute(1.0, 0.3, 0.83193, 1e-08, cdf(4));
        test_absolute(1.0, 0.3, 0.9802267326, 1e-08, cdf(10));
        test_exact(1.0, 1.0, 1.0, cdf(0));
        test_exact(1.0, 1.0, 1.0, cdf(1));
        test_absolute(10.0, 0.75, 0.05631351471, 1e-08, cdf(0));
        test_absolute(10.0, 0.75, 0.1970973015, 1e-08, cdf(1));
        test_absolute(10.0, 0.75, 0.9960578583, 1e-08, cdf(10));
    }

    #[test]
    fn test_sf() {
        let sf = |arg: u64| move |x: NegativeBinomial| x.sf(arg);
        test_absolute(1.0, 0.3, 0.7, 1e-08, sf(0));
        test_absolute(1.0, 0.3, 0.49, 1e-08, sf(1));
        test_absolute(1.0, 0.3, 0.1680699999999986, 1e-08, sf(4));
        test_absolute(1.0, 0.3, 0.019773267430000074, 1e-08, sf(10));
        test_exact(1.0, 1.0, 0.0, sf(0));
        test_exact(1.0, 1.0, 0.0, sf(1));
        test_absolute(10.0, 0.75, 0.9436864852905275, 1e-08, sf(0));
        test_absolute(10.0, 0.75, 0.8029026985168456, 1e-08, sf(1));
        test_absolute(10.0, 0.75, 0.003942141664083465, 1e-08, sf(10));
    }

    #[test]
    fn test_cdf_upper_bound() {
        let cdf = |arg: u64| move |x: NegativeBinomial| x.cdf(arg);
        test_exact(3.0, 0.5, 1.0, cdf(100));
    }

    #[test]
    fn test_discrete() {
        test::check_discrete_distribution(&create_ok(5.0, 0.3), 35);
        test::check_discrete_distribution(&create_ok(10.0, 0.7), 21);
    }
    
    #[test]
    fn test_sf_upper_bound() {
        let sf = |arg: u64| move |x: NegativeBinomial| x.sf(arg);
        test_absolute(3.0, 0.5, 5.282409836586059e-28, 1e-28, sf(100));
    }
}
