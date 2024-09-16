use crate::distribution::{Continuous, ContinuousCDF};
use crate::statistics::*;
use std::f64;

/// Implements the
/// [Exp](https://en.wikipedia.org/wiki/Exp_distribution)
/// distribution and is a special case of the
/// [Gamma](https://en.wikipedia.org/wiki/Gamma_distribution) distribution
/// (referenced [here](./struct.Gamma.html))
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Exp, Continuous};
/// use statrs::statistics::Distribution;
///
/// let n = Exp::new(1.0).unwrap();
/// assert_eq!(n.mean().unwrap(), 1.0);
/// assert_eq!(n.pdf(1.0), 0.3678794411714423215955);
/// ```
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Exp {
    rate: f64,
}

/// Represents the errors that can occur when creating a [`Exp`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum ExpError {
    /// The rate is NaN, zero or less than zero.
    RateInvalid,
}

impl std::fmt::Display for ExpError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ExpError::RateInvalid => write!(f, "Rate is NaN, zero or less than zero"),
        }
    }
}

impl std::error::Error for ExpError {}

impl Exp {
    /// Constructs a new exponential distribution with a
    /// rate (λ) of `rate`.
    ///
    /// # Errors
    ///
    /// Returns an error if rate is `NaN` or `rate <= 0.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Exp;
    ///
    /// let mut result = Exp::new(1.0);
    /// assert!(result.is_ok());
    ///
    /// result = Exp::new(-1.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(rate: f64) -> Result<Exp, ExpError> {
        if rate.is_nan() || rate <= 0.0 {
            Err(ExpError::RateInvalid)
        } else {
            Ok(Exp { rate })
        }
    }

    /// Returns the rate of the exponential distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Exp;
    ///
    /// let n = Exp::new(1.0).unwrap();
    /// assert_eq!(n.rate(), 1.0);
    /// ```
    pub fn rate(&self) -> f64 {
        self.rate
    }
}

impl std::fmt::Display for Exp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Exp({})", self.rate)
    }
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl ::rand::distributions::Distribution<f64> for Exp {
    fn sample<R: ::rand::Rng + ?Sized>(&self, r: &mut R) -> f64 {
        use crate::distribution::ziggurat;

        ziggurat::sample_exp_1(r) / self.rate
    }
}

impl ContinuousCDF<f64, f64> for Exp {
    /// Calculates the cumulative distribution function for the
    /// exponential distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// 1 - e^(-λ * x)
    /// ```
    ///
    /// where `λ` is the rate
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            1.0 - (-self.rate * x).exp()
        }
    }

    /// Calculates the cumulative distribution function for the
    /// exponential distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// e^(-λ * x)
    /// ```
    ///
    /// where `λ` is the rate
    fn sf(&self, x: f64) -> f64 {
        if x < 0.0 {
            1.0
        } else {
            (-self.rate * x).exp()
        }
    }

    /// Calculates the inverse cumulative distribution function.
    ///
    /// # Formula
    ///
    /// ```text
    /// -ln(1 - p) / λ
    /// ```
    ///
    /// where `p` is the probability and `λ` is the rate
    fn inverse_cdf(&self, p: f64) -> f64 {
        -(-p).ln_1p() / self.rate
    }
}

impl Min<f64> for Exp {
    /// Returns the minimum value in the domain of the exponential
    /// distribution representable by a double precision float
    ///
    /// # Formula
    ///
    /// ```text
    /// 0
    /// ```
    fn min(&self) -> f64 {
        0.0
    }
}

impl Max<f64> for Exp {
    /// Returns the maximum value in the domain of the exponential
    /// distribution representable by a double precision float
    ///
    /// # Formula
    ///
    /// ```text
    /// f64::INFINITY
    /// ```
    fn max(&self) -> f64 {
        f64::INFINITY
    }
}

impl Distribution<f64> for Exp {
    /// Returns the mean of the exponential distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// 1 / λ
    /// ```
    ///
    /// where `λ` is the rate
    fn mean(&self) -> Option<f64> {
        Some(1.0 / self.rate)
    }

    /// Returns the variance of the exponential distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// 1 / λ^2
    /// ```
    ///
    /// where `λ` is the rate
    fn variance(&self) -> Option<f64> {
        Some(1.0 / (self.rate * self.rate))
    }

    /// Returns the entropy of the exponential distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// 1 - ln(λ)
    /// ```
    ///
    /// where `λ` is the rate
    fn entropy(&self) -> Option<f64> {
        Some(1.0 - self.rate.ln())
    }

    /// Returns the skewness of the exponential distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// 2
    /// ```
    fn skewness(&self) -> Option<f64> {
        Some(2.0)
    }
}

impl Median<f64> for Exp {
    /// Returns the median of the exponential distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// (1 / λ) * ln2
    /// ```
    ///
    /// where `λ` is the rate
    fn median(&self) -> f64 {
        f64::consts::LN_2 / self.rate
    }
}

impl Mode<Option<f64>> for Exp {
    /// Returns the mode of the exponential distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// 0
    /// ```
    fn mode(&self) -> Option<f64> {
        Some(0.0)
    }
}

impl Continuous<f64, f64> for Exp {
    /// Calculates the probability density function for the exponential
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// λ * e^(-λ * x)
    /// ```
    ///
    /// where `λ` is the rate
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            self.rate * (-self.rate * x).exp()
        }
    }

    /// Calculates the log probability density function for the exponential
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// ln(λ * e^(-λ * x))
    /// ```
    ///
    /// where `λ` is the rate
    fn ln_pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            f64::NEG_INFINITY
        } else {
            self.rate.ln() - self.rate * x
        }
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::internal::*;
    use crate::testing_boiler;

    testing_boiler!(rate: f64; Exp; ExpError);

    #[test]
    fn test_create() {
        create_ok(0.1);
        create_ok(1.0);
        create_ok(10.0);
    }

    #[test]
    fn test_bad_create() {
        create_err(f64::NAN);
        create_err(0.0);
        create_err(-1.0);
        create_err(-10.0);
    }

    #[test]
    fn test_mean() {
        let mean = |x: Exp| x.mean().unwrap();
        test_exact(0.1, 10.0, mean);
        test_exact(1.0, 1.0, mean);
        test_exact(10.0, 0.1, mean);
    }

    #[test]
    fn test_variance() {
        let variance = |x: Exp| x.variance().unwrap();
        test_absolute(0.1, 100.0, 1e-13, variance);
        test_exact(1.0, 1.0, variance);
        test_exact(10.0, 0.01, variance);
    }

    #[test]
    fn test_entropy() {
        let entropy = |x: Exp| x.entropy().unwrap();
        test_absolute(0.1, 3.302585092994045684018, 1e-15, entropy);
        test_exact(1.0, 1.0, entropy);
        test_absolute(10.0, -1.302585092994045684018, 1e-15, entropy);
    }

    #[test]
    fn test_skewness() {
        let skewness = |x: Exp| x.skewness().unwrap();
        test_exact(0.1, 2.0, skewness);
        test_exact(1.0, 2.0, skewness);
        test_exact(10.0, 2.0, skewness);
    }

    #[test]
    fn test_median() {
        let median = |x: Exp| x.median();
        test_absolute(0.1, 6.931471805599453094172, 1e-15, median);
        test_exact(1.0, f64::consts::LN_2, median);
        test_exact(10.0, 0.06931471805599453094172, median);
    }

    #[test]
    fn test_mode() {
        let mode = |x: Exp| x.mode().unwrap();
        test_exact(0.1, 0.0, mode);
        test_exact(1.0, 0.0, mode);
        test_exact(10.0, 0.0, mode);
    }

    #[test]
    fn test_min_max() {
        let min = |x: Exp| x.min();
        let max = |x: Exp| x.max();
        test_exact(0.1, 0.0, min);
        test_exact(1.0, 0.0, min);
        test_exact(10.0, 0.0, min);
        test_exact(0.1, f64::INFINITY, max);
        test_exact(1.0, f64::INFINITY, max);
        test_exact(10.0, f64::INFINITY, max);
    }

    #[test]
    fn test_pdf() {
        let pdf = |arg: f64| move |x: Exp| x.pdf(arg);
        test_exact(0.1, 0.1, pdf(0.0));
        test_exact(1.0, 1.0, pdf(0.0));
        test_exact(10.0, 10.0, pdf(0.0));
        test_is_nan(f64::INFINITY, pdf(0.0));
        test_exact(0.1, 0.09900498337491680535739, pdf(0.1));
        test_absolute(1.0, 0.9048374180359595731642, 1e-15, pdf(0.1));
        test_exact(10.0, 3.678794411714423215955, pdf(0.1));
        test_is_nan(f64::INFINITY, pdf(0.1));
        test_exact(0.1, 0.09048374180359595731642, pdf(1.0));
        test_exact(1.0, 0.3678794411714423215955, pdf(1.0));
        test_absolute(10.0, 4.539992976248485153559e-4, 1e-19, pdf(1.0));
        test_is_nan(f64::INFINITY, pdf(1.0));
        test_exact(0.1, 0.0, pdf(f64::INFINITY));
        test_exact(1.0, 0.0, pdf(f64::INFINITY));
        test_exact(10.0, 0.0, pdf(f64::INFINITY));
        test_is_nan(f64::INFINITY, pdf(f64::INFINITY));
    }

    #[test]
    fn test_neg_pdf() {
        let pdf = |arg: f64| move |x: Exp| x.pdf(arg);
        test_exact(0.1, 0.0, pdf(-1.0));
    }

    #[test]
    fn test_ln_pdf() {
        let ln_pdf = |arg: f64| move |x: Exp| x.ln_pdf(arg);
        test_absolute(0.1, -2.302585092994045684018, 1e-15, ln_pdf(0.0));
        test_exact(1.0, 0.0, ln_pdf(0.0));
        test_exact(10.0, 2.302585092994045684018, ln_pdf(0.0));
        test_is_nan(f64::INFINITY, ln_pdf(0.0));
        test_absolute(0.1, -2.312585092994045684018, 1e-15, ln_pdf(0.1));
        test_exact(1.0, -0.1, ln_pdf(0.1));
        test_absolute(10.0, 1.302585092994045684018, 1e-15, ln_pdf(0.1));
        test_is_nan(f64::INFINITY, ln_pdf(0.1));
        test_exact(0.1, -2.402585092994045684018, ln_pdf(1.0));
        test_exact(1.0, -1.0, ln_pdf(1.0));
        test_exact(10.0, -7.697414907005954315982, ln_pdf(1.0));
        test_is_nan(f64::INFINITY, ln_pdf(1.0));
        test_exact(0.1, f64::NEG_INFINITY, ln_pdf(f64::INFINITY));
        test_exact(1.0, f64::NEG_INFINITY, ln_pdf(f64::INFINITY));
        test_exact(10.0, f64::NEG_INFINITY, ln_pdf(f64::INFINITY));
        test_is_nan(f64::INFINITY, ln_pdf(f64::INFINITY));
    }

    #[test]
    fn test_neg_ln_pdf() {
        let ln_pdf = |arg: f64| move |x: Exp| x.ln_pdf(arg);
        test_exact(0.1, f64::NEG_INFINITY, ln_pdf(-1.0));
    }

    #[test]
    fn test_cdf() {
        let cdf = |arg: f64| move |x: Exp| x.cdf(arg);
        test_exact(0.1, 0.0, cdf(0.0));
        test_exact(1.0, 0.0, cdf(0.0));
        test_exact(10.0, 0.0, cdf(0.0));
        test_is_nan(f64::INFINITY, cdf(0.0));
        test_absolute(0.1, 0.009950166250831946426094, 1e-16, cdf(0.1));
        test_absolute(1.0, 0.0951625819640404268358, 1e-16, cdf(0.1));
        test_exact(10.0, 0.6321205588285576784045, cdf(0.1));
        test_exact(f64::INFINITY, 1.0, cdf(0.1));
        test_absolute(0.1, 0.0951625819640404268358, 1e-16, cdf(1.0));
        test_exact(1.0, 0.6321205588285576784045, cdf(1.0));
        test_exact(10.0, 0.9999546000702375151485, cdf(1.0));
        test_exact(f64::INFINITY, 1.0, cdf(1.0));
        test_exact(0.1, 1.0, cdf(f64::INFINITY));
        test_exact(1.0, 1.0, cdf(f64::INFINITY));
        test_exact(10.0, 1.0, cdf(f64::INFINITY));
        test_exact(f64::INFINITY, 1.0, cdf(f64::INFINITY));
    }

    #[test]
    fn test_inverse_cdf() {
        let distribution = Exp::new(0.42).unwrap();
        assert_eq!(distribution.median(), distribution.inverse_cdf(0.5));

        let distribution = Exp::new(0.042).unwrap();
        assert_eq!(distribution.median(), distribution.inverse_cdf(0.5));

        let distribution = Exp::new(0.0042).unwrap();
        assert_eq!(distribution.median(), distribution.inverse_cdf(0.5));

        let distribution = Exp::new(0.33).unwrap();
        assert_eq!(distribution.median(), distribution.inverse_cdf(0.5));

        let distribution = Exp::new(0.033).unwrap();
        assert_eq!(distribution.median(), distribution.inverse_cdf(0.5));

        let distribution = Exp::new(0.0033).unwrap();
        assert_eq!(distribution.median(), distribution.inverse_cdf(0.5));
    }

    #[test]
    fn test_sf() {
        let sf = |arg: f64| move |x: Exp| x.sf(arg);
        test_exact(0.1, 1.0, sf(0.0));
        test_exact(1.0, 1.0, sf(0.0));
        test_exact(10.0, 1.0, sf(0.0));
        test_is_nan(f64::INFINITY, sf(0.0));
        test_absolute(0.1, 0.9900498337491681, 1e-16, sf(0.1));
        test_absolute(1.0, 0.9048374180359595, 1e-16, sf(0.1));
        test_absolute(10.0, 0.36787944117144233, 1e-15, sf(0.1));
        test_exact(f64::INFINITY, 0.0, sf(0.1));
    }

    #[test]
    fn test_neg_cdf() {
        let cdf = |arg: f64| move |x: Exp| x.cdf(arg);
        test_exact(0.1, 0.0, cdf(-1.0));
    }

    #[test]
    fn test_neg_sf() {
        let sf = |arg: f64| move |x: Exp| x.sf(arg);
        test_exact(0.1, 1.0, sf(-1.0));
    }

    #[test]
    fn test_continuous() {
        test::check_continuous_distribution(&create_ok(0.5), 0.0, 10.0);
        test::check_continuous_distribution(&create_ok(1.5), 0.0, 20.0);
        test::check_continuous_distribution(&create_ok(2.5), 0.0, 50.0);
    }
}
