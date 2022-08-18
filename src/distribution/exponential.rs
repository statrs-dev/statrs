use crate::distribution::{ziggurat, Continuous, ContinuousCDF};
use crate::statistics::*;
use crate::{Result, StatsError};
use rand::Rng;
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
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Exp {
    rate: f64,
}

impl Exp {
    /// Constructs a new exponential distribution with a
    /// rate (λ) of `rate`.
    ///
    /// # Errors
    ///
    /// Returns an error if rate is `NaN` or `rate <= 0.0`
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
    pub fn new(rate: f64) -> Result<Exp> {
        if rate.is_nan() || rate <= 0.0 {
            Err(StatsError::BadParams)
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

impl ::rand::distributions::Distribution<f64> for Exp {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> f64 {
        ziggurat::sample_exp_1(r) / self.rate
    }
}

impl ContinuousCDF<f64, f64> for Exp {
    /// Calculates the cumulative distribution function for the
    /// exponential distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
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
    /// ```ignore
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
}

impl Min<f64> for Exp {
    /// Returns the minimum value in the domain of the exponential
    /// distribution representable by a double precision float
    ///
    /// # Formula
    ///
    /// ```ignore
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
    /// ```ignore
    /// INF
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
    /// ```ignore
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
    /// ```ignore
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
    /// ```ignore
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
    /// ```ignore
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
    /// ```ignore
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
    /// ```ignore
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
    /// ```ignore
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
    /// ```ignore
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
#[cfg(all(test, feature = "nightly"))]
mod tests {
    use std::f64;
    use crate::statistics::*;
    use crate::distribution::{ContinuousCDF, Continuous, Exp};
    use crate::distribution::internal::*;
    use crate::consts::ACC;

    fn try_create(rate: f64) -> Exp {
        let n = Exp::new(rate);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(rate: f64) {
        let n = try_create(rate);
        assert_eq!(rate, n.rate());
    }

    fn bad_create_case(rate: f64) {
        let n = Exp::new(rate);
        assert!(n.is_err());
    }

    fn get_value<F>(rate: f64, eval: F) -> f64
        where F: Fn(Exp) -> f64
    {
        let n = try_create(rate);
        eval(n)
    }

    fn test_case<F>(rate: f64, expected: f64, eval: F)
        where F: Fn(Exp) -> f64
    {
        let x = get_value(rate, eval);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(rate: f64, expected: f64, acc: f64, eval: F)
        where F: Fn(Exp) -> f64
    {
        let x = get_value(rate, eval);
        assert_almost_eq!(expected, x, acc);
    }

    fn test_is_nan<F>(rate: f64, eval: F)
        where F : Fn(Exp) -> f64
    {
        let x = get_value(rate, eval);
        assert!(x.is_nan());
    }

    #[test]
    fn test_create() {
        create_case(0.1);
        create_case(1.0);
        create_case(10.0);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(f64::NAN);
        bad_create_case(0.0);
        bad_create_case(-1.0);
        bad_create_case(-10.0);
    }

    #[test]
    fn test_mean() {
        let mean = |x: Exp| x.mean().unwrap();
        test_case(0.1, 10.0, mean);
        test_case(1.0, 1.0, mean);
        test_case(10.0, 0.1, mean);
    }

    #[test]
    fn test_variance() {
        let variance = |x: Exp| x.variance().unwrap();
        test_almost(0.1, 100.0, 1e-13, variance);
        test_case(1.0, 1.0, variance);
        test_case(10.0, 0.01, variance);
    }

    #[test]
    fn test_entropy() {
        let entropy = |x: Exp| x.entropy().unwrap();
        test_almost(0.1, 3.302585092994045684018, 1e-15, entropy);
        test_case(1.0, 1.0, entropy);
        test_almost(10.0, -1.302585092994045684018, 1e-15, entropy);
    }

    #[test]
    fn test_skewness() {
        let skewness = |x: Exp| x.skewness().unwrap();
        test_case(0.1, 2.0, skewness);
        test_case(1.0, 2.0, skewness);
        test_case(10.0, 2.0, skewness);
    }

    #[test]
    fn test_median() {
        let median = |x: Exp| x.median();
        test_almost(0.1, 6.931471805599453094172, 1e-15, median);
        test_case(1.0, f64::consts::LN_2, median);
        test_case(10.0, 0.06931471805599453094172, median);
    }

    #[test]
    fn test_mode() {
        let mode = |x: Exp| x.mode().unwrap();
        test_case(0.1, 0.0, mode);
        test_case(1.0, 0.0, mode);
        test_case(10.0, 0.0, mode);
    }

    #[test]
    fn test_min_max() {
        let min = |x: Exp| x.min();
        let max = |x: Exp| x.max();
        test_case(0.1, 0.0, min);
        test_case(1.0, 0.0, min);
        test_case(10.0, 0.0, min);
        test_case(0.1, f64::INFINITY, max);
        test_case(1.0, f64::INFINITY, max);
        test_case(10.0, f64::INFINITY, max);
    }

    #[test]
    fn test_pdf() {
        let pdf = |arg: f64| move |x: Exp| x.pdf(arg);
        test_case(0.1, 0.1, pdf(0.0));
        test_case(1.0, 1.0, pdf(0.0));
        test_case(10.0, 10.0, pdf(0.0));
        test_is_nan(f64::INFINITY, pdf(0.0));
        test_case(0.1, 0.09900498337491680535739, pdf(0.1));
        test_almost(1.0, 0.9048374180359595731642, 1e-15, pdf(0.1));
        test_case(10.0, 3.678794411714423215955, pdf(0.1));
        test_is_nan(f64::INFINITY, pdf(0.1));
        test_case(0.1, 0.09048374180359595731642, pdf(1.0));
        test_case(1.0, 0.3678794411714423215955, pdf(1.0));
        test_almost(10.0, 4.539992976248485153559e-4, 1e-19, pdf(1.0));
        test_is_nan(f64::INFINITY, pdf(1.0));
        test_case(0.1, 0.0, pdf(f64::INFINITY));
        test_case(1.0, 0.0, pdf(f64::INFINITY));
        test_case(10.0, 0.0, pdf(f64::INFINITY));
        test_is_nan(f64::INFINITY, pdf(f64::INFINITY));
    }

    #[test]
    fn test_neg_pdf() {
        let pdf = |arg: f64| move |x: Exp| x.pdf(arg);
        test_case(0.1, 0.0, pdf(-1.0));
    }

    #[test]
    fn test_ln_pdf() {
        let ln_pdf = |arg: f64| move |x: Exp| x.ln_pdf(arg);
        test_almost(0.1, -2.302585092994045684018, 1e-15, ln_pdf(0.0));
        test_case(1.0, 0.0, ln_pdf(0.0));
        test_case(10.0, 2.302585092994045684018, ln_pdf(0.0));
        test_is_nan(f64::INFINITY, ln_pdf(0.0));
        test_almost(0.1, -2.312585092994045684018, 1e-15, ln_pdf(0.1));
        test_case(1.0, -0.1, ln_pdf(0.1));
        test_almost(10.0, 1.302585092994045684018, 1e-15, ln_pdf(0.1));
        test_is_nan(f64::INFINITY, ln_pdf(0.1));
        test_case(0.1, -2.402585092994045684018, ln_pdf(1.0));
        test_case(1.0, -1.0, ln_pdf(1.0));
        test_case(10.0, -7.697414907005954315982, ln_pdf(1.0));
        test_is_nan(f64::INFINITY, ln_pdf(1.0));
        test_case(0.1, f64::NEG_INFINITY, ln_pdf(f64::INFINITY));
        test_case(1.0, f64::NEG_INFINITY, ln_pdf(f64::INFINITY));
        test_case(10.0, f64::NEG_INFINITY, ln_pdf(f64::INFINITY));
        test_is_nan(f64::INFINITY, ln_pdf(f64::INFINITY));
    }

    #[test]
    fn test_neg_ln_pdf() {
        let ln_pdf = |arg: f64| move |x: Exp| x.ln_pdf(arg);
        test_case(0.1, f64::NEG_INFINITY, ln_pdf(-1.0));
    }

    #[test]
    fn test_cdf() {
        let cdf = |arg: f64| move |x: Exp| x.cdf(arg);
        test_case(0.1, 0.0, cdf(0.0));
        test_case(1.0, 0.0, cdf(0.0));
        test_case(10.0, 0.0, cdf(0.0));
        test_is_nan(f64::INFINITY, cdf(0.0));
        test_almost(0.1, 0.009950166250831946426094, 1e-16, cdf(0.1));
        test_almost(1.0, 0.0951625819640404268358, 1e-16, cdf(0.1));
        test_case(10.0, 0.6321205588285576784045, cdf(0.1));
        test_case(f64::INFINITY, 1.0, cdf(0.1));
        test_almost(0.1, 0.0951625819640404268358, 1e-16, cdf(1.0));
        test_case(1.0, 0.6321205588285576784045, cdf(1.0));
        test_case(10.0, 0.9999546000702375151485, cdf(1.0));
        test_case(f64::INFINITY, 1.0, cdf(1.0));
        test_case(0.1, 1.0, cdf(f64::INFINITY));
        test_case(1.0, 1.0, cdf(f64::INFINITY));
        test_case(10.0, 1.0, cdf(f64::INFINITY));
        test_case(f64::INFINITY, 1.0, cdf(f64::INFINITY));
    }

    #[test]
    fn test_sf() {
        let sf = |arg: f64| move |x: Exp| x.sf(arg);
        test_case(0.1, 1.0, sf(0.0));
        test_case(1.0, 1.0, sf(0.0));
        test_case(10.0, 1.0, sf(0.0));
        test_is_nan(f64::INFINITY, sf(0.0));
        test_almost(0.1, 0.9900498337491681, 1e-16, sf(0.1));
        test_almost(1.0, 0.9048374180359595, 1e-16, sf(0.1));
        test_almost(10.0, 0.36787944117144233, 1e-15, sf(0.1));
        test_case(f64::INFINITY, 0.0, sf(0.1));
    }

    #[test]
    fn test_neg_cdf() {
        let cdf = |arg: f64| move |x: Exp| x.cdf(arg);
        test_case(0.1, 0.0, cdf(-1.0));
    }

    #[test]
    fn test_neg_sf() {
        let sf = |arg: f64| move |x: Exp| x.sf(arg);
        test_case(0.1, 1.0, sf(-1.0));
    }

    #[test]
    fn test_continuous() {
        test::check_continuous_distribution(&try_create(0.5), 0.0, 10.0);
        test::check_continuous_distribution(&try_create(1.5), 0.0, 20.0);
        test::check_continuous_distribution(&try_create(2.5), 0.0, 50.0);
    }
}
