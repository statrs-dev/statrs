use crate::distribution::{ziggurat, Continuous, Univariate};
use crate::statistics::*;
use crate::{Result, StatsError};
use rand::distributions::Distribution;
use rand::Rng;
use std::f64;

/// Implements the
/// [Exponential](https://en.wikipedia.org/wiki/Exponential_distribution)
/// distribution and is a special case of the
/// [Gamma](https://en.wikipedia.org/wiki/Gamma_distribution) distribution
/// (referenced [here](./struct.Gamma.html))
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Exponential, Continuous};
/// use statrs::statistics::Mean;
///
/// let n = Exponential::new(1.0).unwrap();
/// assert_eq!(n.mean(), 1.0);
/// assert_eq!(n.pdf(1.0), 0.3678794411714423215955);
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Exponential {
    rate: f64,
}

impl Exponential {
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
    /// use statrs::distribution::Exponential;
    ///
    /// let mut result = Exponential::new(1.0);
    /// assert!(result.is_ok());
    ///
    /// result = Exponential::new(-1.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(rate: f64) -> Result<Exponential> {
        if rate.is_nan() || rate <= 0.0 {
            Err(StatsError::BadParams)
        } else {
            Ok(Exponential { rate })
        }
    }

    /// Returns the rate of the exponential distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Exponential;
    ///
    /// let n = Exponential::new(1.0).unwrap();
    /// assert_eq!(n.rate(), 1.0);
    /// ```
    pub fn rate(&self) -> f64 {
        self.rate
    }
}

impl Distribution<f64> for Exponential {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> f64 {
        ziggurat::sample_exp_1(r) / self.rate
    }
}

impl Univariate<f64, f64> for Exponential {
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
}

impl Min<f64> for Exponential {
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

impl Max<f64> for Exponential {
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

impl Mean<f64> for Exponential {
    /// Returns the mean of the exponential distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 1 / λ
    /// ```
    ///
    /// where `λ` is the rate
    fn mean(&self) -> f64 {
        1.0 / self.rate
    }
}

impl Variance<f64> for Exponential {
    /// Returns the variance of the exponential distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 1 / λ^2
    /// ```
    ///
    /// where `λ` is the rate
    fn variance(&self) -> f64 {
        1.0 / (self.rate * self.rate)
    }

    /// Returns the standard deviation of the exponential distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// sqrt(1 / λ^2)
    /// ```
    ///
    /// where `λ` is the rate
    fn std_dev(&self) -> f64 {
        1.0 / self.rate
    }
}

impl Entropy<f64> for Exponential {
    /// Returns the entropy of the exponential distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 1 - ln(λ)
    /// ```
    ///
    /// where `λ` is the rate
    fn entropy(&self) -> f64 {
        1.0 - self.rate.ln()
    }
}

impl Skewness<f64> for Exponential {
    /// Returns the skewness of the exponential distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 2
    /// ```
    fn skewness(&self) -> f64 {
        2.0
    }
}

impl Median<f64> for Exponential {
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

impl Mode<f64> for Exponential {
    /// Returns the mode of the exponential distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 0
    /// ```
    fn mode(&self) -> f64 {
        0.0
    }
}

impl Continuous<f64, f64> for Exponential {
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
#[cfg(test)]
mod test {
    use std::f64;
    use crate::statistics::*;
    use crate::distribution::{Univariate, Continuous, Exponential};
    use crate::distribution::internal::*;

    fn try_create(rate: f64) -> Exponential {
        let n = Exponential::new(rate);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(rate: f64) {
        let n = try_create(rate);
        assert_eq!(rate, n.rate());
    }

    fn bad_create_case(rate: f64) {
        let n = Exponential::new(rate);
        assert!(n.is_err());
    }

    fn get_value<F>(rate: f64, eval: F) -> f64
        where F: Fn(Exponential) -> f64
    {
        let n = try_create(rate);
        eval(n)
    }

    fn test_case<F>(rate: f64, expected: f64, eval: F)
        where F: Fn(Exponential) -> f64
    {
        let x = get_value(rate, eval);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(rate: f64, expected: f64, acc: f64, eval: F)
        where F: Fn(Exponential) -> f64
    {
        let x = get_value(rate, eval);
        assert_almost_eq!(expected, x, acc);
    }

    fn test_is_nan<F>(rate: f64, eval: F)
        where F : Fn(Exponential) -> f64
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
        test_case(0.1, 10.0, |x| x.mean());
        test_case(1.0, 1.0, |x| x.mean());
        test_case(10.0, 0.1, |x| x.mean());
    }

    #[test]
    fn test_variance() {
        test_almost(0.1, 100.0, 1e-13, |x| x.variance());
        test_case(1.0, 1.0, |x| x.variance());
        test_case(10.0, 0.01, |x| x.variance());
    }

    #[test]
    fn test_std_dev() {
        test_case(0.1, 10.0, |x| x.std_dev());
        test_case(1.0, 1.0, |x| x.std_dev());
        test_case(10.0, 0.1, |x| x.std_dev());
    }

    #[test]
    fn test_entropy() {
        test_almost(0.1, 3.302585092994045684018, 1e-15, |x| x.entropy());
        test_case(1.0, 1.0, |x| x.entropy());
        test_almost(10.0, -1.302585092994045684018, 1e-15, |x| x.entropy());
    }

    #[test]
    fn test_skewness() {
        test_case(0.1, 2.0, |x| x.skewness());
        test_case(1.0, 2.0, |x| x.skewness());
        test_case(10.0, 2.0, |x| x.skewness());
    }

    #[test]
    fn test_median() {
        test_almost(0.1, 6.931471805599453094172, 1e-15, |x| x.median());
        test_case(1.0, f64::consts::LN_2, |x| x.median());
        test_case(10.0, 0.06931471805599453094172, |x| x.median());
    }

    #[test]
    fn test_mode() {
        test_case(0.1, 0.0, |x| x.mode());
        test_case(1.0, 0.0, |x| x.mode());
        test_case(10.0, 0.0, |x| x.mode());
    }

    #[test]
    fn test_min_max() {
        test_case(0.1, 0.0, |x| x.min());
        test_case(1.0, 0.0, |x| x.min());
        test_case(10.0, 0.0, |x| x.min());
        test_case(0.1, f64::INFINITY, |x| x.max());
        test_case(1.0, f64::INFINITY, |x| x.max());
        test_case(10.0, f64::INFINITY, |x| x.max());
    }

    #[test]
    fn test_pdf() {
        test_case(0.1, 0.1, |x| x.pdf(0.0));
        test_case(1.0, 1.0, |x| x.pdf(0.0));
        test_case(10.0, 10.0, |x| x.pdf(0.0));
        test_is_nan(f64::INFINITY, |x| x.pdf(0.0));
        test_case(0.1, 0.09900498337491680535739, |x| x.pdf(0.1));
        test_almost(1.0, 0.9048374180359595731642, 1e-15, |x| x.pdf(0.1));
        test_case(10.0, 3.678794411714423215955, |x| x.pdf(0.1));
        test_is_nan(f64::INFINITY, |x| x.pdf(0.1));
        test_case(0.1, 0.09048374180359595731642, |x| x.pdf(1.0));
        test_case(1.0, 0.3678794411714423215955, |x| x.pdf(1.0));
        test_almost(10.0, 4.539992976248485153559e-4, 1e-19, |x| x.pdf(1.0));
        test_is_nan(f64::INFINITY, |x| x.pdf(1.0));
        test_case(0.1, 0.0, |x| x.pdf(f64::INFINITY));
        test_case(1.0, 0.0, |x| x.pdf(f64::INFINITY));
        test_case(10.0, 0.0, |x| x.pdf(f64::INFINITY));
        test_is_nan(f64::INFINITY, |x| x.pdf(f64::INFINITY));
    }

    #[test]
    fn test_neg_pdf() {
        test_case(0.1, 0.0, |x| x.pdf(-1.0));
    }

    #[test]
    fn test_ln_pdf() {
        test_almost(0.1, -2.302585092994045684018, 1e-15, |x| x.ln_pdf(0.0));
        test_case(1.0, 0.0, |x| x.ln_pdf(0.0));
        test_case(10.0, 2.302585092994045684018, |x| x.ln_pdf(0.0));
        test_is_nan(f64::INFINITY, |x| x.ln_pdf(0.0));
        test_almost(0.1, -2.312585092994045684018, 1e-15, |x| x.ln_pdf(0.1));
        test_case(1.0, -0.1, |x| x.ln_pdf(0.1));
        test_almost(10.0, 1.302585092994045684018, 1e-15, |x| x.ln_pdf(0.1));
        test_is_nan(f64::INFINITY, |x| x.ln_pdf(0.1));
        test_case(0.1, -2.402585092994045684018, |x| x.ln_pdf(1.0));
        test_case(1.0, -1.0, |x| x.ln_pdf(1.0));
        test_case(10.0, -7.697414907005954315982, |x| x.ln_pdf(1.0));
        test_is_nan(f64::INFINITY, |x| x.ln_pdf(1.0));
        test_case(0.1, f64::NEG_INFINITY, |x| x.ln_pdf(f64::INFINITY));
        test_case(1.0, f64::NEG_INFINITY, |x| x.ln_pdf(f64::INFINITY));
        test_case(10.0, f64::NEG_INFINITY, |x| x.ln_pdf(f64::INFINITY));
        test_is_nan(f64::INFINITY, |x| x.ln_pdf(f64::INFINITY));
    }

    #[test]
    fn test_neg_ln_pdf() {
        test_case(0.1, f64::NEG_INFINITY, |x| x.ln_pdf(-1.0));
    }

    #[test]
    fn test_cdf() {
        test_case(0.1, 0.0, |x| x.cdf(0.0));
        test_case(1.0, 0.0, |x| x.cdf(0.0));
        test_case(10.0, 0.0, |x| x.cdf(0.0));
        test_is_nan(f64::INFINITY, |x| x.cdf(0.0));
        test_almost(0.1, 0.009950166250831946426094, 1e-16, |x| x.cdf(0.1));
        test_almost(1.0, 0.0951625819640404268358, 1e-16, |x| x.cdf(0.1));
        test_case(10.0, 0.6321205588285576784045, |x| x.cdf(0.1));
        test_case(f64::INFINITY, 1.0, |x| x.cdf(0.1));
        test_almost(0.1, 0.0951625819640404268358, 1e-16, |x| x.cdf(1.0));
        test_case(1.0, 0.6321205588285576784045, |x| x.cdf(1.0));
        test_case(10.0, 0.9999546000702375151485, |x| x.cdf(1.0));
        test_case(f64::INFINITY, 1.0, |x| x.cdf(1.0));
        test_case(0.1, 1.0, |x| x.cdf(f64::INFINITY));
        test_case(1.0, 1.0, |x| x.cdf(f64::INFINITY));
        test_case(10.0, 1.0, |x| x.cdf(f64::INFINITY));
        test_case(f64::INFINITY, 1.0, |x| x.cdf(f64::INFINITY));
    }

    #[test]
    fn test_neg_cdf() {
        test_case(0.1, 0.0, |x| x.cdf(-1.0));
    }

    #[test]
    fn test_continuous() {
        test::check_continuous_distribution(&try_create(0.5), 0.0, 10.0);
        test::check_continuous_distribution(&try_create(1.5), 0.0, 20.0);
        test::check_continuous_distribution(&try_create(2.5), 0.0, 50.0);
    }
}
