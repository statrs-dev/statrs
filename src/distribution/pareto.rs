use crate::distribution::{Continuous, ContinuousCDF};
use crate::statistics::*;
use crate::{Result, StatsError};
use rand::distributions::OpenClosed01;
use rand::Rng;
use std::f64;

/// Implements the [Pareto](https://en.wikipedia.org/wiki/Pareto_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Pareto, Continuous};
/// use statrs::statistics::Distribution;
/// use statrs::prec;
///
/// let p = Pareto::new(1.0, 2.0).unwrap();
/// assert_eq!(p.mean().unwrap(), 2.0);
/// assert!(prec::almost_eq(p.pdf(2.0), 0.25, 1e-15));
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Pareto {
    scale: f64,
    shape: f64,
}

impl Pareto {
    /// Constructs a new Pareto distribution with scale `scale`, and `shape`
    /// shape.
    ///
    /// # Errors
    ///
    /// Returns an error if any of `scale` or `shape` are `NaN`.
    /// Returns an error if `scale <= 0.0` or `shape <= 0.0`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Pareto;
    ///
    /// let mut result = Pareto::new(1.0, 2.0);
    /// assert!(result.is_ok());
    ///
    /// result = Pareto::new(0.0, 0.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(scale: f64, shape: f64) -> Result<Pareto> {
        let is_nan = scale.is_nan() || shape.is_nan();
        if is_nan || scale <= 0.0 || shape <= 0.0 {
            Err(StatsError::BadParams)
        } else {
            Ok(Pareto { scale, shape })
        }
    }

    /// Returns the scale of the Pareto distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Pareto;
    ///
    /// let n = Pareto::new(1.0, 2.0).unwrap();
    /// assert_eq!(n.scale(), 1.0);
    /// ```
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Returns the shape of the Pareto distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Pareto;
    ///
    /// let n = Pareto::new(1.0, 2.0).unwrap();
    /// assert_eq!(n.shape(), 2.0);
    /// ```
    pub fn shape(&self) -> f64 {
        self.shape
    }
}

impl ::rand::distributions::Distribution<f64> for Pareto {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        // Inverse transform sampling
        let u: f64 = rng.sample(OpenClosed01);
        self.scale * u.powf(-1.0 / self.shape)
    }
}

impl ContinuousCDF<f64, f64> for Pareto {
    /// Calculates the cumulative distribution function for the Pareto
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// if x < x_m {
    ///     0
    /// } else {
    ///     1 - (x_m/x)^α
    /// }
    /// ```
    ///
    /// where `x_m` is the scale and `α` is the shape
    fn cdf(&self, x: f64) -> f64 {
        if x < self.scale {
            0.0
        } else {
            1.0 - (self.scale / x).powf(self.shape)
        }
    }

    /// Calculates the survival function for the Pareto
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// if x < x_m {
    ///     1
    /// } else {
    ///     (x_m/x)^α
    /// }
    /// ```
    ///
    /// where `x_m` is the scale and `α` is the shape
    fn sf(&self, x: f64) -> f64 {
        if x < self.scale {
            1.0
        } else {
            (self.scale / x).powf(self.shape)
        }
    }
}

impl Min<f64> for Pareto {
    /// Returns the minimum value in the domain of the Pareto distribution
    /// representable by a double precision float
    ///
    /// # Formula
    ///
    /// ```ignore
    /// x_m
    /// ```
    ///
    /// where `x_m` is the scale
    fn min(&self) -> f64 {
        self.scale
    }
}

impl Max<f64> for Pareto {
    /// Returns the maximum value in the domain of the Pareto distribution
    /// representable by a double precision float
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

impl Distribution<f64> for Pareto {
    /// Returns the mean of the Pareto distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// if α <= 1 {
    ///     INF
    /// } else {
    ///     (α * x_m)/(α - 1)
    /// }
    /// ```
    ///
    /// where `x_m` is the scale and `α` is the shape
    fn mean(&self) -> Option<f64> {
        if self.shape <= 1.0 {
            None
        } else {
            Some((self.shape * self.scale) / (self.shape - 1.0))
        }
    }
    /// Returns the variance of the Pareto distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// if α <= 2 {
    ///     INF
    /// } else {
    ///     (x_m/(α - 1))^2 * (α/(α - 2))
    /// }
    /// ```
    ///
    /// where `x_m` is the scale and `α` is the shape
    fn variance(&self) -> Option<f64> {
        if self.shape <= 2.0 {
            None
        } else {
            let a = self.scale / (self.shape - 1.0); // just a temporary variable
            Some(a * a * self.shape / (self.shape - 2.0))
        }
    }
    /// Returns the entropy for the Pareto distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln(α/x_m) - 1/α - 1
    /// ```
    ///
    /// where `x_m` is the scale and `α` is the shape
    fn entropy(&self) -> Option<f64> {
        Some(self.shape.ln() - self.scale.ln() - (1.0 / self.shape) - 1.0)
    }
    /// Returns the skewness of the Pareto distribution
    ///
    /// # Panics
    ///
    /// If `α <= 3.0`
    ///
    /// where `α` is the shape
    ///
    /// # Formula
    ///
    /// ```ignore
    ///     (2*(α + 1)/(α - 3))*sqrt((α - 2)/α)
    /// ```
    ///
    /// where `α` is the shape
    fn skewness(&self) -> Option<f64> {
        if self.shape <= 3.0 {
            None
        } else {
            Some(
                (2.0 * (self.shape + 1.0) / (self.shape - 3.0))
                    * ((self.shape - 2.0) / self.shape).sqrt(),
            )
        }
    }
}

impl Median<f64> for Pareto {
    /// Returns the median of the Pareto distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// x_m*2^(1/α)
    /// ```
    ///
    /// where `x_m` is the scale and `α` is the shape
    fn median(&self) -> f64 {
        self.scale * (2f64.powf(1.0 / self.shape))
    }
}

impl Mode<Option<f64>> for Pareto {
    /// Returns the mode of the Pareto distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// x_m
    /// ```
    ///
    /// where `x_m` is the scale
    fn mode(&self) -> Option<f64> {
        Some(self.scale)
    }
}

impl Continuous<f64, f64> for Pareto {
    /// Calculates the probability density function for the Pareto distribution
    /// at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// if x < x_m {
    ///     0
    /// } else {
    ///     (α * x_m^α)/(x^(α + 1))
    /// }
    /// ```
    ///
    /// where `x_m` is the scale and `α` is the shape
    fn pdf(&self, x: f64) -> f64 {
        if x < self.scale {
            0.0
        } else {
            (self.shape * self.scale.powf(self.shape)) / x.powf(self.shape + 1.0)
        }
    }

    /// Calculates the log probability density function for the Pareto
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// if x < x_m {
    ///     -INF
    /// } else {
    ///     ln(α) + α*ln(x_m) - (α + 1)*ln(x)
    /// }
    /// ```
    ///
    /// where `x_m` is the scale and `α` is the shape
    fn ln_pdf(&self, x: f64) -> f64 {
        if x < self.scale {
            f64::NEG_INFINITY
        } else {
            self.shape.ln() + self.shape * self.scale.ln() - (self.shape + 1.0) * x.ln()
        }
    }
}

#[rustfmt::skip]
#[cfg(all(test, feature = "nightly"))]
mod tests {
    use crate::statistics::*;
    use crate::distribution::{ContinuousCDF, Continuous, Pareto};
    use crate::distribution::internal::*;
    use crate::consts::ACC;

    fn try_create(scale: f64, shape: f64) -> Pareto {
        let p = Pareto::new(scale, shape);
        assert!(p.is_ok());
        p.unwrap()
    }

    fn create_case(scale: f64, shape: f64) {
        let p = try_create(scale, shape);
        assert_eq!(scale, p.scale());
        assert_eq!(shape, p.shape());
    }

    fn bad_create_case(scale: f64, shape: f64) {
        let p = Pareto::new(scale, shape);
        assert!(p.is_err());
    }

    fn get_value<T, F>(scale: f64, shape: f64, eval: F) -> T
        where F: Fn(Pareto) -> T
    {
        let p = try_create(scale, shape);
        eval(p)
    }

    fn test_case<F>(scale: f64, shape: f64, expected: f64, eval: F)
        where F: Fn(Pareto) -> f64
    {
        let x = get_value(scale, shape, eval);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(scale: f64, shape: f64, expected: f64, acc: f64, eval: F)
        where F: Fn(Pareto) -> f64
    {
        let p = try_create(scale, shape);
        let x = eval(p);
        assert_almost_eq!(expected, x, acc);
    }

    #[test]
    fn test_create() {
        create_case(10.0, 0.1);
        create_case(5.0, 1.0);
        create_case(0.1, 10.0);
        create_case(10.0, 100.0);
        create_case(1.0, f64::INFINITY);
        create_case(f64::INFINITY, f64::INFINITY);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(0.0, 0.0);
        bad_create_case(1.0, -1.0);
        bad_create_case(-1.0, 1.0);
        bad_create_case(-1.0, -1.0);
        bad_create_case(f64::NAN, 1.0);
        bad_create_case(1.0, f64::NAN);
        bad_create_case(f64::NAN, f64::NAN);
    }

    #[test]
    fn test_variance() {
        let variance = |x: Pareto| x.variance().unwrap();
        test_case(1.0, 3.0, 0.75, variance);
        test_almost(10.0, 10.0, 125.0 / 81.0, 1e-13, variance);
    }

    #[test]
    #[should_panic]
    fn test_variance_degen() {
        let variance = |x: Pareto| x.variance().unwrap();
        test_case(1.0, 1.0, f64::INFINITY, variance); // shape <= 2.0
    }

    #[test]
    fn test_entropy() {
        let entropy = |x: Pareto| x.entropy().unwrap();
        test_case(0.1, 0.1, -11.0, entropy);
        test_case(1.0, 1.0, -2.0, entropy);
        test_case(10.0, 10.0, -1.1, entropy);
        test_case(3.0, 1.0, -2.0 - 3f64.ln(), entropy);
        test_case(1.0, 3.0, -4.0/3.0 + 3f64.ln(), entropy);
    }

    #[test]
    fn test_skewness() {
        let skewness = |x: Pareto| x.skewness().unwrap();
        test_case(1.0, 4.0, 5.0*2f64.sqrt(), skewness);
        test_case(1.0, 100.0, (707.0/485.0)*2f64.sqrt(), skewness);
    }

    #[test]
    #[should_panic]
    fn test_skewness_invalid_shape() {
        let skewness = |x: Pareto| x.skewness().unwrap();
        get_value(1.0, 3.0, skewness);
    }

    #[test]
    fn test_mode() {
        let mode = |x: Pareto| x.mode().unwrap();
        test_case(0.1, 1.0, 0.1, mode);
        test_case(2.0, 1.0, 2.0, mode);
        test_case(10.0, f64::INFINITY, 10.0, mode);
        test_case(f64::INFINITY, 1.0, f64::INFINITY, mode);
    }

    #[test]
    fn test_median() {
        let median = |x: Pareto| x.median();
        test_case(0.1, 0.1, 102.4, median);
        test_case(1.0, 1.0, 2.0, median);
        test_case(10.0, 10.0, 10.0*2f64.powf(0.1), median);
        test_case(3.0, 0.5, 12.0, median);
        test_case(10.0, f64::INFINITY, 10.0, median);
    }

    #[test]
    fn test_min_max() {
        let min = |x: Pareto| x.min();
        let max = |x: Pareto| x.max();
        test_case(0.2, f64::INFINITY, 0.2, min);
        test_case(10.0, f64::INFINITY, 10.0, min);
        test_case(f64::INFINITY, 1.0, f64::INFINITY, min);
        test_case(1.0, 0.1, f64::INFINITY, max);
        test_case(3.0, 10.0, f64::INFINITY, max);
    }

    #[test]
    fn test_pdf() {
        let pdf = |arg: f64| move |x: Pareto| x.pdf(arg);
        test_case(1.0, 1.0, 0.0, pdf(0.1));
        test_case(1.0, 1.0, 1.0, pdf(1.0));
        test_case(1.0, 1.0, 4.0/9.0, pdf(1.5));
        test_case(1.0, 1.0, 1.0/25.0, pdf(5.0));
        test_case(1.0, 1.0, 1.0/2500.0, pdf(50.0));
        test_case(1.0, 4.0, 4.0, pdf(1.0));
        test_case(1.0, 4.0, 128.0/243.0, pdf(1.5));
        test_case(1.0, 4.0, 1.0/78125000.0, pdf(50.0));
        test_case(3.0, 2.0, 2.0/3.0, pdf(3.0));
        test_case(3.0, 2.0, 18.0/125.0, pdf(5.0));
        test_almost(25.0, 100.0, 1.5777218104420236e-30, 1e-50, pdf(50.0));
        test_almost(100.0, 25.0, 6.6003546737276816e-6, 1e-16, pdf(150.0));
        test_case(1.0, 2.0, 0.0, pdf(f64::INFINITY));
    }

    #[test]
    fn test_ln_pdf() {
        let ln_pdf = |arg: f64| move |x: Pareto| x.ln_pdf(arg);
        test_case(1.0, 1.0, f64::NEG_INFINITY, ln_pdf(0.1));
        test_case(1.0, 1.0, 0.0, ln_pdf(1.0));
        test_almost(1.0, 1.0, 4f64.ln() - 9f64.ln(), 1e-14, ln_pdf(1.5));
        test_almost(1.0, 1.0, -(25f64.ln()), 1e-14, ln_pdf(5.0));
        test_almost(1.0, 1.0, -(2500f64.ln()), 1e-14, ln_pdf(50.0));
        test_almost(1.0, 4.0, 4f64.ln(), 1e-14, ln_pdf(1.0));
        test_almost(1.0, 4.0, 128f64.ln() - 243f64.ln(), 1e-14, ln_pdf(1.5));
        test_almost(1.0, 4.0, -(78125000f64.ln()), 1e-14, ln_pdf(50.0));
        test_almost(3.0, 2.0, 2f64.ln() - 3f64.ln(), 1e-14, ln_pdf(3.0));
        test_almost(3.0, 2.0, 18f64.ln() - 125f64.ln(), 1e-14, ln_pdf(5.0));
        test_almost(25.0, 100.0, 1.5777218104420236e-30f64.ln(), 1e-12, ln_pdf(50.0));
        test_almost(100.0, 25.0, 6.6003546737276816e-6f64.ln(), 1e-12, ln_pdf(150.0));
        test_case(1.0, 2.0, f64::NEG_INFINITY, ln_pdf(f64::INFINITY));
    }

    #[test]
    fn test_cdf() {
        let cdf = |arg: f64| move |x: Pareto| x.cdf(arg);
        test_case(0.1, 0.1, 0.0, cdf(0.1));
        test_case(1.0, 1.0, 0.0, cdf(1.0));
        test_case(5.0, 5.0, 0.0, cdf(2.0));
        test_case(7.0, 7.0, 0.9176457, cdf(10.0));
        test_case(10.0, 10.0, 50700551.0/60466176.0, cdf(12.0));
        test_case(5.0, 1.0, 0.5, cdf(10.0));
        test_case(3.0, 10.0, 1023.0/1024.0, cdf(6.0));
        test_case(1.0, 1.0, 1.0, cdf(f64::INFINITY));
    }

    #[test]
    fn test_sf() {
        let sf = |arg: f64| move |x: Pareto| x.sf(arg);
        test_case(0.1, 0.1, 1.0, sf(0.1));
        test_case(1.0, 1.0, 1.0, sf(1.0));
        test_case(5.0, 5.0, 1.0, sf(2.0));
        test_almost(7.0, 7.0, 0.08235429999999999, 1e-14, sf(10.0));
        test_almost(10.0, 10.0, 0.16150558288984573, 1e14, sf(12.0));
        test_case(5.0, 1.0, 0.5, sf(10.0));
        test_almost(3.0, 10.0, 0.0009765625, 1e-14, sf(6.0));
        test_case(1.0, 1.0, 0.0, sf(f64::INFINITY));
    }

    #[test]
    fn test_continuous() {
        test::check_continuous_distribution(&try_create(1.0, 10.0), 1.0, 10.0);
        test::check_continuous_distribution(&try_create(0.1, 2.0), 0.1, 100.0);
    }
}
