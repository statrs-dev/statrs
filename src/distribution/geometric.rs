use crate::distribution::{Discrete, DiscreteCDF};
use crate::statistics::*;
use crate::{Result, StatsError};
use rand::distributions::OpenClosed01;
use rand::Rng;
use std::{f64, u64};

/// Implements the
/// [Geometric](https://en.wikipedia.org/wiki/Geometric_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Geometric, Discrete};
/// use statrs::statistics::Distribution;
///
/// let n = Geometric::new(0.3).unwrap();
/// assert_eq!(n.mean().unwrap(), 1.0 / 0.3);
/// assert_eq!(n.pmf(1), 0.3);
/// assert_eq!(n.pmf(2), 0.21);
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Geometric {
    p: f64,
}

impl Geometric {
    /// Constructs a new shifted geometric distribution with a probability
    /// of `p`
    ///
    /// # Errors
    ///
    /// Returns an error if `p` is not in `(0, 1]`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Geometric;
    ///
    /// let mut result = Geometric::new(0.5);
    /// assert!(result.is_ok());
    ///
    /// result = Geometric::new(0.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(p: f64) -> Result<Geometric> {
        if p <= 0.0 || p > 1.0 || p.is_nan() {
            Err(StatsError::BadParams)
        } else {
            Ok(Geometric { p })
        }
    }

    /// Returns the probability `p` of the geometric
    /// distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Geometric;
    ///
    /// let n = Geometric::new(0.5).unwrap();
    /// assert_eq!(n.p(), 0.5);
    /// ```
    pub fn p(&self) -> f64 {
        self.p
    }
}

impl ::rand::distributions::Distribution<f64> for Geometric {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> f64 {
        if ulps_eq!(self.p, 1.0) {
            1.0
        } else {
            let x: f64 = r.sample(OpenClosed01);
            x.log(1.0 - self.p).ceil()
        }
    }
}

impl DiscreteCDF<u64, f64> for Geometric {
    /// Calculates the cumulative distribution function for the geometric
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 1 - (1 - p) ^ x
    /// ```
    fn cdf(&self, x: u64) -> f64 {
        if x == 0 {
            0.0
        } else {
            // 1 - (1 - p) ^ x = 1 - exp(log(1 - p)*x)
            //                 = -expm1(log1p(-p)*x))
            //                 = -((-p).ln_1p() * x).exp_m1()
            -((-self.p).ln_1p() * (x as f64)).exp_m1()
        }
    }

    /// Calculates the survival function for the geometric
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (1 - p) ^ x
    /// ```
    fn sf(&self, x: u64) -> f64 {
        // (1-p) ^ x = exp(log(1-p)*x)
        //           = exp(log1p(-p) * x)
        if x == 0 {
            1.0
        } else {
            ((-self.p).ln_1p() * (x as f64)).exp()
        }
    }
}

impl Min<u64> for Geometric {
    /// Returns the minimum value in the domain of the
    /// geometric distribution representable by a 64-bit
    /// integer
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 1
    /// ```
    fn min(&self) -> u64 {
        1
    }
}

impl Max<u64> for Geometric {
    /// Returns the maximum value in the domain of the
    /// geometric distribution representable by a 64-bit
    /// integer
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 2^63 - 1
    /// ```
    fn max(&self) -> u64 {
        u64::MAX
    }
}

impl Distribution<f64> for Geometric {
    /// Returns the mean of the geometric distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 1 / p
    /// ```
    fn mean(&self) -> Option<f64> {
        Some(1.0 / self.p)
    }
    /// Returns the standard deviation of the geometric distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (1 - p) / p^2
    /// ```
    fn variance(&self) -> Option<f64> {
        Some((1.0 - self.p) / (self.p * self.p))
    }
    /// Returns the entropy of the geometric distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (-(1 - p) * log_2(1 - p) - p * log_2(p)) / p
    /// ```
    fn entropy(&self) -> Option<f64> {
        let inv = 1.0 / self.p;
        Some(-inv * (1. - self.p).log(2.0) + (inv - 1.).log(2.0))
    }
    /// Returns the skewness of the geometric distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (2 - p) / sqrt(1 - p)
    /// ```
    fn skewness(&self) -> Option<f64> {
        if ulps_eq!(self.p, 1.0) {
            return Some(f64::INFINITY);
        };
        Some((2.0 - self.p) / (1.0 - self.p).sqrt())
    }
}

impl Mode<Option<u64>> for Geometric {
    /// Returns the mode of the geometric distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 1
    /// ```
    fn mode(&self) -> Option<u64> {
        Some(1)
    }
}

impl Median<f64> for Geometric {
    /// Returns the median of the geometric distribution
    ///
    /// # Remarks
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ceil(-1 / log_2(1 - p))
    /// ```
    fn median(&self) -> f64 {
        (-f64::consts::LN_2 / (1.0 - self.p).ln()).ceil()
    }
}

impl Discrete<u64, f64> for Geometric {
    /// Calculates the probability mass function for the geometric
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (1 - p)^(x - 1) * p
    /// ```
    fn pmf(&self, x: u64) -> f64 {
        if x == 0 {
            0.0
        } else {
            (1.0 - self.p).powi(x as i32 - 1) * self.p
        }
    }

    /// Calculates the log probability mass function for the geometric
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln((1 - p)^(x - 1) * p)
    /// ```
    fn ln_pmf(&self, x: u64) -> f64 {
        if x == 0 {
            f64::NEG_INFINITY
        } else if ulps_eq!(self.p, 1.0) && x == 1 {
            0.0
        } else if ulps_eq!(self.p, 1.0) {
            f64::NEG_INFINITY
        } else {
            ((x - 1) as f64 * (1.0 - self.p).ln()) + self.p.ln()
        }
    }
}

#[rustfmt::skip]
#[cfg(all(test, feature = "nightly"))]
mod tests {
    use std::fmt::Debug;
    use crate::statistics::*;
    use crate::distribution::{DiscreteCDF, Discrete, Geometric};
    use crate::distribution::internal::*;
    use crate::consts::ACC;

    fn try_create(p: f64) -> Geometric {
        let n = Geometric::new(p);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(p: f64) {
        let n = try_create(p);
        assert_eq!(p, n.p());
    }

    fn bad_create_case(p: f64) {
        let n = Geometric::new(p);
        assert!(n.is_err());
    }

    fn get_value<T, F>(p: f64, eval: F) -> T
        where T: PartialEq + Debug,
              F: Fn(Geometric) -> T
    {
        let n = try_create(p);
        eval(n)
    }

    fn test_case<T, F>(p: f64, expected: T, eval: F)
        where T: PartialEq + Debug,
              F: Fn(Geometric) -> T
    {
        let x = get_value(p, eval);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(p: f64, expected: f64, acc: f64, eval: F)
        where F: Fn(Geometric) -> f64
    {
        let x = get_value(p, eval);
        assert_almost_eq!(expected, x, acc);
    }

    fn test_is_nan<F>(p: f64, eval: F)
        where F: Fn(Geometric) -> f64
    {
        let x = get_value(p, eval);
        assert!(x.is_nan());
    }

    #[test]
    fn test_create() {
        create_case(0.3);
        create_case(1.0);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(f64::NAN);
        bad_create_case(0.0);
        bad_create_case(-1.0);
        bad_create_case(2.0);
    }

    #[test]
    fn test_mean() {
        let mean = |x: Geometric| x.mean().unwrap();
        test_case(0.3, 1.0 / 0.3, mean);
        test_case(1.0, 1.0, mean);
    }

    #[test]
    fn test_variance() {
        let variance = |x: Geometric| x.variance().unwrap();
        test_case(0.3, 0.7 / (0.3 * 0.3), variance);
        test_case(1.0, 0.0, variance);
    }

    #[test]
    fn test_entropy() {
        let entropy = |x: Geometric| x.entropy().unwrap();
        test_almost(0.3, 2.937636330768973333333, 1e-14, entropy);
        test_is_nan(1.0, entropy);
    }

    #[test]
    fn test_skewness() {
        let skewness = |x: Geometric| x.skewness().unwrap();
        test_almost(0.3, 2.031888635868469187947, 1e-15, skewness);
        test_case(1.0, f64::INFINITY, skewness);
    }

    #[test]
    fn test_median() {
        let median = |x: Geometric| x.median();
        test_case(0.0001, 6932.0, median);
        test_case(0.1, 7.0, median);
        test_case(0.3, 2.0, median);
        test_case(0.9, 1.0, median);
        // test_case(0.99, 1.0, median);
        test_case(1.0, 0.0, median);
    }

    #[test]
    fn test_mode() {
        let mode = |x: Geometric| x.mode().unwrap();
        test_case(0.3, 1, mode);
        test_case(1.0, 1, mode);
    }

    #[test]
    fn test_min_max() {
        let min = |x: Geometric| x.min();
        let max = |x: Geometric| x.max();
        test_case(0.3, 1, min);
        test_case(0.3, u64::MAX, max);
    }

    #[test]
    fn test_pmf() {
        let pmf = |arg: u64| move |x: Geometric| x.pmf(arg);
        test_case(0.3, 0.3, pmf(1));
        test_case(0.3, 0.21, pmf(2));
        test_case(1.0, 1.0, pmf(1));
        test_case(1.0, 0.0, pmf(2));
        test_almost(0.5, 0.5, 1e-10, pmf(1));
        test_almost(0.5, 0.25, 1e-10, pmf(2));
    }

    #[test]
    fn test_pmf_lower_bound() {
        let pmf = |arg: u64| move |x: Geometric| x.pmf(arg);
        test_case(0.3, 0.0, pmf(0));
    }

    #[test]
    fn test_ln_pmf() {
        let ln_pmf = |arg: u64| move |x: Geometric| x.ln_pmf(arg);
        test_almost(0.3, -1.203972804325935992623, 1e-15, ln_pmf(1));
        test_almost(0.3, -1.560647748264668371535, 1e-15, ln_pmf(2));
        test_case(1.0, 0.0, ln_pmf(1));
        test_case(1.0, f64::NEG_INFINITY, ln_pmf(2));
    }

    #[test]
    fn test_ln_pmf_lower_bound() {
        let ln_pmf = |arg: u64| move |x: Geometric| x.ln_pmf(arg);
        test_case(0.3, f64::NEG_INFINITY, ln_pmf(0));
    }

    #[test]
    fn test_cdf() {
        let cdf = |arg: u64| move |x: Geometric| x.cdf(arg);
        test_case(1.0, 1.0, cdf(1));
        test_case(1.0, 1.0, cdf(2));
        test_almost(0.5, 0.5, 1e-15, cdf(1));
        test_almost(0.5, 0.75, 1e-15, cdf(2));
    }

    #[test]
    fn test_sf() {
        let sf = |arg: u64| move |x: Geometric| x.sf(arg);
        test_case(1.0, 0.0, sf(1));
        test_case(1.0, 0.0, sf(2));
        test_almost(0.5, 0.5, 1e-15, sf(1));
        test_almost(0.5, 0.25, 1e-15, sf(2));
    }

    #[test]
    fn test_cdf_small_p() {
        //
        // Expected values were computed with the arbitrary precision
        // library mpmath in Python, e.g.:
        //
        //   import mpmath
        //   mpmath.mp.dps = 400
        //   p = mpmath.mpf(1e-9)
        //   k = 5
        //   cdf = float(1 - (1 - p)**k)
        //   # cdf is 4.99999999e-09
        //
        let geom = Geometric::new(1e-9f64).unwrap();

        let cdf = geom.cdf(5u64);
        let expected = 4.99999999e-09;
        assert_relative_eq!(cdf, expected, epsilon = 0.0, max_relative = 1e-15);
    }

    #[test]
    fn test_sf_small_p() {
        let geom = Geometric::new(1e-9f64).unwrap();

        let sf = geom.sf(5u64);
        let expected = 0.999999995;
        assert_relative_eq!(sf, expected, epsilon = 0.0, max_relative = 1e-15);
    }

    #[test]
    fn test_cdf_very_small_p() {
        //
        // Expected values were computed with the arbitrary precision
        // library mpmath in Python, e.g.:
        //
        //   import mpmath
        //   mpmath.mp.dps = 400
        //   p = mpmath.mpf(1e-17)
        //   k = 100000000000000
        //   cdf = float(1 - (1 - p)**k)
        //   # cdf is 0.0009995001666250085
        //
        let geom = Geometric::new(1e-17f64).unwrap();

        let cdf = geom.cdf(10u64);
        let expected = 1e-16f64;
        assert_relative_eq!(cdf, expected, epsilon = 0.0, max_relative = 1e-15);

        let cdf = geom.cdf(100000000000000u64);
        let expected = 0.0009995001666250085f64;
        assert_relative_eq!(cdf, expected, epsilon = 0.0, max_relative = 1e-15);
    }

    #[test]
    fn test_sf_very_small_p() {
        let geom = Geometric::new(1e-17f64).unwrap();

        let sf = geom.sf(10u64);
        let expected =  0.9999999999999999;
        assert_relative_eq!(sf, expected, epsilon = 0.0, max_relative = 1e-15);

        let sf = geom.sf(100000000000000u64);
        let expected = 0.999000499833375;
        assert_relative_eq!(sf, expected, epsilon = 0.0, max_relative = 1e-15);
    }

    #[test]
    fn test_cdf_lower_bound() {
        let cdf = |arg: u64| move |x: Geometric| x.cdf(arg);
        test_case(0.3, 0.0, cdf(0));
    }

    #[test]
    fn test_sf_lower_bound() {
        let sf = |arg: u64| move |x: Geometric| x.sf(arg);
        test_case(0.3, 1.0, sf(0));
    }

    #[test]
    fn test_discrete() {
        test::check_discrete_distribution(&try_create(0.3), 100);
        test::check_discrete_distribution(&try_create(0.6), 100);
        test::check_discrete_distribution(&try_create(1.0), 1);
    }
}
