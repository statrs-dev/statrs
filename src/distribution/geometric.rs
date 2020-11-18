use crate::distribution::{Discrete, Univariate};
use crate::statistics::*;
use crate::{Result, StatsError};
use rand::distributions::Distribution;
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
/// use statrs::statistics::Mean;
///
/// let n = Geometric::new(0.3).unwrap();
/// assert_eq!(n.mean(), 1.0 / 0.3);
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

impl Distribution<f64> for Geometric {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> f64 {
        if ulps_eq!(self.p, 1.0) {
            1.0
        } else {
            let x: f64 = r.sample(OpenClosed01);
            x.log(1.0 - self.p).ceil()
        }
    }
}

impl Univariate<u64, f64> for Geometric {
    /// Calculates the cumulative distribution function for the geometric
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 1 - (1 - p) ^ (x + 1)
    /// ```
    fn cdf(&self, x: f64) -> f64 {
        if x < 1.0 {
            0.0
        } else if x.is_infinite() {
            1.0
        } else {
            1.0 - (1.0 - self.p).powf(x.floor())
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

impl Mean<f64> for Geometric {
    /// Returns the mean of the geometric distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 1 / p
    /// ```
    fn mean(&self) -> f64 {
        1.0 / self.p
    }
}

impl Variance<f64> for Geometric {
    /// Returns the standard deviation of the geometric distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (1 - p) / p^2
    /// ```
    fn variance(&self) -> f64 {
        (1.0 - self.p) / (self.p * self.p)
    }

    /// Returns the standard deviation of the geometric distribution
    ///
    /// # Remarks
    ///
    /// Returns `NAN` if `p` is `1`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// sqrt(1 - p) / p
    /// ```
    fn std_dev(&self) -> f64 {
        (1.0 - self.p).sqrt() / self.p
    }
}

impl Entropy<f64> for Geometric {
    /// Returns the entropy of the geometric distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (-(1 - p) * log_2(1 - p) - p * log_2(p)) / p
    /// ```
    fn entropy(&self) -> f64 {
        (-self.p * self.p.log(2.0) - (1.0 - self.p) * (1.0 - self.p).log(2.0)) / self.p
    }
}

impl Skewness<f64> for Geometric {
    /// Returns the skewness of the geometric distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (2 - p) / sqrt(1 - p)
    /// ```
    fn skewness(&self) -> f64 {
        (2.0 - self.p) / (1.0 - self.p).sqrt()
    }
}

impl Mode<u64> for Geometric {
    /// Returns the mode of the geometric distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 1
    /// ```
    fn mode(&self) -> u64 {
        1
    }
}

impl Median<f64> for Geometric {
    /// Returns the median of the geometric distribution
    ///
    /// # Remarks
    ///
    /// Returns `1` if `p` is `1`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ceil(-1 / log_2(1 - p))
    /// ```
    fn median(&self) -> f64 {
        if ulps_eq!(self.p, 1.0) {
            1.0
        } else {
            (-f64::consts::LN_2 / (1.0 - self.p).ln()).ceil()
        }
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
#[cfg(test)]
mod test {
    use std::fmt::Debug;
    use std::{u64, f64};
    use crate::statistics::*;
    use crate::distribution::{Univariate, Discrete, Geometric};
    use crate::distribution::internal::*;

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
        test_case(0.3, 1.0 / 0.3, |x| x.mean());
        test_case(1.0, 1.0, |x| x.mean());
    }

    #[test]
    fn test_variance() {
        test_case(0.3, 0.7 / (0.3 * 0.3), |x| x.variance());
        test_case(1.0, 0.0, |x| x.variance());
    }

    #[test]
    fn test_std_dev() {
        test_case(0.3, 0.7f64.sqrt() / 0.3, |x| x.std_dev());
        test_case(1.0, 0.0, |x| x.std_dev());
    }

    #[test]
    fn test_entropy() {
        test_almost(0.3, 2.937636330768973333333, 1e-14, |x| x.entropy());
        test_is_nan(1.0, |x| x.entropy());
    }

    #[test]
    fn test_skewness() {
        test_almost(0.3, 2.031888635868469187947, 1e-15, |x| x.skewness());
        test_case(1.0, f64::INFINITY, |x| x.skewness());
    }

    #[test]
    fn test_median() {
        test_case(0.0001, 6932.0, |x| x.median());
        test_case(0.1, 7.0, |x| x.median());
        test_case(0.3, 2.0, |x| x.median());
        test_case(0.9, 1.0, |x| x.median());
        test_case(1.0, 1.0, |x| x.median());
    }

    #[test]
    fn test_mode() {
        test_case(0.3, 1, |x| x.mode());
        test_case(1.0, 1, |x| x.mode());
    }

    #[test]
    fn test_min_max() {
        test_case(0.3, 1, |x| x.min());
        test_case(0.3, u64::MAX, |x| x.max());
    }

    #[test]
    fn test_pmf() {
        test_case(0.3, 0.3, |x| x.pmf(1));
        test_case(0.3, 0.21, |x| x.pmf(2));
        test_case(1.0, 1.0, |x| x.pmf(1));
        test_case(1.0, 0.0, |x| x.pmf(2));
        test_almost(0.5, 0.5, 1e-10, |x| x.pmf(1));
        test_almost(0.5, 0.25, 1e-10, |x| x.pmf(2));
    }

    #[test]
    fn test_pmf_lower_bound() {
        test_case(0.3, 0.0, |x| x.pmf(0));
    }

    #[test]
    fn test_ln_pmf() {
        test_almost(0.3, -1.203972804325935992623, 1e-15, |x| x.ln_pmf(1));
        test_almost(0.3, -1.560647748264668371535, 1e-15, |x| x.ln_pmf(2));
        test_case(1.0, 0.0, |x| x.ln_pmf(1));
        test_case(1.0, f64::NEG_INFINITY, |x| x.ln_pmf(2));
    }

    #[test]
    fn test_ln_pmf_lower_bound() {
        test_case(0.3, f64::NEG_INFINITY, |x| x.ln_pmf(0));
    }

    #[test]
    fn test_cdf() {
        test_case(1.0, 1.0, |x| x.cdf(1.0));
        test_case(1.0, 1.0, |x| x.cdf(2.0));
        test_almost(0.5, 0.5, 1e-10, |x| x.cdf(1.0));
        test_almost(0.5, 0.75, 1e-10, |x| x.cdf(2.0));
    }

    #[test]
    fn test_cdf_lower_bound() {
        test_case(0.3, 0.0, |x| x.cdf(0.0));
    }

    #[test]
    fn test_discrete() {
        test::check_discrete_distribution(&try_create(0.3), 100);
        test::check_discrete_distribution(&try_create(0.6), 100);
        test::check_discrete_distribution(&try_create(1.0), 1);
    }
}
