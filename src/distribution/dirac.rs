use crate::distribution::{Continuous, Univariate};
use crate::statistics::*;
use crate::{Result, StatsError};
use rand::distributions::Distribution;
use rand::Rng;
use std::f64;

/// Implements the [Dirac Delta](https://en.wikipedia.org/wiki/Dirac_delta_function#As_a_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Dirac};
///
/// let n = Dirac::new(3.0).unwrap();
/// assert_eq!(n.mean(), 3.0);
/// assert_eq!(n.pdf(1.0), 0.0);
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Dirac(f64);

impl Dirac {
    ///  Constructs a new dirac distribution function at value `v`.
    ///
    /// # Errors
    ///
    /// Returns an error if `v` is not-a-number.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Dirac;
    ///
    /// let mut result = Dirac::new(0.0);
    /// assert!(result.is_ok());
    ///
    /// result = Dirac::new(f64::NAN);
    /// assert!(result.is_err());
    /// ```
    pub fn new(v: f64) -> Result<Self> {
        if v.is_nan() {
            Err(StatsError::BadParams)
        } else {
            Ok(Dirac(v))
        }
    }
}

impl Distribution<f64> for Dirac {
    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 {
        self.0
    }
}

impl Univariate<f64, f64> for Dirac {
    /// Calculates the cumulative distribution function for the
    /// dirac distribution at `x`
    ///
    /// Where the value is 1 if x > `v`, 0 otherwise.
    ///
    fn cdf(&self, x: f64) -> f64 {
      if x < self.0 { 0.0 } else { 1.0 }
    }
}

impl Min<f64> for Dirac {
    /// Returns the minimum value in the domain of the
    /// dirac distribution representable by a double precision float
    ///
    /// # Formula
    ///
    /// ```ignore
    /// v
    /// ```
    fn min(&self) -> f64 { self.0 }
}

impl Max<f64> for Dirac {
    /// Returns the maximum value in the domain of the
    /// dirac distribution representable by a double precision float
    ///
    /// # Formula
    ///
    /// ```ignore
    /// v
    /// ```
    fn max(&self) -> f64 { self.0 }
}

impl Mean<f64> for Dirac {
    /// Returns the mean of the dirac distribution
    ///
    /// # Remarks
    ///
    /// Since the only value that can be produced by this distribution is `v` with probability
    /// 1, it is just `v`.
    fn mean(&self) -> f64 { self.0 }
}

impl Variance<f64> for Dirac {
    /// Returns the variance of the dirac distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 0
    /// ```
    ///
    /// Since only one value can be produced there is no variance.
    fn variance(&self) -> f64 { 0.0 }

    /// Returns the standard deviation of the dirac distribution
    ///
    /// # Remarks
    ///
    /// Since there is no variance in draws from this distribution the standard deviation is
    /// also 0.
    fn std_dev(&self) -> f64 { 0.0 }
}

impl Entropy<f64> for Dirac {
    /// Returns the entropy of the dirac distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 0
    /// ```
    ///
    /// Since this distribution has full certainty, it encodes no information
    fn entropy(&self) -> f64 { 0.0 }
}

impl Skewness<f64> for Dirac {
    /// Returns the skewness of the dirac distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 0
    /// ```
    fn skewness(&self) -> f64 { 0.0 }
}

impl Median<f64> for Dirac {
    /// Returns the median of the dirac distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// v
    /// ```
    ///
    /// where `v` is the point of the dirac distribution
    fn median(&self) -> f64 {
        self.0
    }
}

impl Mode<f64> for Dirac {
    /// Returns the mode of the dirac distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// v
    /// ```
    ///
    /// where `v` is the point of the dirac distribution
    fn mode(&self) -> f64 {
        self.0
    }
}

impl Continuous<f64, f64> for Dirac {
    /// Calculates the probability density function for the dirac distribution
    /// at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 1 if x = v, 0 otherwise
    /// ```
    ///
    /// where `v` is point of this dirac distribution
    fn pdf(&self, x: f64) -> f64 {
      if x == self.0 { 1.0 } else { 0.0 }
    }

    /// Calculates the log probability density function for the dirac
    /// distribution
    /// at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln(1 if x = v, 0 otherwise)
    /// ```
    ///
    /// where `v` is the point of this dirac distribution
    ///
    /// # Remarks
    /// This distribution is usually negative infinity everywhere except at `v`.
    fn ln_pdf(&self, x: f64) -> f64 {
      if self.0 == x { 0.0 } else { f64::NEG_INFINITY }
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use std::f64;
    use crate::statistics::*;
    use crate::distribution::{Univariate, Continuous, Dirac};
    use crate::distribution::internal::*;

    fn try_create(v: f64) -> Dirac {
        let d = Dirac::new(v);
        assert!(d.is_ok());
        d.unwrap()
    }

    fn create_case(v: f64) {
        let d = try_create(v);
        assert_eq!(v, d.mean());
    }

    fn bad_create_case(v: f64) {
        let d = Dirac::new(v);
        assert!(d.is_err());
    }

    fn test_case<F>(v: f64, expected: f64, eval: F)
        where F: Fn(Dirac) -> f64
    {
        let x = eval(try_create(v));
        assert_eq!(expected, x);
    }

    #[test]
    fn test_create() {
        create_case(10.0);
        create_case(-5.0);
        create_case(10.0);
        create_case(100.0);
        create_case(f64::INFINITY);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(f64::NAN);
    }

    #[test]
    fn test_variance() {
        test_case(0.0, 0.0, |x| x.variance());
        test_case(-5.0, 0.0, |x| x.variance());
        test_case(f64::INFINITY, 0.0, |x| x.variance());
    }

    #[test]
    fn test_entropy() {
        test_case(0.0, 0.0, |x| x.entropy());
        test_case(f64::INFINITY, 0.0, |x| x.entropy());
    }

    #[test]
    fn test_skewness() {
        test_case(0.0, 0.0, |x| x.skewness());
        test_case(4.0, 0.0, |x| x.skewness());
        test_case(0.3, 0.0, |x| x.skewness());
        test_case(f64::INFINITY, 0.0, |x| x.skewness());
    }

    #[test]
    fn test_mode() {
        test_case(0.0, 0.0, |x| x.mode());
        test_case(3.0, 3.0, |x| x.mode());
        test_case(f64::INFINITY, f64::INFINITY, |x| x.mode());
    }

    #[test]
    fn test_median() {
        test_case(0.0, 0.0, |x| x.median());
        test_case(3.0, 3.0, |x| x.median());
        test_case(f64::INFINITY, f64::INFINITY, |x| x.median());
    }

    #[test]
    fn test_min_max() {
        test_case(0.0, 0.0, |x| x.min());
        test_case(3.0, 3.0, |x| x.min());
        test_case(f64::INFINITY, f64::INFINITY, |x| x.min());

        test_case(0.0, 0.0, |x| x.max());
        test_case(3.0, 3.0, |x| x.max());
        test_case(f64::NEG_INFINITY, f64::NEG_INFINITY, |x| x.max());
    }

    #[test]
    fn test_pdf() {
        test_case(0.0, 0.0, |x| x.pdf(1.0));
        test_case(3.0, 1.0, |x| x.pdf(3.0));
        test_case(f64::NEG_INFINITY, 0.0, |x| x.pdf(1.0));
        test_case(f64::NEG_INFINITY, 1.0, |x| x.pdf(f64::NEG_INFINITY));
    }

    #[test]
    fn test_ln_pdf() {
        test_case(0.0, 0.0, |x| x.ln_pdf(0.0));
        test_case(3.0, 0.0, |x| x.ln_pdf(3.0));
        test_case(f64::INFINITY, f64::NEG_INFINITY, |x| x.ln_pdf(1.0));
        test_case(f64::INFINITY, 0.0, |x| x.ln_pdf(f64::INFINITY));
    }

    #[test]
    fn test_cdf() {
        test_case(0.0, 1.0, |x| x.cdf(0.0));
        test_case(3.0, 1.0, |x| x.cdf(3.0));
        test_case(f64::INFINITY, 0.0, |x| x.cdf(1.0));
        test_case(f64::INFINITY, 1.0, |x| x.cdf(f64::INFINITY));
    }
}
