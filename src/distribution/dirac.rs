use crate::distribution::ContinuousCDF;
use crate::statistics::*;

/// Implements the [Dirac Delta](https://en.wikipedia.org/wiki/Dirac_delta_function#As_a_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Dirac, Continuous};
/// use statrs::statistics::Distribution;
///
/// let n = Dirac::new(3.0).unwrap();
/// assert_eq!(n.mean().unwrap(), 3.0);
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Dirac(f64);

/// Represents the errors that can occur when creating a [`Dirac`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum DiracError {
    /// The value v is NaN.
    ValueInvalid,
}

impl std::fmt::Display for DiracError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            DiracError::ValueInvalid => write!(f, "Value v is NaN"),
        }
    }
}

impl std::error::Error for DiracError {}

impl Dirac {
    /// Constructs a new dirac distribution function at value `v`.
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
    pub fn new(v: f64) -> Result<Self, DiracError> {
        if v.is_nan() {
            Err(DiracError::ValueInvalid)
        } else {
            Ok(Dirac(v))
        }
    }
}

impl std::fmt::Display for Dirac {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "δ_{}", self.0)
    }
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl ::rand::distributions::Distribution<f64> for Dirac {
    fn sample<R: ::rand::Rng + ?Sized>(&self, _: &mut R) -> f64 {
        self.0
    }
}

impl ContinuousCDF<f64, f64> for Dirac {
    /// Calculates the cumulative distribution function for the
    /// dirac distribution at `x`
    ///
    /// Where the value is 1 if x > `v`, 0 otherwise.
    fn cdf(&self, x: f64) -> f64 {
        if x < self.0 {
            0.0
        } else {
            1.0
        }
    }

    /// Calculates the survival function for the
    /// dirac distribution at `x`
    ///
    /// Where the value is 0 if x > `v`, 1 otherwise.
    fn sf(&self, x: f64) -> f64 {
        if x < self.0 {
            1.0
        } else {
            0.0
        }
    }
}

impl Min<f64> for Dirac {
    /// Returns the minimum value in the domain of the
    /// dirac distribution representable by a double precision float
    ///
    /// # Formula
    ///
    /// ```text
    /// v
    /// ```
    fn min(&self) -> f64 {
        self.0
    }
}

impl Max<f64> for Dirac {
    /// Returns the maximum value in the domain of the
    /// dirac distribution representable by a double precision float
    ///
    /// # Formula
    ///
    /// ```text
    /// v
    /// ```
    fn max(&self) -> f64 {
        self.0
    }
}

impl Distribution<f64> for Dirac {
    /// Returns the mean of the dirac distribution
    ///
    /// # Remarks
    ///
    /// Since the only value that can be produced by this distribution is `v` with probability
    /// 1, it is just `v`.
    fn mean(&self) -> Option<f64> {
        Some(self.0)
    }

    /// Returns the variance of the dirac distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// 0
    /// ```
    ///
    /// Since only one value can be produced there is no variance.
    fn variance(&self) -> Option<f64> {
        Some(0.0)
    }

    /// Returns the entropy of the dirac distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// 0
    /// ```
    ///
    /// Since this distribution has full certainty, it encodes no information
    fn entropy(&self) -> Option<f64> {
        Some(0.0)
    }

    /// Returns the skewness of the dirac distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// 0
    /// ```
    fn skewness(&self) -> Option<f64> {
        Some(0.0)
    }
}

impl Median<f64> for Dirac {
    /// Returns the median of the dirac distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// v
    /// ```
    ///
    /// where `v` is the point of the dirac distribution
    fn median(&self) -> f64 {
        self.0
    }
}

impl Mode<Option<f64>> for Dirac {
    /// Returns the mode of the dirac distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// v
    /// ```
    ///
    /// where `v` is the point of the dirac distribution
    fn mode(&self) -> Option<f64> {
        Some(self.0)
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing_boiler;

    testing_boiler!(v: f64; Dirac; DiracError);

    #[test]
    fn test_create() {
        create_ok(10.0);
        create_ok(-5.0);
        create_ok(10.0);
        create_ok(100.0);
        create_ok(f64::INFINITY);
    }

    #[test]
    fn test_bad_create() {
        create_err(f64::NAN);
    }

    #[test]
    fn test_variance() {
        let variance = |x: Dirac| x.variance().unwrap();
        test_exact(0.0, 0.0, variance);
        test_exact(-5.0, 0.0, variance);
        test_exact(f64::INFINITY, 0.0, variance);
    }

    #[test]
    fn test_entropy() {
        let entropy = |x: Dirac| x.entropy().unwrap();
        test_exact(0.0, 0.0, entropy);
        test_exact(f64::INFINITY, 0.0, entropy);
    }

    #[test]
    fn test_skewness() {
        let skewness = |x: Dirac| x.skewness().unwrap();
        test_exact(0.0, 0.0, skewness);
        test_exact(4.0, 0.0, skewness);
        test_exact(0.3, 0.0, skewness);
        test_exact(f64::INFINITY, 0.0, skewness);
    }

    #[test]
    fn test_mode() {
        let mode = |x: Dirac| x.mode().unwrap();
        test_exact(0.0, 0.0, mode);
        test_exact(3.0, 3.0, mode);
        test_exact(f64::INFINITY, f64::INFINITY, mode);
    }

    #[test]
    fn test_median() {
        let median = |x: Dirac| x.median();
        test_exact(0.0, 0.0, median);
        test_exact(3.0, 3.0, median);
        test_exact(f64::INFINITY, f64::INFINITY, median);
    }

    #[test]
    fn test_min_max() {
        let min = |x: Dirac| x.min();
        let max = |x: Dirac| x.max();
        test_exact(0.0, 0.0, min);
        test_exact(3.0, 3.0, min);
        test_exact(f64::INFINITY, f64::INFINITY, min);

        test_exact(0.0, 0.0, max);
        test_exact(3.0, 3.0, max);
        test_exact(f64::NEG_INFINITY, f64::NEG_INFINITY, max);
    }

    #[test]
    fn test_cdf() {
        let cdf = |arg: f64| move |x: Dirac| x.cdf(arg);
        test_exact(0.0, 1.0, cdf(0.0));
        test_exact(3.0, 1.0, cdf(3.0));
        test_exact(f64::INFINITY, 0.0, cdf(1.0));
        test_exact(f64::INFINITY, 1.0, cdf(f64::INFINITY));
    }

    #[test]
    fn test_sf() {
        let sf = |arg: f64| move |x: Dirac| x.sf(arg);
        test_exact(0.0, 0.0, sf(0.0));
        test_exact(3.0, 0.0, sf(3.0));
        test_exact(f64::INFINITY, 1.0, sf(1.0));
        test_exact(f64::INFINITY, 0.0, sf(f64::INFINITY));
    }
}
