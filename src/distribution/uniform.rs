use crate::distribution::{Continuous, ContinuousCDF};
use crate::statistics::*;
use std::f64;
use std::fmt::Debug;

/// Implements the [Continuous
/// Uniform](https://en.wikipedia.org/wiki/Uniform_distribution_(continuous))
/// distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Uniform, Continuous};
/// use statrs::statistics::Distribution;
///
/// let n = Uniform::new(0.0, 1.0).unwrap();
/// assert_eq!(n.mean().unwrap(), 0.5);
/// assert_eq!(n.pdf(0.5), 1.0);
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Uniform {
    min: f64,
    max: f64,
}

/// Represents the errors that can occur when creating a [`Uniform`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum UniformError {
    /// The minimum is NaN or infinite.
    MinInvalid,

    /// The maximum is NaN or infinite.
    MaxInvalid,

    /// The maximum is not greater than the minimum.
    MaxNotGreaterThanMin,
}

impl std::fmt::Display for UniformError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            UniformError::MinInvalid => write!(f, "Minimum is NaN or infinite"),
            UniformError::MaxInvalid => write!(f, "Maximum is NaN or infinite"),
            UniformError::MaxNotGreaterThanMin => {
                write!(f, "Maximum is not greater than the minimum")
            }
        }
    }
}

impl std::error::Error for UniformError {}

impl Uniform {
    /// Constructs a new uniform distribution with a min of `min` and a max
    /// of `max`.
    ///
    /// # Errors
    ///
    /// Returns an error if `min` or `max` are `NaN` or infinite.
    /// Returns an error if `min >= max`.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Uniform;
    /// use std::f64;
    ///
    /// let mut result = Uniform::new(0.0, 1.0);
    /// assert!(result.is_ok());
    ///
    /// result = Uniform::new(f64::NAN, f64::NAN);
    /// assert!(result.is_err());
    ///
    /// result = Uniform::new(f64::NEG_INFINITY, 1.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(min: f64, max: f64) -> Result<Uniform, UniformError> {
        if !min.is_finite() {
            return Err(UniformError::MinInvalid);
        }

        if !max.is_finite() {
            return Err(UniformError::MaxInvalid);
        }

        if min < max {
            Ok(Uniform { min, max })
        } else {
            Err(UniformError::MaxNotGreaterThanMin)
        }
    }

    /// Constructs a new standard uniform distribution with
    /// a lower bound 0 and an upper bound of 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Uniform;
    ///
    /// let uniform = Uniform::standard();
    /// ```
    pub fn standard() -> Self {
        Self { min: 0.0, max: 1.0 }
    }
}

impl Default for Uniform {
    fn default() -> Self {
        Self::standard()
    }
}

impl std::fmt::Display for Uniform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Uni([{},{}])", self.min, self.max)
    }
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl ::rand::distributions::Distribution<f64> for Uniform {
    fn sample<R: ::rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let d = rand::distributions::Uniform::new_inclusive(self.min, self.max);
        rng.sample(d)
    }
}

impl ContinuousCDF<f64, f64> for Uniform {
    /// Calculates the cumulative distribution function for the uniform
    /// distribution
    /// at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// (x - min) / (max - min)
    /// ```
    fn cdf(&self, x: f64) -> f64 {
        if x <= self.min {
            0.0
        } else if x >= self.max {
            1.0
        } else {
            (x - self.min) / (self.max - self.min)
        }
    }

    /// Calculates the survival function for the uniform
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// (max - x) / (max - min)
    /// ```
    fn sf(&self, x: f64) -> f64 {
        if x <= self.min {
            1.0
        } else if x >= self.max {
            0.0
        } else {
            (self.max - x) / (self.max - self.min)
        }
    }

    /// Finds the value of `x` where `F(p) = x`
    fn inverse_cdf(&self, p: f64) -> f64 {
        if !(0.0..=1.0).contains(&p) {
            panic!("p must be in [0, 1], was {p}");
        } else if p == 0.0 {
            self.min
        } else if p == 1.0 {
            self.max
        } else {
            (self.max - self.min) * p + self.min
        }
    }
}

impl Min<f64> for Uniform {
    fn min(&self) -> f64 {
        self.min
    }
}

impl Max<f64> for Uniform {
    fn max(&self) -> f64 {
        self.max
    }
}

impl Distribution<f64> for Uniform {
    /// Returns the mean for the continuous uniform distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// (min + max) / 2
    /// ```
    fn mean(&self) -> Option<f64> {
        Some((self.min + self.max) / 2.0)
    }

    /// Returns the variance for the continuous uniform distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// (max - min)^2 / 12
    /// ```
    fn variance(&self) -> Option<f64> {
        Some((self.max - self.min) * (self.max - self.min) / 12.0)
    }

    /// Returns the entropy for the continuous uniform distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// ln(max - min)
    /// ```
    fn entropy(&self) -> Option<f64> {
        Some((self.max - self.min).ln())
    }

    /// Returns the skewness for the continuous uniform distribution
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

impl Median<f64> for Uniform {
    /// Returns the median for the continuous uniform distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// (min + max) / 2
    /// ```
    fn median(&self) -> f64 {
        (self.min + self.max) / 2.0
    }
}

impl Mode<Option<f64>> for Uniform {
    /// Returns the mode for the continuous uniform distribution
    ///
    /// # Remarks
    ///
    /// Since every element has an equal probability, mode simply
    /// returns the middle element
    ///
    /// # Formula
    ///
    /// ```text
    /// N/A // (max + min) / 2 for the middle element
    /// ```
    fn mode(&self) -> Option<f64> {
        Some((self.min + self.max) / 2.0)
    }
}

impl Continuous<f64, f64> for Uniform {
    /// Calculates the probability density function for the continuous uniform
    /// distribution at `x`
    ///
    /// # Remarks
    ///
    /// Returns `0.0` if `x` is not in `[min, max]`
    ///
    /// # Formula
    ///
    /// ```text
    /// 1 / (max - min)
    /// ```
    fn pdf(&self, x: f64) -> f64 {
        if x < self.min || x > self.max {
            0.0
        } else {
            1.0 / (self.max - self.min)
        }
    }

    /// Calculates the log probability density function for the continuous
    /// uniform
    /// distribution at `x`
    ///
    /// # Remarks
    ///
    /// Returns `f64::NEG_INFINITY` if `x` is not in `[min, max]`
    ///
    /// # Formula
    ///
    /// ```text
    /// ln(1 / (max - min))
    /// ```
    fn ln_pdf(&self, x: f64) -> f64 {
        if x < self.min || x > self.max {
            f64::NEG_INFINITY
        } else {
            -(self.max - self.min).ln()
        }
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::internal::*;
    use crate::testing_boiler;

    testing_boiler!(min: f64, max: f64; Uniform; UniformError);

    #[test]
    fn test_create() {
        create_ok(0.0, 0.1);
        create_ok(0.0, 1.0);
        create_ok(-5.0, 11.0);
        create_ok(-5.0, 100.0);
    }

    #[test]
    fn test_bad_create() {
        let invalid = [
            (0.0, 0.0, UniformError::MaxNotGreaterThanMin),
            (f64::NAN, 1.0, UniformError::MinInvalid),
            (1.0, f64::NAN, UniformError::MaxInvalid),
            (f64::NAN, f64::NAN, UniformError::MinInvalid),
            (0.0, f64::INFINITY, UniformError::MaxInvalid),
            (1.0, 0.0, UniformError::MaxNotGreaterThanMin),
        ];
        
        for (min, max, err) in invalid {
            test_create_err(min, max, err);
        }
    }

    #[test]
    fn test_variance() {
        let variance = |x: Uniform| x.variance().unwrap();
        test_exact(-0.0, 2.0, 1.0 / 3.0, variance);
        test_exact(0.0, 2.0, 1.0 / 3.0, variance);
        test_absolute(0.1, 4.0, 1.2675, 1e-15, variance);
        test_exact(10.0, 11.0, 1.0 / 12.0, variance);
    }

    #[test]
    fn test_entropy() {
        let entropy = |x: Uniform| x.entropy().unwrap();
        test_exact(-0.0, 2.0, 0.6931471805599453094172, entropy);
        test_exact(0.0, 2.0, 0.6931471805599453094172, entropy);
        test_absolute(0.1, 4.0, 1.360976553135600743431, 1e-15, entropy);
        test_exact(1.0, 10.0, 2.19722457733621938279, entropy);
        test_exact(10.0, 11.0, 0.0, entropy);
    }

    #[test]
    fn test_skewness() {
        let skewness = |x: Uniform| x.skewness().unwrap();
        test_exact(-0.0, 2.0, 0.0, skewness);
        test_exact(0.0, 2.0, 0.0, skewness);
        test_exact(0.1, 4.0, 0.0, skewness);
        test_exact(1.0, 10.0, 0.0, skewness);
        test_exact(10.0, 11.0, 0.0, skewness);
    }

    #[test]
    fn test_mode() {
        let mode = |x: Uniform| x.mode().unwrap();
        test_exact(-0.0, 2.0, 1.0, mode);
        test_exact(0.0, 2.0, 1.0, mode);
        test_exact(0.1, 4.0, 2.05, mode);
        test_exact(1.0, 10.0, 5.5, mode);
        test_exact(10.0, 11.0, 10.5, mode);
    }

    #[test]
    fn test_median() {
        let median = |x: Uniform| x.median();
        test_exact(-0.0, 2.0, 1.0, median);
        test_exact(0.0, 2.0, 1.0, median);
        test_exact(0.1, 4.0, 2.05, median);
        test_exact(1.0, 10.0, 5.5, median);
        test_exact(10.0, 11.0, 10.5, median);
    }

    #[test]
    fn test_pdf() {
        let pdf = |arg: f64| move |x: Uniform| x.pdf(arg);
        test_exact(0.0, 0.1, 0.0, pdf(-5.0));
        test_exact(0.0, 0.1, 10.0, pdf(0.05));
        test_exact(0.0, 0.1, 0.0, pdf(5.0));
        test_exact(0.0, 1.0, 0.0, pdf(-5.0));
        test_exact(0.0, 1.0, 1.0, pdf(0.5));
        test_exact(0.0, 0.1, 0.0, pdf(5.0));
        test_exact(0.0, 10.0, 0.0, pdf(-5.0));
        test_exact(0.0, 10.0, 0.1, pdf(1.0));
        test_exact(0.0, 10.0, 0.1, pdf(5.0));
        test_exact(0.0, 10.0, 0.0, pdf(11.0));
        test_exact(-5.0, 100.0, 0.0, pdf(-10.0));
        test_exact(-5.0, 100.0, 0.009523809523809523809524, pdf(-5.0));
        test_exact(-5.0, 100.0, 0.009523809523809523809524, pdf(0.0));
        test_exact(-5.0, 100.0, 0.0, pdf(101.0));
    }

    #[test]
    fn test_ln_pdf() {
        let ln_pdf = |arg: f64| move |x: Uniform| x.ln_pdf(arg);
        test_exact(0.0, 0.1, f64::NEG_INFINITY, ln_pdf(-5.0));
        test_absolute(0.0, 0.1, 2.302585092994045684018, 1e-15, ln_pdf(0.05));
        test_exact(0.0, 0.1, f64::NEG_INFINITY, ln_pdf(5.0));
        test_exact(0.0, 1.0, f64::NEG_INFINITY, ln_pdf(-5.0));
        test_exact(0.0, 1.0, 0.0, ln_pdf(0.5));
        test_exact(0.0, 0.1, f64::NEG_INFINITY, ln_pdf(5.0));
        test_exact(0.0, 10.0, f64::NEG_INFINITY, ln_pdf(-5.0));
        test_exact(0.0, 10.0, -2.302585092994045684018, ln_pdf(1.0));
        test_exact(0.0, 10.0, -2.302585092994045684018, ln_pdf(5.0));
        test_exact(0.0, 10.0, f64::NEG_INFINITY, ln_pdf(11.0));
        test_exact(-5.0, 100.0, f64::NEG_INFINITY, ln_pdf(-10.0));
        test_exact(-5.0, 100.0, -4.653960350157523371101, ln_pdf(-5.0));
        test_exact(-5.0, 100.0, -4.653960350157523371101, ln_pdf(0.0));
        test_exact(-5.0, 100.0, f64::NEG_INFINITY, ln_pdf(101.0));
    }

    #[test]
    fn test_cdf() {
        let cdf = |arg: f64| move |x: Uniform| x.cdf(arg);
        test_exact(0.0, 0.1, 0.5, cdf(0.05));
        test_exact(0.0, 1.0, 0.5, cdf(0.5));
        test_exact(0.0, 10.0, 0.1, cdf(1.0));
        test_exact(0.0, 10.0, 0.5, cdf(5.0));
        test_exact(-5.0, 100.0, 0.0, cdf(-5.0));
        test_exact(-5.0, 100.0, 0.04761904761904761904762, cdf(0.0));
    }

    #[test]
    fn test_inverse_cdf() {
        let inverse_cdf = |arg: f64| move |x: Uniform| x.inverse_cdf(arg);
        test_exact(0.0, 0.1, 0.05, inverse_cdf(0.5));
        test_exact(0.0, 10.0, 5.0, inverse_cdf(0.5));
        test_exact(1.0, 10.0, 1.0, inverse_cdf(0.0));
        test_exact(1.0, 10.0, 4.0, inverse_cdf(1.0 / 3.0));
        test_exact(1.0, 10.0, 10.0, inverse_cdf(1.0));
    }

    #[test]
    fn test_cdf_lower_bound() {
        let cdf = |arg: f64| move |x: Uniform| x.cdf(arg);
        test_exact(0.0, 3.0, 0.0, cdf(-1.0));
    }

    #[test]
    fn test_cdf_upper_bound() {
        let cdf = |arg: f64| move |x: Uniform| x.cdf(arg);
        test_exact(0.0, 3.0, 1.0, cdf(5.0));
    }


    #[test]
    fn test_sf() {
        let sf = |arg: f64| move |x: Uniform| x.sf(arg);
        test_exact(0.0, 0.1, 0.5, sf(0.05));
        test_exact(0.0, 1.0, 0.5, sf(0.5));
        test_exact(0.0, 10.0, 0.9, sf(1.0));
        test_exact(0.0, 10.0, 0.5, sf(5.0));
        test_exact(-5.0, 100.0, 1.0, sf(-5.0));
        test_exact(-5.0, 100.0, 0.9523809523809523, sf(0.0));
    }

    #[test]
    fn test_sf_lower_bound() {
        let sf = |arg: f64| move |x: Uniform| x.sf(arg);
        test_exact(0.0, 3.0, 1.0, sf(-1.0));
    }

    #[test]
    fn test_sf_upper_bound() {
        let sf = |arg: f64| move |x: Uniform| x.sf(arg);
        test_exact(0.0, 3.0, 0.0, sf(5.0));
    }

    #[test]
    fn test_continuous() {
        test::check_continuous_distribution(&create_ok(0.0, 10.0), 0.0, 10.0);
        test::check_continuous_distribution(&create_ok(-2.0, 15.0), -2.0, 15.0);
    }

    #[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
    #[test]
    fn test_samples_in_range() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        use rand::distributions::Distribution;

        let seed = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
        ];
        let mut r: StdRng = SeedableRng::from_seed(seed);

        let min = -0.5;
        let max = 0.5;
        let num_trials = 10_000;
        let n = create_ok(min, max);

        assert!((0..num_trials)
            .map(|_| n.sample::<StdRng>(&mut r))
            .all(|v| (min <= v) && (v < max))
        );
    }

    #[test]
    fn test_default() {
        let n = Uniform::default();

        let n_mean = n.mean().unwrap();
        let n_std  = n.std_dev().unwrap();

        // Check that the mean of the distribution is close to 1 / 2
        assert_almost_eq!(n_mean, 0.5, 1e-15);
        // Check that the standard deviation of the distribution is close to 1 / sqrt(12)
        assert_almost_eq!(n_std, 0.288_675_134_594_812_9, 1e-15);
    }
}
