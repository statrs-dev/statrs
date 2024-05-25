use crate::distribution::{Discrete, DiscreteCDF};
use crate::statistics::*;
use crate::{Result, StatsError};
use rand::Rng;

/// Implements the [Discrete
/// Uniform](https://en.wikipedia.org/wiki/Discrete_uniform_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{DiscreteUniform, Discrete};
/// use statrs::statistics::Distribution;
///
/// let n = DiscreteUniform::new(0, 5).unwrap();
/// assert_eq!(n.mean().unwrap(), 2.5);
/// assert_eq!(n.pmf(3), 1.0 / 6.0);
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct DiscreteUniform {
    min: i64,
    max: i64,
}

impl DiscreteUniform {
    /// Constructs a new discrete uniform distribution with a minimum value
    /// of `min` and a maximum value of `max`.
    ///
    /// # Errors
    ///
    /// Returns an error if `max < min`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::DiscreteUniform;
    ///
    /// let mut result = DiscreteUniform::new(0, 5);
    /// assert!(result.is_ok());
    ///
    /// result = DiscreteUniform::new(5, 0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(min: i64, max: i64) -> Result<DiscreteUniform> {
        if max < min {
            Err(StatsError::BadParams)
        } else {
            Ok(DiscreteUniform { min, max })
        }
    }
}

impl ::rand::distributions::Distribution<f64> for DiscreteUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        rng.gen_range(self.min..=self.max) as f64
    }
}

impl DiscreteCDF<i64, f64> for DiscreteUniform {
    /// Calculates the cumulative distribution function for the
    /// discrete uniform distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// (floor(x) - min + 1) / (max - min + 1)
    /// ```
    fn cdf(&self, x: i64) -> f64 {
        if x < self.min {
            0.0
        } else if x >= self.max {
            1.0
        } else {
            let lower = self.min as f64;
            let upper = self.max as f64;
            let ans = (x as f64 - lower + 1.0) / (upper - lower + 1.0);
            if ans > 1.0 {
                1.0
            } else {
                ans
            }
        }
    }

    fn sf(&self, x: i64) -> f64 {
        // 1. - self.cdf(x)
        if x < self.min {
            1.0
        } else if x >= self.max {
            0.0
        } else {
            let lower = self.min as f64;
            let upper = self.max as f64;
            let ans = (upper - x as f64) / (upper - lower + 1.0);
            if ans > 1.0 {
                1.0
            } else {
                ans
            }
        }
    }
}

impl Min<i64> for DiscreteUniform {
    /// Returns the minimum value in the domain of the discrete uniform
    /// distribution
    ///
    /// # Remarks
    ///
    /// This is the same value as the minimum passed into the constructor
    fn min(&self) -> i64 {
        self.min
    }
}

impl Max<i64> for DiscreteUniform {
    /// Returns the maximum value in the domain of the discrete uniform
    /// distribution
    ///
    /// # Remarks
    ///
    /// This is the same value as the maximum passed into the constructor
    fn max(&self) -> i64 {
        self.max
    }
}

impl Distribution<f64> for DiscreteUniform {
    /// Returns the mean of the discrete uniform distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// (min + max) / 2
    /// ```
    fn mean(&self) -> Option<f64> {
        Some((self.min + self.max) as f64 / 2.0)
    }

    /// Returns the variance of the discrete uniform distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// ((max - min + 1)^2 - 1) / 12
    /// ```
    fn variance(&self) -> Option<f64> {
        let diff = (self.max - self.min) as f64;
        Some(((diff + 1.0) * (diff + 1.0) - 1.0) / 12.0)
    }

    /// Returns the entropy of the discrete uniform distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// ln(max - min + 1)
    /// ```
    fn entropy(&self) -> Option<f64> {
        let diff = (self.max - self.min) as f64;
        Some((diff + 1.0).ln())
    }

    /// Returns the skewness of the discrete uniform distribution
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

impl Median<f64> for DiscreteUniform {
    /// Returns the median of the discrete uniform distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// (max + min) / 2
    /// ```
    fn median(&self) -> f64 {
        (self.min + self.max) as f64 / 2.0
    }
}

impl Mode<Option<i64>> for DiscreteUniform {
    /// Returns the mode for the discrete uniform distribution
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
    fn mode(&self) -> Option<i64> {
        Some(((self.min + self.max) as f64 / 2.0).floor() as i64)
    }
}

impl Discrete<i64, f64> for DiscreteUniform {
    /// Calculates the probability mass function for the discrete uniform
    /// distribution at `x`
    ///
    /// # Remarks
    ///
    /// Returns `0.0` if `x` is not in `[min, max]`
    ///
    /// # Formula
    ///
    /// ```text
    /// 1 / (max - min + 1)
    /// ```
    fn pmf(&self, x: i64) -> f64 {
        if x >= self.min && x <= self.max {
            1.0 / (self.max - self.min + 1) as f64
        } else {
            0.0
        }
    }

    /// Calculates the log probability mass function for the discrete uniform
    /// distribution at `x`
    ///
    /// # Remarks
    ///
    /// Returns `f64::NEG_INFINITY` if `x` is not in `[min, max]`
    ///
    /// # Formula
    ///
    /// ```text
    /// ln(1 / (max - min + 1))
    /// ```
    fn ln_pmf(&self, x: i64) -> f64 {
        if x >= self.min && x <= self.max {
            -((self.max - self.min + 1) as f64).ln()
        } else {
            f64::NEG_INFINITY
        }
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use std::fmt::Debug;
    use crate::statistics::*;
    use crate::distribution::{DiscreteCDF, Discrete, DiscreteUniform};
    use crate::consts::ACC;

    fn try_create(min: i64, max: i64) -> DiscreteUniform {
        let n = DiscreteUniform::new(min, max);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(min: i64, max: i64) {
        let n = try_create(min, max);
        assert_eq!(min, n.min());
        assert_eq!(max, n.max());
    }

    fn bad_create_case(min: i64, max: i64) {
        let n = DiscreteUniform::new(min, max);
        assert!(n.is_err());
    }

    fn get_value<T, F>(min: i64, max: i64, eval: F) -> T
        where T: PartialEq + Debug,
              F: Fn(DiscreteUniform) -> T
    {
        let n = try_create(min, max);
        eval(n)
    }

    fn test_case<T, F>(min: i64, max: i64, expected: T, eval: F)
        where T: PartialEq + Debug,
              F: Fn(DiscreteUniform) -> T
    {
        let x = get_value(min, max, eval);
        assert_eq!(expected, x);
    }

    #[test]
    fn test_create() {
        create_case(-10, 10);
        create_case(0, 4);
        create_case(10, 20);
        create_case(20, 20);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(-1, -2);
        bad_create_case(6, 5);
    }

    #[test]
    fn test_mean() {
        let mean = |x: DiscreteUniform| x.mean().unwrap();
        test_case(-10, 10, 0.0, mean);
        test_case(0, 4, 2.0, mean);
        test_case(10, 20, 15.0, mean);
        test_case(20, 20, 20.0, mean);
    }

    #[test]
    fn test_variance() {
        let variance = |x: DiscreteUniform| x.variance().unwrap();
        test_case(-10, 10, 36.66666666666666666667, variance);
        test_case(0, 4, 2.0, variance);
        test_case(10, 20, 10.0, variance);
        test_case(20, 20, 0.0, variance);
    }

    #[test]
    fn test_entropy() {
        let entropy = |x: DiscreteUniform| x.entropy().unwrap();
        test_case(-10, 10, 3.0445224377234229965005979803657054342845752874046093, entropy);
        test_case(0, 4, 1.6094379124341003746007593332261876395256013542685181, entropy);
        test_case(10, 20, 2.3978952727983705440619435779651292998217068539374197, entropy);
        test_case(20, 20, 0.0, entropy);
    }

    #[test]
    fn test_skewness() {
        let skewness = |x: DiscreteUniform| x.skewness().unwrap();
        test_case(-10, 10, 0.0, skewness);
        test_case(0, 4, 0.0, skewness);
        test_case(10, 20, 0.0, skewness);
        test_case(20, 20, 0.0, skewness);
    }

    #[test]
    fn test_median() {
        let median = |x: DiscreteUniform| x.median();
        test_case(-10, 10, 0.0, median);
        test_case(0, 4, 2.0, median);
        test_case(10, 20, 15.0, median);
        test_case(20, 20, 20.0, median);
    }

    #[test]
    fn test_mode() {
        let mode = |x: DiscreteUniform| x.mode().unwrap();
        test_case(-10, 10, 0, mode);
        test_case(0, 4, 2, mode);
        test_case(10, 20, 15, mode);
        test_case(20, 20, 20, mode);
    }

    #[test]
    fn test_pmf() {
        let pmf = |arg: i64| move |x: DiscreteUniform| x.pmf(arg);
        test_case(-10, 10, 0.04761904761904761904762, pmf(-5));
        test_case(-10, 10, 0.04761904761904761904762, pmf(1));
        test_case(-10, 10, 0.04761904761904761904762, pmf(10));
        test_case(-10, -10, 0.0, pmf(0));
        test_case(-10, -10, 1.0, pmf(-10));
    }

    #[test]
    fn test_ln_pmf() {
        let ln_pmf = |arg: i64| move |x: DiscreteUniform| x.ln_pmf(arg);
        test_case(-10, 10, -3.0445224377234229965005979803657054342845752874046093, ln_pmf(-5));
        test_case(-10, 10, -3.0445224377234229965005979803657054342845752874046093, ln_pmf(1));
        test_case(-10, 10, -3.0445224377234229965005979803657054342845752874046093, ln_pmf(10));
        test_case(-10, -10, f64::NEG_INFINITY, ln_pmf(0));
        test_case(-10, -10, 0.0, ln_pmf(-10));
    }

    #[test]
    fn test_cdf() {
        let cdf = |arg: i64| move |x: DiscreteUniform| x.cdf(arg);
        test_case(-10, 10, 0.2857142857142857142857, cdf(-5));
        test_case(-10, 10, 0.5714285714285714285714, cdf(1));
        test_case(-10, 10, 1.0, cdf(10));
        test_case(-10, -10, 1.0, cdf(-10));
    }

    #[test]
    fn test_sf() {
        let sf = |arg: i64| move |x: DiscreteUniform| x.sf(arg);
        test_case(-10, 10, 0.7142857142857142857143, sf(-5));
        test_case(-10, 10, 0.42857142857142855, sf(1));
        test_case(-10, 10, 0.0, sf(10));
        test_case(-10, -10, 0.0, sf(-10));
    }

    #[test]
    fn test_cdf_lower_bound() {
        let cdf = |arg: i64| move |x: DiscreteUniform| x.cdf(arg);
        test_case(0, 3, 0.0, cdf(-1));
    }

    #[test]
    fn test_sf_lower_bound() {
        let sf = |arg: i64| move |x: DiscreteUniform| x.sf(arg);
        test_case(0, 3, 1.0, sf(-1));
    }

    #[test]
    fn test_cdf_upper_bound() {
        let cdf = |arg: i64| move |x: DiscreteUniform| x.cdf(arg);
        test_case(0, 3, 1.0, cdf(5));
    }
}
