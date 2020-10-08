use crate::distribution::{Discrete, Univariate, Soliton};
use crate::statistics::*;
use crate::{Result, StatsError};
use rand::distributions::Distribution;
use rand::Rng;
use std::f64;

/// Implements the [Discrete
/// Uniform](https://en.wikipedia.org/wiki/Discrete_uniform_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{DiscreteUniform, Discrete};
/// use statrs::statistics::Mean;
///
/// let n = DiscreteUniform::new(0, 5).unwrap();
/// assert_eq!(n.mean(), 2.5);
/// assert_eq!(n.pmf(3), 1.0 / 6.0);
/// ```

#[derive(Debug, Clone, PartialEq)]
pub struct RobustSoliton {
    min: i64,
    max: i64,
    spike: Option<i64>,
    cumulative_probability_table: Vec<f64>,
    ripple: Ripple,
    fail_probability: f64,
}

impl RobustSoliton {
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
    pub fn new(max: i64, heuristic: bool, ripple: f64, fail_probability: f64) -> Result<RobustSoliton> {
        if max < 1 {
            Err(StatsError::BadParams)
        } else {
            let pmf_table: Vec<f64> = Vec::new();
            Ok(RobustSoliton {
                min: 1,
                max,
                spike: None,
                cumulative_probability_table: pmf_table,
                ripple: match heuristic {
                    false => Ripple::Fixed(ripple),
                    true => Ripple::Heuristic(ripple),
                },
                fail_probability,
            })
        }
    }

    fn normalization_factor(&self) -> f64 {
        let mut normalization_factor: f64 = 0.0;
        for i in 1..(self.max+1) {
            normalization_factor += self.soliton(i);
            normalization_factor += self.additive_probability(i);
        }
        normalization_factor
    }

    fn additive_probability(&self, val: i64) -> f64 {
        let ripple_size = self.ripple.size(self.max, self.fail_probability);
        let swap = self.max as f64 / ripple_size;
        if val == 0 || val > self.max {
            panic!("Point must be in range (0,max]. Given {} - {}", val, self.max);
        } else if (val as f64) < swap {
            ripple_size / ((val * self.max) as f64)
        } else if (val as f64) == swap {
            (ripple_size * (ripple_size / self.fail_probability).ln()) / (self.max as f64)
        } else {
            0.0
        }
    }
}

#[derive(PartialEq, Clone, Debug)]
enum Ripple {
    Fixed(f64),
    Heuristic(f64)
}

impl Ripple {
    fn size(&self, max: i64, fail_probability: f64) -> f64 {
        match self {
            &Ripple::Fixed(n) => n,
            &Ripple::Heuristic(n) => {
                n * (max as f64 / fail_probability).ln() * (max as f64).sqrt()
            }
        }
    }
}

impl Distribution<f64> for RobustSoliton {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> f64 {
        r.gen_range(self.min, self.max + 1) as f64
    }
}

impl Univariate<i64, f64> for RobustSoliton {
    /// Calculates the cumulative distribution function for the
    /// discrete uniform distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (floor(x) - min + 1) / (max - min + 1)
    /// ```
    fn cdf(&self, x: f64) -> f64 {
        if x < self.min as f64 {
            0.0
        } else if x >= self.max as f64 {
            1.0
        } else {
            let lower = self.min as f64;
            let upper = self.max as f64;
            let ans = (x.floor() - lower + 1.0) / (upper - lower + 1.0);
            if ans > 1.0 {
                1.0
            } else {
                ans
            }
        }
    }
}

impl Min<i64> for RobustSoliton {
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

impl Max<i64> for RobustSoliton {
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

impl Mean<f64> for RobustSoliton {
    /// Returns the mean of the discrete uniform distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (min + max) / 2
    /// ```
    fn mean(&self) -> f64 {
        (self.min + self.max) as f64 / 2.0
    }
}

impl Variance<f64> for RobustSoliton {
    /// Returns the variance of the discrete uniform distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ((max - min + 1)^2 - 1) / 12
    /// ```
    fn variance(&self) -> f64 {
        let diff = (self.max - self.min) as f64;
        ((diff + 1.0) * (diff + 1.0) - 1.0) / 12.0
    }

    /// Returns the standard deviation of the discrete uniform distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// sqrt(((max - min + 1)^2 - 1) / 12)
    /// ```
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

impl Entropy<f64> for RobustSoliton {
    /// Returns the entropy of the discrete uniform distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln(max - min + 1)
    /// ```
    fn entropy(&self) -> f64 {
        let diff = (self.max - self.min) as f64;
        (diff + 1.0).ln()
    }
}

impl Skewness<f64> for RobustSoliton {
    /// Returns the skewness of the discrete uniform distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 0
    /// ```
    fn skewness(&self) -> f64 {
        0.0
    }
}

impl Median<f64> for RobustSoliton {
    /// Returns the median of the discrete uniform distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (max + min) / 2
    /// ```
    fn median(&self) -> f64 {
        (self.min + self.max) as f64 / 2.0
    }
}

impl Mode<i64> for RobustSoliton {
    /// Returns the mode for the discrete uniform distribution
    ///
    /// # Remarks
    ///
    /// Since every element has an equal probability, mode simply
    /// returns the middle element
    ///
    /// # Formula
    ///
    /// ```ignore
    /// N/A // (max + min) / 2 for the middle element
    /// ```
    fn mode(&self) -> i64 {
        ((self.min + self.max) as f64 / 2.0).floor() as i64
    }
}

impl Discrete<i64, f64> for RobustSoliton {
    /// Calculates the probability mass function for the discrete uniform
    /// distribution at `x`
    ///
    /// # Remarks
    ///
    /// Returns `0.0` if `x` is not in `[min, max]`
    ///
    /// # Formula
    ///
    /// ```ignore
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
    /// ```ignore
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

impl Soliton<i64, f64> for RobustSoliton {
    /// Calculates the ideal soliton for the
    /// discrete uniform distribution at `x`
    ///
    /// # Remarks
    ///
    /// Returns `0.0` if `x` is not in `[min, max]`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ```
    fn soliton(&self, x: i64) -> f64 {
        if x > 1 && x < self.max {
            let ideal_sol = {
                if x > 1 && x <= self.max {
                    1.0 / ((x as f64) * (x as f64 - 1.0))
                } else if x == 1 {
                    1.0 / self.max as f64
                } else {
                    // Point must be in range (0, limit]
                    0.0
                }
            };
            (ideal_sol + self.additive_probability(x))/(self.normalization_factor())
        } else if x == 1 {
            1.0
        } else {
            // Point must be in range (0, limit]
            0.0
        }
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use std::fmt::Debug;
    use std::f64;
    use crate::statistics::*;
    use crate::distribution::{Univariate, Discrete, Soliton, RobustSoliton};

    fn try_create(max: i64, heuristic: bool, ripple: f64, fail_probability: f64) -> RobustSoliton {
        let n = RobustSoliton::new(max, heuristic, ripple, fail_probability);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(max: i64, heuristic: bool, ripple: f64, fail_probability: f64) {
        let n = try_create(max, heuristic, ripple, fail_probability);
        assert_eq!(1, n.min());
        assert_eq!(max, n.max());
    }

    fn bad_create_case(max: i64, heuristic: bool, ripple: f64, fail_probability: f64) {
        let n = RobustSoliton::new(max, heuristic, ripple, fail_probability);
        assert!(n.is_err());
    }

    fn get_value<T, F>(max: i64, heuristic: bool, ripple: f64, fail_probability: f64, eval: F) -> T
        where T: PartialEq + Debug,
              F: Fn(RobustSoliton) -> T
    {
        let n = try_create(max, heuristic, ripple, fail_probability);
        eval(n)
    }

    fn test_case<T, F>(max: i64, heuristic: bool, ripple: f64, fail_probability: f64, expected: T, eval: F)
        where T: PartialEq + Debug,
              F: Fn(RobustSoliton) -> T
    {
        let x = get_value(max, heuristic, ripple, fail_probability, eval);
        assert_eq!(expected, x);
    }

    #[test]
    fn test_create() {
        create_case(5, true, 0.1, 0.1);
        create_case(10, true, 0.1, 0.1);
        create_case(25, true, 0.1, 0.1);
        create_case(50, true, 0.1, 0.1);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(-1, true, 0.1, 0.1);
        bad_create_case(5, true, 0.1, 0.1);
    }

    #[test]
    fn test_mean() {
        test_case(10, true, 0.1, 0.1, 0.0, |x| x.mean());
        test_case(10, true, 0.1, 0.1, 2.0, |x| x.mean());
        test_case(10, true, 0.1, 0.1, 15.0, |x| x.mean());
        test_case(10, true, 0.1, 0.1, 20.0, |x| x.mean());
    }

    #[test]
    fn test_variance() {
        test_case(10, true, 0.1, 0.1, 36.66666666666666666667, |x| x.variance());
        test_case(4, true, 0.1, 0.1, 2.0, |x| x.variance());
        test_case(20, true, 0.1, 0.1, 10.0, |x| x.variance());
        test_case(20, true, 0.1, 0.1, 0.0, |x| x.variance());
    }

    #[test]
    fn test_std_dev() {
        test_case(10, true, 0.1, 0.1, (36.66666666666666666667f64).sqrt(), |x| x.std_dev());
        test_case(4, true, 0.1, 0.1, (2.0f64).sqrt(), |x| x.std_dev());
        test_case(20, true, 0.1, 0.1, (10.0f64).sqrt(), |x| x.std_dev());
        test_case(20, true, 0.1, 0.1, 0.0, |x| x.std_dev());
    }

    #[test]
    fn test_entropy() {
        test_case(10, true, 0.1, 0.1, 3.0445224377234229965005979803657054342845752874046093, |x| x.entropy());
        test_case(4, true, 0.1, 0.1, 1.6094379124341003746007593332261876395256013542685181, |x| x.entropy());
        test_case(20, true, 0.1, 0.1, 2.3978952727983705440619435779651292998217068539374197, |x| x.entropy());
        test_case(20, true, 0.1, 0.1, 0.0, |x| x.entropy());
    }

    #[test]
    fn test_skewness() {
        test_case(10, true, 0.1, 0.1, 0.0, |x| x.skewness());
        test_case(4, true, 0.1, 0.1, 0.0, |x| x.skewness());
        test_case(20, true, 0.1, 0.1, 0.0, |x| x.skewness());
        test_case(20, true, 0.1, 0.1, 0.0, |x| x.skewness());
    }

    #[test]
    fn test_median() {
        test_case(10, true, 0.1, 0.1, 0.0, |x| x.median());
        test_case(4, true, 0.1, 0.1, 2.0, |x| x.median());
        test_case(20, true, 0.1, 0.1, 15.0, |x| x.median());
        test_case(20, true, 0.1, 0.1, 20.0, |x| x.median());
    }

    #[test]
    fn test_mode() {
        test_case(10, true, 0.1, 0.1, 0, |x| x.mode());
        test_case(4, true, 0.1, 0.1, 2, |x| x.mode());
        test_case(20, true, 0.1, 0.1, 15, |x| x.mode());
        test_case(20, true, 0.1, 0.1, 20, |x| x.mode());
    }

    #[test]
    fn test_pmf() {
        test_case(10, true, 0.1, 0.1, 0.04761904761904761904762, |x| x.pmf(5));
        test_case(10, true, 0.1, 0.1, 0.04761904761904761904762, |x| x.pmf(1));
        test_case(10, true, 0.1, 0.1, 0.04761904761904761904762, |x| x.pmf(10));
        test_case(10, true, 0.1, 0.1, 0.0, |x| x.pmf(0));
    }

    #[test]
    fn test_ln_pmf() {
        test_case(10, true, 0.1, 0.1, -3.0445224377234229965005979803657054342845752874046093, |x| x.ln_pmf(1));
        test_case(10, true, 0.1, 0.1, -3.0445224377234229965005979803657054342845752874046093, |x| x.ln_pmf(10));
        test_case(10, true, 0.1, 0.1, f64::NEG_INFINITY, |x| x.ln_pmf(0));
        test_case(10, true, 0.1, 0.1,  0.0, |x| x.ln_pmf(10));
    }

    #[test]
    fn test_cdf() {
        test_case(10, true, 0.1, 0.1, 0.5714285714285714285714, |x| x.cdf(1.0));
        test_case(10, true, 0.1, 0.1, 1.0, |x| x.cdf(10.0));
    }

    #[test]
    fn test_cdf_lower_bound() {
        test_case(3, true, 0.1, 0.1, 0.0, |x| x.cdf(1.0));
    }

    #[test]
    fn test_cdf_upper_bound() {
        test_case(3, true, 0.1, 0.1, 1.0, |x| x.cdf(5.0));
    }

    #[test]
    fn test_solition() {
        test_case(10, true, 0.1, 0.1, 0.0, |x| x.soliton(-1));
        test_case(10, true, 0.1, 0.1, 0.0, |x| x.soliton(0));
        test_case(10, true, 0.1, 0.1, 1.0, |x| x.soliton(10));
    }
}
