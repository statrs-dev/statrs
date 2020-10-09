use crate::distribution::Soliton;
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
pub struct IdealSoliton {
    min: i64,
    max: i64,
}

impl IdealSoliton {
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
    pub fn new(max: i64) -> Result<IdealSoliton> {
        if max < 1 {
            Err(StatsError::BadParams)
        } else {
            Ok(IdealSoliton { min: 1, max })
        }
    }
}

impl Distribution<f64> for IdealSoliton {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> f64 {
        r.gen_range(0, 1) as f64
    }
}

impl Min<i64> for IdealSoliton {
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

impl Max<i64> for IdealSoliton {
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

impl Mean<f64> for IdealSoliton {
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

impl Variance<f64> for IdealSoliton {
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

impl Soliton<i64, f64> for IdealSoliton {
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
    /// p(1) = 1 / (max)
    /// p(x) = 1/(x(x-1))
    /// ```
    fn soliton(&self, x: i64) -> f64 {
        if x > 1 && x < self.max {
            1.0 / ((x as f64) * (x as f64 - 1.0))
        } else if x == 1 {
            1.0 / self.max as f64
        } else {
            // Point must be in range (0, limit]
            0.0
        }
    }

    fn normalization_factor(&self) -> f64 {
        0.0
    }

    fn additive_probability(&self, _x: i64) -> f64 {
        0.0
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use std::fmt::Debug;
    use std::f64;
    use crate::statistics::*;
    use crate::distribution::IdealSoliton;

    fn try_create(max: i64) -> IdealSoliton {
        let n = IdealSoliton::new(max);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(max: i64) {
        let n = try_create(max);
        assert_eq!(1, n.min());
        assert_eq!(max, n.max());
    }

    fn bad_create_case(max: i64) {
        let n = IdealSoliton::new(max);
        assert!(n.is_err());
    }

    fn get_value<T, F>(max: i64, eval: F) -> T
        where T: PartialEq + Debug,
              F: Fn(IdealSoliton) -> T
    {
        let n = try_create(max);
        eval(n)
    }

    fn test_case<T, F>(max: i64, expected: T, eval: F)
        where T: PartialEq + Debug,
              F: Fn(IdealSoliton) -> T
    {
        let x = get_value(max, eval);
        assert_eq!(expected, x);
    }

    fn test_case_greater<T, F>(max: i64, expected: T, eval: F)
        where T: PartialEq + Debug + Into<f64>,
              F: Fn(IdealSoliton) -> T
    {
        let sol = get_value(max, eval);
        let a: f64 = sol.into();
        let b = expected.into();
        assert!(a > b, "{} greater than {}", a, b);
    }

    #[test]
    fn test_create() {
        create_case(10);
        create_case(4);
        create_case(20);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(-2);
        bad_create_case(0);
    }

    #[test]
    fn test_mean() {
        test_case_greater(10, 0.9, |x| x.mean());
    }

    #[test]
    fn test_variance() {
        test_case(10, 8.25, |x| x.variance());
    }

    #[test]
    fn test_std_dev() {
        test_case(10, (8.25f64).sqrt(), |x| x.std_dev());
    }
}
