use crate::distribution::{Continuous, ContinuousCDF, Uniform};
use crate::statistics::*;
use crate::{Result, StatsError};
use ::num_traits::float::Float;
use core::cmp::Ordering;
use rand::Rng;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
struct NonNAN<T>(T);

impl<T: PartialEq> Eq for NonNAN<T> {}

impl<T: PartialOrd> PartialOrd for NonNAN<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<T: PartialOrd> Ord for NonNAN<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// Implements the [Empirical
/// Distribution](https://en.wikipedia.org/wiki/Empirical_distribution_function)
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Continuous, Empirical};
/// use statrs::statistics::Distribution;
///
/// let samples = vec![0.0, 5.0, 10.0];
///
/// let empirical = Empirical::from_vec(samples);
/// assert_eq!(empirical.mean().unwrap(), 5.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Empirical {
    sum: f64,
    mean_and_var: Option<(f64, f64)>,
    // keys are data points, values are number of data points with equal value
    data: BTreeMap<NonNAN<f64>, u64>,
}

impl Empirical {
    /// Constructs a new discrete uniform distribution with a minimum value
    /// of `min` and a maximum value of `max`.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Empirical;
    ///
    /// let mut result = Empirical::new();
    /// assert!(result.is_ok());
    ///
    /// ```
    pub fn new() -> Result<Empirical> {
        Ok(Empirical {
            sum: 0.,
            mean_and_var: None,
            data: BTreeMap::new(),
        })
    }
    pub fn from_vec(src: Vec<f64>) -> Empirical {
        let mut empirical = Empirical::new().unwrap();
        for elt in src.into_iter() {
            empirical.add(elt);
        }
        empirical
    }
    pub fn add(&mut self, data_point: f64) {
        if !data_point.is_nan() {
            self.sum += 1.;
            match self.mean_and_var {
                Some((mean, var)) => {
                    let sum = self.sum;
                    let var = var + (sum - 1.) * (data_point - mean) * (data_point - mean) / sum;
                    let mean = mean + (data_point - mean) / sum;
                    self.mean_and_var = Some((mean, var));
                }
                None => {
                    self.mean_and_var = Some((data_point, 0.));
                }
            }
            *self.data.entry(NonNAN(data_point)).or_insert(0) += 1;
        }
    }
    pub fn remove(&mut self, data_point: f64) {
        if !data_point.is_nan() {
            if let (Some(val), Some((mean, var))) =
                (self.data.remove(&NonNAN(data_point)), self.mean_and_var)
            {
                if val == 1 && self.data.is_empty() {
                    self.mean_and_var = None;
                    self.sum = 0.;
                    return;
                };
                // reset mean and var
                let mean = (self.sum * mean - data_point) / (self.sum - 1.);
                let var =
                    var - (self.sum - 1.) * (data_point - mean) * (data_point - mean) / self.sum;
                self.sum -= 1.;
                if val != 1 {
                    self.data.insert(NonNAN(data_point), val - 1);
                };
                self.mean_and_var = Some((mean, var));
            }
        }
    }
    // Due to issues with rounding and floating-point accuracy the default
    // implementation may be ill-behaved.
    // Specialized inverse cdfs should be used whenever possible.
    // Performs a binary search on the domain of `cdf` to obtain an approximation
    // of `F^-1(p) := inf { x | F(x) >= p }`. Needless to say, performance may
    // may be lacking.
    // This function is identical to the default method implementation in the
    // `ContinuousCDF` trait and is used to implement the rand trait `Distribution`.
    fn __inverse_cdf(&self, p: f64) -> f64 {
        if p == 0.0 {
            return self.min();
        };
        if p == 1.0 {
            return self.max();
        };
        let mut high = 2.0;
        let mut low = -high;
        while self.cdf(low) > p {
            low = low + low;
        }
        while self.cdf(high) < p {
            high = high + high;
        }
        let mut i = 16;
        while i != 0 {
            let mid = (high + low) / 2.0;
            if self.cdf(mid) >= p {
                high = mid;
            } else {
                low = mid;
            }
            i -= 1;
        }
        (high + low) / 2.0
    }
}

impl ::rand::distributions::Distribution<f64> for Empirical {
    fn sample<R: ?Sized + Rng>(&self, rng: &mut R) -> f64 {
        let uniform = Uniform::new(0.0, 1.0).unwrap();
        self.__inverse_cdf(uniform.sample(rng))
    }
}

/// Panics if number of samples is zero
impl Max<f64> for Empirical {
    fn max(&self) -> f64 {
        self.data.keys().rev().map(|key| key.0) .next().unwrap()
    }
}

/// Panics if number of samples is zero
impl Min<f64> for Empirical {
    fn min(&self) -> f64 {
        self.data.keys().map(|key| key.0).next().unwrap()
    }
}

impl Distribution<f64> for Empirical {
    fn mean(&self) -> Option<f64> {
        self.mean_and_var.map(|(mean, _)| mean)
    }
    fn variance(&self) -> Option<f64> {
        self.mean_and_var.map(|(_, var)| var / (self.sum - 1.))
    }
}

impl ContinuousCDF<f64, f64> for Empirical {
    fn cdf(&self, x: f64) -> f64 {
        let mut sum = 0;
        for (keys, values) in &self.data {
            if keys.0 > x {
                return sum as f64 / self.sum;
            }
            sum += values;
        }
        sum as f64 / self.sum
    }

    fn sf(&self, x: f64) -> f64 {
        let mut sum = 0;
        for (keys, values) in self.data.iter().rev() {
            if keys.0 <= x {
                return sum as f64 / self.sum;
            }
            sum += values;
        }
        sum as f64 / self.sum
    }
}

#[cfg(all(test, feature = "nightly"))]
mod tests {
    use super::*;
    #[test]
    fn test_cdf() {
        let samples = vec![5.0, 10.0];
        let mut empirical = Empirical::from_vec(samples);
        assert_eq!(empirical.cdf(0.0), 0.0);
        assert_eq!(empirical.cdf(5.0), 0.5);
        assert_eq!(empirical.cdf(5.5), 0.5);
        assert_eq!(empirical.cdf(6.0), 0.5);
        assert_eq!(empirical.cdf(10.0), 1.0);
        assert_eq!(empirical.min(), 5.0);
        assert_eq!(empirical.max(), 10.0);
        empirical.add(2.0);
        empirical.add(2.0);
        assert_eq!(empirical.cdf(0.0), 0.0);
        assert_eq!(empirical.cdf(5.0), 0.75);
        assert_eq!(empirical.cdf(5.5), 0.75);
        assert_eq!(empirical.cdf(6.0), 0.75);
        assert_eq!(empirical.cdf(10.0), 1.0);
        assert_eq!(empirical.min(), 2.0);
        assert_eq!(empirical.max(), 10.0);
        let unchanged = empirical.clone();
        empirical.add(2.0);
        empirical.remove(2.0);
        // because of rounding errors, this doesn't hold in general
        // due to the mean and variance being calculated in a streaming way
        assert_eq!(unchanged, empirical);
    }

    #[test]
    fn test_sf() {
        let samples = vec![5.0, 10.0];
        let mut empirical = Empirical::from_vec(samples);
        assert_eq!(empirical.sf(0.0), 1.0);
        assert_eq!(empirical.sf(5.0), 0.5);
        assert_eq!(empirical.sf(5.5), 0.5);
        assert_eq!(empirical.sf(6.0), 0.5);
        assert_eq!(empirical.sf(10.0), 0.0);
        assert_eq!(empirical.min(), 5.0);
        assert_eq!(empirical.max(), 10.0);
        empirical.add(2.0);
        empirical.add(2.0);
        assert_eq!(empirical.sf(0.0), 1.0);
        assert_eq!(empirical.sf(5.0), 0.25);
        assert_eq!(empirical.sf(5.5), 0.25);
        assert_eq!(empirical.sf(6.0), 0.25);
        assert_eq!(empirical.sf(10.0), 0.0);
        assert_eq!(empirical.min(), 2.0);
        assert_eq!(empirical.max(), 10.0);
        let unchanged = empirical.clone();
        empirical.add(2.0);
        empirical.remove(2.0);
         //because of rounding errors, this doesn't hold in general
         //due to the mean and variance being calculated in a streaming way
        assert_eq!(unchanged, empirical);
    }
}
