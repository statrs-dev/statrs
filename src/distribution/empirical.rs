use crate::distribution::ContinuousCDF;
use crate::statistics::*;
use core::cmp::Ordering;
use std::collections::BTreeMap;

#[derive(Clone, PartialEq, Debug)]
struct NonNan<T>(T);

impl<T: PartialEq> Eq for NonNan<T> {}

impl<T: PartialOrd> PartialOrd for NonNan<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: PartialOrd> Ord for NonNan<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap()
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
/// let empirical = Empirical::from_iter(samples);
/// assert_eq!(empirical.mean().unwrap(), 5.0);
/// ```
#[derive(Clone, PartialEq, Debug)]
pub struct Empirical {
    sum: f64,
    mean_and_var: Option<(f64, f64)>,
    // keys are data points, values are number of data points with equal value
    data: BTreeMap<NonNan<f64>, u64>,
}

impl Empirical {
    /// Constructs a new discrete uniform distribution with a minimum value
    /// of `min` and a maximum value of `max`.
    ///
    /// Note that this will always succeed and never return the [`Err`][Result::Err] variant.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Empirical;
    ///
    /// let mut result = Empirical::new();
    /// assert!(result.is_ok());
    /// ```
    #[allow(clippy::result_unit_err)]
    pub fn new() -> Result<Empirical, ()> {
        Ok(Empirical {
            sum: 0.,
            mean_and_var: None,
            data: BTreeMap::new(),
        })
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
            *self.data.entry(NonNan(data_point)).or_insert(0) += 1;
        }
    }

    pub fn remove(&mut self, data_point: f64) {
        if !data_point.is_nan() {
            if let (Some(val), Some((mean, var))) =
                (self.data.remove(&NonNan(data_point)), self.mean_and_var)
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
                    self.data.insert(NonNan(data_point), val - 1);
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

impl std::fmt::Display for Empirical {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut enumerated_values = self
            .data
            .iter()
            .flat_map(|(&NonNan(x), &count)| std::iter::repeat(x).take(count as usize));

        if let Some(x) = enumerated_values.next() {
            write!(f, "Empirical([{x:.3e}")?;
        } else {
            return write!(f, "Empirical(∅)");
        }

        for val in enumerated_values.by_ref().take(4) {
            write!(f, ", {val:.3e}")?;
        }
        if enumerated_values.next().is_some() {
            write!(f, ", ...")?;
        }
        write!(f, "])")
    }
}

impl FromIterator<f64> for Empirical {
    fn from_iter<T: IntoIterator<Item = f64>>(iter: T) -> Self {
        let mut empirical = Self::new().unwrap();
        for elt in iter {
            empirical.add(elt);
        }
        empirical
    }
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl ::rand::distributions::Distribution<f64> for Empirical {
    fn sample<R: ::rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use crate::distribution::Uniform;

        let uniform = Uniform::new(0.0, 1.0).unwrap();
        self.__inverse_cdf(uniform.sample(rng))
    }
}

/// Panics if number of samples is zero
impl Max<f64> for Empirical {
    fn max(&self) -> f64 {
        self.data.keys().rev().map(|key| key.0).next().unwrap()
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

    fn inverse_cdf(&self, p: f64) -> f64 {
        self.__inverse_cdf(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_cdf() {
        let samples = vec![5.0, 10.0];
        let mut empirical = Empirical::from_iter(samples);
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
        let mut empirical = Empirical::from_iter(samples);
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
        // because of rounding errors, this doesn't hold in general
        // due to the mean and variance being calculated in a streaming way
        assert_eq!(unchanged, empirical);
    }

    #[test]
    fn test_display() {
        let mut e = Empirical::new().unwrap();
        assert_eq!(e.to_string(), "Empirical(∅)");
        e.add(1.0);
        assert_eq!(e.to_string(), "Empirical([1.000e0])");
        e.add(1.0);
        assert_eq!(e.to_string(), "Empirical([1.000e0, 1.000e0])");
        e.add(2.0);
        assert_eq!(e.to_string(), "Empirical([1.000e0, 1.000e0, 2.000e0])");
        e.add(2.0);
        assert_eq!(
            e.to_string(),
            "Empirical([1.000e0, 1.000e0, 2.000e0, 2.000e0])"
        );
        e.add(5.0);
        assert_eq!(
            e.to_string(),
            "Empirical([1.000e0, 1.000e0, 2.000e0, 2.000e0, 5.000e0])"
        );
        e.add(5.0);
        assert_eq!(
            e.to_string(),
            "Empirical([1.000e0, 1.000e0, 2.000e0, 2.000e0, 5.000e0, ...])"
        );
    }
}
