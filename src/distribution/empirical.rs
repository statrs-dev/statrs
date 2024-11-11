use crate::distribution::ContinuousCDF;
use crate::statistics::*;
use non_nan::NonNan;
use std::collections::btree_map::{BTreeMap, Entry};
use std::convert::Infallible;
use std::ops::Bound;

mod non_nan {
    use core::cmp::Ordering;

    #[derive(Clone, Copy, PartialEq, Debug)]
    pub struct NonNan<T>(T);

    impl<T: Copy> NonNan<T> {
        pub fn get(self) -> T {
            self.0
        }
    }

    impl NonNan<f64> {
        #[inline]
        pub fn new(x: f64) -> Option<Self> {
            if x.is_nan() {
                None
            } else {
                Some(Self(x))
            }
        }
    }

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
    // keys are data points, values are number of data points with equal value
    data: BTreeMap<NonNan<f64>, u64>,

    // The following fields are only logically valid if !data.is_empty():
    /// Total amount of data points (== sum of all _values_ inside self.data).
    /// Must be 0 iff data.is_empty()
    sum: u64,
    mean: f64,
    var: f64,
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
    pub fn new() -> Result<Empirical, Infallible> {
        Ok(Empirical {
            data: BTreeMap::new(),
            sum: 0,
            mean: 0.0,
            var: 0.0,
        })
    }

    pub fn add(&mut self, data_point: f64) {
        let map_key = match NonNan::new(data_point) {
            Some(valid) => valid,
            None => return,
        };

        self.sum += 1;
        let sum = self.sum as f64;
        self.var += (sum - 1.) * (data_point - self.mean) * (data_point - self.mean) / sum;
        self.mean += (data_point - self.mean) / sum;

        self.data
            .entry(map_key)
            .and_modify(|c| *c += 1)
            .or_insert(1);
    }

    pub fn remove(&mut self, data_point: f64) {
        let map_key = match NonNan::new(data_point) {
            Some(valid) => valid,
            None => return,
        };

        let mut entry = match self.data.entry(map_key) {
            Entry::Occupied(entry) => entry,
            Entry::Vacant(_) => return, // no entry found
        };

        if *entry.get() == 1 {
            entry.remove();
            if self.data.is_empty() {
                // logically, this should not need special handling.
                // FP math can result in mean or var being != 0.0 though.
                self.sum = 0;
                self.mean = 0.0;
                self.var = 0.0;
                return;
            }
        } else {
            *entry.get_mut() -= 1;
        }

        // reset mean and var
        let sum = self.sum as f64;
        self.mean = (sum * self.mean - data_point) / (sum - 1.);
        self.var -= (sum - 1.) * (data_point - self.mean) * (data_point - self.mean) / sum;
        self.sum -= 1;
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
            .flat_map(|(x, &count)| std::iter::repeat(x.get()).take(count as usize));

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
        self.data.keys().rev().map(|key| key.get()).next().unwrap()
    }
}

/// Panics if number of samples is zero
impl Min<f64> for Empirical {
    fn min(&self) -> f64 {
        self.data.keys().map(|key| key.get()).next().unwrap()
    }
}

impl Distribution<f64> for Empirical {
    fn mean(&self) -> Option<f64> {
        if self.data.is_empty() {
            None
        } else {
            Some(self.mean)
        }
    }

    fn variance(&self) -> Option<f64> {
        if self.data.is_empty() {
            None
        } else {
            Some(self.var / (self.sum as f64 - 1.))
        }
    }
}

impl ContinuousCDF<f64, f64> for Empirical {
    fn cdf(&self, x: f64) -> f64 {
        let start = Bound::Unbounded;
        let end = Bound::Included(NonNan::new(x).expect("x must not be NaN"));

        let sum: u64 = self.data.range((start, end)).map(|(_, v)| v).sum();
        sum as f64 / self.sum as f64
    }

    fn sf(&self, x: f64) -> f64 {
        let start = Bound::Excluded(NonNan::new(x).expect("x must not be NaN"));
        let end = Bound::Unbounded;

        let sum: u64 = self.data.range((start, end)).map(|(_, v)| v).sum();
        sum as f64 / self.sum as f64
    }

    fn inverse_cdf(&self, p: f64) -> f64 {
        self.__inverse_cdf(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_nan() {
        let mut empirical = Empirical::new().unwrap();

        // should not panic
        empirical.add(f64::NAN);
    }

    #[test]
    fn test_remove_nan() {
        let mut empirical = Empirical::new().unwrap();

        empirical.add(5.2);
        // should not panic
        empirical.remove(f64::NAN);
    }

    #[test]
    fn test_remove_nonexisting() {
        let mut empirical = Empirical::new().unwrap();

        empirical.add(5.2);
        // should not panic
        empirical.remove(10.0);
    }

    #[test]
    fn test_remove_all() {
        let mut empirical = Empirical::new().unwrap();

        empirical.add(17.123);
        empirical.add(-10.0);
        empirical.add(0.0);
        empirical.remove(-10.0);
        empirical.remove(17.123);
        empirical.remove(0.0);

        assert!(empirical.mean().is_none());
        assert!(empirical.variance().is_none());
    }

    #[test]
    fn test_mean() {
        fn test_mean_for_samples(expected_mean: f64, samples: Vec<f64>) {
            let dist = Empirical::from_iter(samples);
            assert_relative_eq!(dist.mean().unwrap(), expected_mean);
        }

        let dist = Empirical::from_iter(vec![]);
        assert!(dist.mean().is_none());

        test_mean_for_samples(4.0, vec![4.0; 100]);
        test_mean_for_samples(-0.2, vec![-0.2; 100]);
        test_mean_for_samples(28.5, vec![21.3, 38.4, 12.7, 41.6]);
    }

    #[test]
    fn test_var() {
        fn test_var_for_samples(expected_var: f64, samples: Vec<f64>) {
            let dist = Empirical::from_iter(samples);
            assert_relative_eq!(dist.variance().unwrap(), expected_var);
        }

        let dist = Empirical::from_iter(vec![]);
        assert!(dist.variance().is_none());

        test_var_for_samples(0.0, vec![4.0; 100]);
        test_var_for_samples(0.0, vec![-0.2; 100]);
        test_var_for_samples(190.36666666666667, vec![21.3, 38.4, 12.7, 41.6]);
    }

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
