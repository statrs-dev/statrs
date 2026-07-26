use crate::distribution::ContinuousCDF;
use crate::statistics::*;
use alloc::collections::btree_map::{BTreeMap, Entry};
use core::convert::Infallible;
use core::ops::Bound;
use non_nan::NonNan;

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
            if x.is_nan() { None } else { Some(Self(x)) }
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

impl core::fmt::Display for Empirical {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut enumerated_values = self
            .data
            .iter()
            .flat_map(|(x, &count)| core::iter::repeat_n(x.get(), count as usize));

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
impl ::rand::distr::Distribution<f64> for Empirical {
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

/// Panics if number of samples is zero
impl Median<f64> for Empirical {
    /// Returns the sample median of the observed data.
    ///
    /// # Remarks
    ///
    /// For an odd number of observations this is the middle order statistic. For
    /// an even number it is the mean of the two middle ones, which is the usual
    /// convention (and the one NumPy and R's `median` use by default), chosen
    /// because it is the only value equidistant from both. Note the result then
    /// need not be a value that was actually observed.
    ///
    /// Repeated observations count with their multiplicity, so the median of
    /// `[1, 1, 1, 2]` is `1`, not `1.5`.
    fn median(&self) -> f64 {
        assert!(
            !self.data.is_empty(),
            "Cannot compute the median of zero samples"
        );

        // The lower middle observation is at 0-based rank (n - 1) / 2; for even
        // n the upper one follows it. Walking the BTreeMap visits keys in
        // ascending order, so accumulating counts gives order statistics.
        let n = self.sum;
        let lower_rank = (n - 1) / 2;
        let need_two = n % 2 == 0;

        let mut seen = 0;
        let mut lower = None;
        for (key, &count) in self.data.iter() {
            seen += count;
            if lower.is_none() && seen > lower_rank {
                if !need_two {
                    return key.get();
                }
                lower = Some(key.get());
                // The next rank up may live in this same key when it has
                // multiplicity, in which case both middles are equal.
                if seen > lower_rank + 1 {
                    return key.get();
                }
            } else if let Some(lo) = lower {
                return 0.5 * (lo + key.get());
            }
        }

        unreachable!("the median rank is always within a non-empty data set")
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

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::prec;

    /// Reference implementation: sort and take the middle, or average the two
    /// middles. Deliberately the naive O(n log n) version, so it shares no logic
    /// with the BTreeMap walk it checks.
    fn median_by_sorting(data: &[f64]) -> f64 {
        let mut v = data.to_vec();
        v.sort_by(|a, b| a.total_cmp(b));
        let n = v.len();
        if n % 2 == 1 {
            v[n / 2]
        } else {
            0.5 * (v[n / 2 - 1] + v[n / 2])
        }
    }

    #[test]
    fn test_median() {
        // odd count -> the middle observation
        let e: Empirical = [3.0, 1.0, 2.0].into_iter().collect();
        assert_eq!(e.median(), 2.0);

        // even count -> mean of the two middles, which was never observed
        let e: Empirical = [1.0, 2.0, 3.0, 4.0].into_iter().collect();
        assert_eq!(e.median(), 2.5);

        // multiplicity counts: the two middles are both 1.0 here
        let e: Empirical = [1.0, 1.0, 1.0, 2.0].into_iter().collect();
        assert_eq!(e.median(), 1.0);

        // a single repeated value
        let e: Empirical = [7.0; 5].into_iter().collect();
        assert_eq!(e.median(), 7.0);

        // one observation
        let e: Empirical = [42.0].into_iter().collect();
        assert_eq!(e.median(), 42.0);

        // two observations straddle
        let e: Empirical = [1.0, 4.0].into_iter().collect();
        assert_eq!(e.median(), 2.5);

        // negatives and duplicates together
        let e: Empirical = [-5.0, -1.0, -1.0, 0.0, 3.0].into_iter().collect();
        assert_eq!(e.median(), -1.0);
    }

    /// Agreement with the sorting reference across many shapes of data,
    /// including heavy duplication, which is where the multiplicity handling in
    /// the BTreeMap walk could go wrong.
    #[test]
    fn test_median_matches_sorting_reference() {
        let cases: &[&[f64]] = &[
            &[1.0],
            &[1.0, 2.0],
            &[2.0, 1.0],
            &[1.0, 2.0, 3.0],
            &[1.0, 1.0, 2.0, 2.0],
            &[1.0, 1.0, 1.0, 2.0, 2.0],
            &[1.0, 2.0, 2.0, 2.0, 3.0],
            &[5.0, 5.0, 5.0, 5.0],
            &[1.0, 1.0, 1.0, 1.0, 9.0],
            &[9.0, 1.0, 1.0, 1.0, 1.0],
            &[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0],
            &[0.0, 0.0, 0.0, 1.0],
            &[0.0, 1.0, 1.0, 1.0],
            &[1e300, -1e300],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ];
        for case in cases {
            let e: Empirical = case.iter().copied().collect();
            let want = median_by_sorting(case);
            let got = e.median();
            assert_eq!(got, want, "median of {case:?} was {got}, expected {want}");
        }

        // Longer runs built deterministically, with the modulus chosen to force
        // many repeats.
        for len in 1..40usize {
            for modulus in [1u64, 2, 3, 7] {
                let data: Vec<f64> = (0..len as u64)
                    .map(|i| ((i * 37 + 11) % modulus.max(1)) as f64)
                    .collect();
                let e: Empirical = data.iter().copied().collect();
                assert_eq!(
                    e.median(),
                    median_by_sorting(&data),
                    "len {len}, modulus {modulus}, data {data:?}"
                );
            }
        }
    }

    /// The median must fall between the extremes, and coincide with them when
    /// the data is constant.
    #[test]
    fn test_median_within_bounds() {
        let e: Empirical = [1.0, 5.0, 2.0, 9.0, 3.0].into_iter().collect();
        assert!(e.median() >= e.min() && e.median() <= e.max());

        let e: Empirical = [4.0; 3].into_iter().collect();
        assert_eq!(e.median(), e.min());
        assert_eq!(e.median(), e.max());
    }

    #[test]
    #[should_panic(expected = "Cannot compute the median of zero samples")]
    fn test_median_of_empty_panics() {
        Empirical::new().unwrap().median();
    }

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
            prec::assert_relative_eq!(dist.mean().unwrap(), expected_mean);
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
            prec::assert_relative_eq!(dist.variance().unwrap(), expected_var);
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
