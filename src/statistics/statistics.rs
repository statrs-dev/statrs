/// Enumeration of possible tie-breaking strategies
/// when computing ranks
#[derive(Debug, Copy, Clone)]
pub enum RankTieBreaker {
    /// Replaces ties with their mean
    Average,
    /// Replace ties with their minimum
    Min,
    /// Replace ties with their maximum
    Max,
    /// Permutation with increasing values at each index of ties
    First,
}

/// The `Statistics` trait provides a host of statistical utilities for
/// analyzing
/// data sets
pub trait Statistics<T> {
    /// Returns the minimum value in the data
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64;
    /// use statrs::statistics::Statistics;
    ///
    /// let x = &[];
    /// assert!(Statistics::min(x).is_nan());
    ///
    /// let y = &[0.0, f64::NAN, 3.0, -2.0];
    /// assert!(Statistics::min(y).is_nan());
    ///
    /// let z = &[0.0, 3.0, -2.0];
    /// assert_eq!(Statistics::min(z), -2.0);
    /// ```
    fn min(self) -> T;

    /// Returns the maximum value in the data
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64;
    /// use statrs::statistics::Statistics;
    ///
    /// let x = &[];
    /// assert!(Statistics::max(x).is_nan());
    ///
    /// let y = &[0.0, f64::NAN, 3.0, -2.0];
    /// assert!(Statistics::max(y).is_nan());
    ///
    /// let z = &[0.0, 3.0, -2.0];
    /// assert_eq!(Statistics::max(z), 3.0);
    /// ```
    fn max(self) -> T;

    /// Returns the minimum absolute value in the data
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64;
    /// use statrs::statistics::Statistics;
    ///
    /// let x = &[];
    /// assert!(x.abs_min().is_nan());
    ///
    /// let y = &[0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.abs_min().is_nan());
    ///
    /// let z = &[0.0, 3.0, -2.0];
    /// assert_eq!(z.abs_min(), 0.0);
    /// ```
    fn abs_min(self) -> T;

    /// Returns the maximum absolute value in the data
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64;
    /// use statrs::statistics::Statistics;
    ///
    /// let x = &[];
    /// assert!(x.abs_max().is_nan());
    ///
    /// let y = &[0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.abs_max().is_nan());
    ///
    /// let z = &[0.0, 3.0, -2.0, -8.0];
    /// assert_eq!(z.abs_max(), 8.0);
    /// ```
    fn abs_max(self) -> T;

    /// Evaluates the sample mean, an estimate of the population
    /// mean.
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`
    ///
    /// # Examples
    ///
    /// ```
    /// #[macro_use]
    /// extern crate statrs;
    ///
    /// use std::f64;
    /// use statrs::statistics::Statistics;
    ///
    /// # fn main() {
    /// let x = &[];
    /// assert!(x.mean().is_nan());
    ///
    /// let y = &[0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.mean().is_nan());
    ///
    /// let z = &[0.0, 3.0, -2.0];
    /// assert_almost_eq!(z.mean(), 1.0 / 3.0, 1e-15);
    /// # }
    /// ```
    fn mean(self) -> T;

    /// Evaluates the geometric mean of the data
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`.
    /// Returns `f64::NAN` if an entry is less than `0`. Returns `0`
    /// if no entry is less than `0` but there are entries equal to `0`.
    ///
    /// # Examples
    ///
    /// ```
    /// #[macro_use]
    /// extern crate statrs;
    ///
    /// use std::f64;
    /// use statrs::statistics::Statistics;
    ///
    /// # fn main() {
    /// let x = &[];
    /// assert!(x.geometric_mean().is_nan());
    ///
    /// let y = &[0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.geometric_mean().is_nan());
    ///
    /// let mut z = &[0.0, 3.0, -2.0];
    /// assert!(z.geometric_mean().is_nan());
    ///
    /// z = &[0.0, 3.0, 2.0];
    /// assert_eq!(z.geometric_mean(), 0.0);
    ///
    /// z = &[1.0, 2.0, 3.0];
    /// // test value from online calculator, could be more accurate
    /// assert_almost_eq!(z.geometric_mean(), 1.81712, 1e-5);
    /// # }
    /// ```
    fn geometric_mean(self) -> T;

    /// Evaluates the harmonic mean of the data
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`, or if
    /// any value
    /// in data is less than `0`. Returns `0` if there are no values less than
    /// `0` but
    /// there exists values equal to `0`.
    ///
    /// # Examples
    ///
    /// ```
    /// #[macro_use]
    /// extern crate statrs;
    ///
    /// use std::f64;
    /// use statrs::statistics::Statistics;
    ///
    /// # fn main() {
    /// let x = &[];
    /// assert!(x.harmonic_mean().is_nan());
    ///
    /// let y = &[0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.harmonic_mean().is_nan());
    ///
    /// let mut z = &[0.0, 3.0, -2.0];
    /// assert!(z.harmonic_mean().is_nan());
    ///
    /// z = &[0.0, 3.0, 2.0];
    /// assert_eq!(z.harmonic_mean(), 0.0);
    ///
    /// z = &[1.0, 2.0, 3.0];
    /// // test value from online calculator, could be more accurate
    /// assert_almost_eq!(z.harmonic_mean(), 1.63636, 1e-5);
    /// # }
    /// ```
    fn harmonic_mean(self) -> T;

    /// Estimates the unbiased population variance from the provided samples
    ///
    /// # Remarks
    ///
    /// On a dataset of size `N`, `N-1` is used as a normalizer (Bessel's
    /// correction).
    ///
    /// Returns `f64::NAN` if data has less than two entries or if any entry is
    /// `f64::NAN`
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64;
    /// use statrs::statistics::Statistics;
    ///
    /// let x = &[];
    /// assert!(x.variance().is_nan());
    ///
    /// let y = &[0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.variance().is_nan());
    ///
    /// let z = &[0.0, 3.0, -2.0];
    /// assert_eq!(z.variance(), 19.0 / 3.0);
    /// ```
    fn variance(self) -> T;

    /// Estimates the unbiased population standard deviation from the provided
    /// samples
    ///
    /// # Remarks
    ///
    /// On a dataset of size `N`, `N-1` is used as a normalizer (Bessel's
    /// correction).
    ///
    /// Returns `f64::NAN` if data has less than two entries or if any entry is
    /// `f64::NAN`
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64;
    /// use statrs::statistics::Statistics;
    ///
    /// let x = &[];
    /// assert!(x.std_dev().is_nan());
    ///
    /// let y = &[0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.std_dev().is_nan());
    ///
    /// let z = &[0.0, 3.0, -2.0];
    /// assert_eq!(z.std_dev(), (19f64 / 3.0).sqrt());
    /// ```
    fn std_dev(self) -> T;

    /// Evaluates the population variance from a full population.
    ///
    /// # Remarks
    ///
    /// On a dataset of size `N`, `N` is used as a normalizer and would thus
    /// be biased if applied to a subset
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64;
    /// use statrs::statistics::Statistics;
    ///
    /// let x = &[];
    /// assert!(x.population_variance().is_nan());
    ///
    /// let y = &[0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.population_variance().is_nan());
    ///
    /// let z = &[0.0, 3.0, -2.0];
    /// assert_eq!(z.population_variance(), 38.0 / 9.0);
    /// ```
    fn population_variance(self) -> T;

    /// Evaluates the population standard deviation from a full population.
    ///
    /// # Remarks
    ///
    /// On a dataset of size `N`, `N` is used as a normalizer and would thus
    /// be biased if applied to a subset
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64;
    /// use statrs::statistics::Statistics;
    ///
    /// let x = &[];
    /// assert!(x.population_std_dev().is_nan());
    ///
    /// let y = &[0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.population_std_dev().is_nan());
    ///
    /// let z = &[0.0, 3.0, -2.0];
    /// assert_eq!(z.population_std_dev(), (38f64 / 9.0).sqrt());
    /// ```
    fn population_std_dev(self) -> T;

    /// Estimates the unbiased population covariance between the two provided
    /// samples
    ///
    /// # Remarks
    ///
    /// On a dataset of size `N`, `N-1` is used as a normalizer (Bessel's
    /// correction).
    ///
    /// Returns `f64::NAN` if data has less than two entries or if any entry is
    /// `f64::NAN`
    ///
    /// # Panics
    ///
    /// If the two sample containers do not contain the same number of elements
    ///
    /// # Examples
    ///
    /// ```
    /// #[macro_use]
    /// extern crate statrs;
    ///
    /// use std::f64;
    /// use statrs::statistics::Statistics;
    ///
    /// # fn main() {
    /// let x = &[];
    /// assert!(x.covariance(&[]).is_nan());
    ///
    /// let y1 = &[0.0, f64::NAN, 3.0, -2.0];
    /// let y2 = &[-5.0, 4.0, 10.0, f64::NAN];
    /// assert!(y1.covariance(y2).is_nan());
    ///
    /// let z1 = &[0.0, 3.0, -2.0];
    /// let z2 = &[-5.0, 4.0, 10.0];
    /// assert_almost_eq!(z1.covariance(z2), -5.5, 1e-14);
    /// # }
    /// ```
    fn covariance(self, other: Self) -> T;

    /// Evaluates the population covariance between the two provider populations
    ///
    /// # Remarks
    ///
    /// On a dataset of size `N`, `N` is used as a normalizer and would thus be
    /// biased if applied to a subset
    ///
    /// Returns `f64::NAN` if data is empty or any entry is `f64::NAN`
    ///
    /// # Panics
    ///
    /// If the two sample containers do not contain the same number of elements
    ///
    /// # Examples
    ///
    /// ```
    /// #[macro_use]
    /// extern crate statrs;
    ///
    /// use std::f64;
    /// use statrs::statistics::Statistics;
    ///
    /// # fn main() {
    /// let x = &[];
    /// assert!(x.population_covariance(&[]).is_nan());
    ///
    /// let y1 = &[0.0, f64::NAN, 3.0, -2.0];
    /// let y2 = &[-5.0, 4.0, 10.0, f64::NAN];
    /// assert!(y1.population_covariance(y2).is_nan());
    ///
    /// let z1 = &[0.0, 3.0, -2.0];
    /// let z2 = &[-5.0, 4.0, 10.0];
    /// assert_almost_eq!(z1.population_covariance(z2), -11.0 / 3.0, 1e-14);
    /// # }
    /// ```
    fn population_covariance(self, other: Self) -> T;

    /// Estimates the quadratic mean (Root Mean Square) of the data
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty or any entry is `f64::NAN`
    ///
    /// # Examples
    ///
    /// ```
    /// #[macro_use]
    /// extern crate statrs;
    ///
    /// use std::f64;
    /// use statrs::statistics::Statistics;
    ///
    /// # fn main() {
    /// let x = &[];
    /// assert!(x.quadratic_mean().is_nan());
    ///
    /// let y = &[0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.quadratic_mean().is_nan());
    ///
    /// let z = &[0.0, 3.0, -2.0];
    /// // test value from online calculator, could be more accurate
    /// assert_almost_eq!(z.quadratic_mean(), 2.08167, 1e-5);
    /// # }
    /// ```
    fn quadratic_mean(self) -> T;
}
