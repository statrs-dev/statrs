use super::RankTieBreaker;

/// The `OrderStatistics` trait provides statistical utilities
/// having to do with ordering. All the algorithms are in-place thus requiring
/// a mutable borrow.
pub trait OrderStatistics<T> {
    /// Returns the order statistic `(order 1..N)` from the data
    ///
    /// # Remarks
    ///
    /// No sorting is assumed. Order must be one-based (between `1` and `N`
    /// inclusive)
    /// Returns `f64::NAN` if order is outside the viable range or data is
    /// empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::OrderStatistics;
    /// use statrs::statistics::Data;
    ///
    /// let x = [];
    /// let mut x = Data::new(x);
    /// assert!(x.order_statistic(1).is_nan());
    ///
    /// let y = [0.0, 3.0, -2.0];
    /// let mut y = Data::new(y);
    /// assert!(y.order_statistic(0).is_nan());
    /// assert!(y.order_statistic(4).is_nan());
    /// assert_eq!(y.order_statistic(2), 0.0);
    /// assert!(y != Data::new([0.0, 3.0, -2.0]));
    /// ```
    fn order_statistic(&mut self, order: usize) -> T;

    /// Returns the median value from the data
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::OrderStatistics;
    /// use statrs::statistics::Data;
    ///
    /// let x = [];
    /// let mut x = Data::new(x);
    /// assert!(x.median().is_nan());
    ///
    /// let y = [0.0, 3.0, -2.0];
    /// let mut y = Data::new(y);
    /// assert_eq!(y.median(), 0.0);
    /// assert!(y != Data::new([0.0, 3.0, -2.0]));
    fn median(&mut self) -> T;

    /// Estimates the tau-th quantile from the data. The tau-th quantile
    /// is the data value where the cumulative distribution function crosses
    /// tau.
    ///
    /// # Remarks
    ///
    /// No sorting is assumed. Tau must be between `0` and `1` inclusive.
    /// Returns `f64::NAN` if data is empty or tau is outside the inclusive
    /// range.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::OrderStatistics;
    /// use statrs::statistics::Data;
    ///
    /// let x = [];
    /// let mut x = Data::new(x);
    /// assert!(x.quantile(0.5).is_nan());
    ///
    /// let y = [0.0, 3.0, -2.0];
    /// let mut y = Data::new(y);
    /// assert!(y.quantile(-1.0).is_nan());
    /// assert!(y.quantile(2.0).is_nan());
    /// assert_eq!(y.quantile(0.5), 0.0);
    /// assert!(y != Data::new([0.0, 3.0, -2.0]));
    /// ```
    fn quantile(&mut self, tau: f64) -> T;

    /// Estimates the p-Percentile value from the data.
    ///
    /// # Remarks
    ///
    /// Use quantile for non-integer percentiles. `p` must be between `0` and
    /// `100` inclusive.
    /// Returns `f64::NAN` if data is empty or `p` is outside the inclusive
    /// range.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::OrderStatistics;
    /// use statrs::statistics::Data;
    ///
    /// let x = [];
    /// let mut x = Data::new(x);
    /// assert!(x.percentile(0).is_nan());
    ///
    /// let y = [1.0, 5.0, 3.0, 4.0, 10.0, 9.0, 6.0, 7.0, 8.0, 2.0];
    /// let mut y = Data::new(y);
    /// assert_eq!(y.percentile(0), 1.0);
    /// assert_eq!(y.percentile(50), 5.5);
    /// assert_eq!(y.percentile(100), 10.0);
    /// assert!(y.percentile(105).is_nan());
    /// assert!(y != Data::new([1.0, 5.0, 3.0, 4.0, 10.0, 9.0, 6.0, 7.0, 8.0, 2.0]));
    /// ```
    fn percentile(&mut self, p: usize) -> T;

    /// Estimates the first quartile value from the data.
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty
    ///
    /// # Examples
    ///
    /// ```
    /// #[macro_use]
    /// extern crate statrs;
    ///
    /// use statrs::statistics::OrderStatistics;
    /// use statrs::statistics::Data;
    ///
    /// # fn main() {
    /// let x = [];
    /// let mut x = Data::new(x);
    /// assert!(x.lower_quartile().is_nan());
    ///
    /// let y = [2.0, 1.0, 3.0, 4.0];
    /// let mut y = Data::new(y);
    /// assert_almost_eq!(y.lower_quartile(), 1.416666666666666, 1e-15);
    /// assert!(y != Data::new([2.0, 1.0, 3.0, 4.0]));
    /// # }
    /// ```
    fn lower_quartile(&mut self) -> T;

    /// Estimates the third quartile value from the data.
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty
    ///
    /// # Examples
    ///
    /// ```
    /// #[macro_use]
    /// extern crate statrs;
    ///
    /// use statrs::statistics::OrderStatistics;
    /// use statrs::statistics::Data;
    ///
    /// # fn main() {
    /// let x = [];
    /// let mut x = Data::new(x);
    /// assert!(x.upper_quartile().is_nan());
    ///
    /// let y = [2.0, 1.0, 3.0, 4.0];
    /// let mut y = Data::new(y);
    /// assert_almost_eq!(y.upper_quartile(), 3.5833333333333333, 1e-15);
    /// assert!(y != Data::new([2.0, 1.0, 3.0, 4.0]));
    /// # }
    /// ```
    fn upper_quartile(&mut self) -> T;

    /// Estimates the inter-quartile range from the data.
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty
    ///
    /// # Examples
    ///
    /// ```
    /// #[macro_use]
    /// extern crate statrs;
    ///
    /// use statrs::statistics::Data;
    /// use statrs::statistics::OrderStatistics;
    ///
    /// # fn main() {
    /// let x = [];
    /// let mut x = Data::new(x);
    /// assert!(x.interquartile_range().is_nan());
    ///
    /// let y = [2.0, 1.0, 3.0, 4.0];
    /// let mut y = Data::new(y);
    /// assert_almost_eq!(y.interquartile_range(), 2.166666666666667, 1e-15);
    /// assert!(y != Data::new([2.0, 1.0, 3.0, 4.0]));
    /// # }
    /// ```
    fn interquartile_range(&mut self) -> T;

    /// Evaluates the rank of each entry of the data.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::{OrderStatistics, RankTieBreaker};
    /// use statrs::statistics::Data;
    ///
    /// let x = [];
    /// let mut x = Data::new(x);
    /// assert_eq!(x.ranks(RankTieBreaker::Average).len(), 0);
    ///
    /// let y = [1.0, 3.0, 2.0, 2.0];
    /// let mut y = Data::new([1.0, 3.0, 2.0, 2.0]);
    /// assert_eq!(y.clone().ranks(RankTieBreaker::Average), [1.0, 4.0,
    /// 2.5, 2.5]);
    /// assert_eq!(y.clone().ranks(RankTieBreaker::Min), [1.0, 4.0, 2.0,
    /// 2.0]);
    /// ```
    fn ranks(&mut self, tie_breaker: RankTieBreaker) -> Vec<T>;
}
