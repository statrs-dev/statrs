use crate::statistics::*;
use std::f64;

impl OrderStatistics<f64> for [f64] {
    fn order_statistic(&mut self, order: usize) -> f64 {
        let n = self.len();
        match order {
            1 => self.min(),
            _ if order == n => self.max(),
            _ if order < 1 || order > n => f64::NAN,
            _ => select_inplace(self, order - 1),
        }
    }

    fn median(&mut self) -> f64 {
        let k = self.len() / 2;
        if self.len() % 2 != 0 {
            select_inplace(self, k)
        } else {
            (select_inplace(self, k.saturating_sub(1)) + select_inplace(self, k)) / 2.0
        }
    }

    fn quantile(&mut self, tau: f64) -> f64 {
        if tau < 0.0 || tau > 1.0 || self.is_empty() {
            return f64::NAN;
        }

        let h = (self.len() as f64 + 1.0 / 3.0) * tau + 1.0 / 3.0;
        let hf = h as i64;

        if hf <= 0 || tau == 0.0 {
            return self.min();
        }
        if hf >= self.len() as i64 || ulps_eq!(tau, 1.0) {
            return self.max();
        }

        let a = select_inplace(self, (hf as usize).saturating_sub(1));
        let b = select_inplace(self, hf as usize);
        a + (h - hf as f64) * (b - a)
    }

    fn percentile(&mut self, p: usize) -> f64 {
        self.quantile(p as f64 / 100.0)
    }

    fn lower_quartile(&mut self) -> f64 {
        self.quantile(0.25)
    }

    fn upper_quartile(&mut self) -> f64 {
        self.quantile(0.75)
    }

    fn interquartile_range(&mut self) -> f64 {
        self.upper_quartile() - self.lower_quartile()
    }

    fn ranks(&mut self, tie_breaker: RankTieBreaker) -> Vec<f64> {
        let n = self.len();
        let mut ranks: Vec<f64> = vec![0.0; n];
        let mut enumerated: Vec<_> = self.iter().enumerate().collect();
        enumerated.sort_by(|(_, el_a), (_, el_b)| el_a.partial_cmp(el_b).unwrap());
        match tie_breaker {
            RankTieBreaker::First => {
                for (i, idx) in enumerated.into_iter().map(|(idx, _)| idx).enumerate() {
                    ranks[idx] = (i + 1) as f64
                }
                ranks
            }
            _ => {
                let mut prev = 0;
                let mut prev_idx = 0;
                let mut prev_elt = 0.0;
                for (i, (idx, elt)) in enumerated.iter().cloned().enumerate() {
                    if i == 0 {
                        prev_idx = idx;
                        prev_elt = *elt;
                    }
                    if (*elt - prev_elt).abs() <= 0.0 {
                        continue;
                    }
                    if i == prev + 1 {
                        ranks[prev_idx] = i as f64;
                    } else {
                        handle_rank_ties(&mut ranks, &enumerated, prev, i, tie_breaker);
                    }
                    prev = i;
                    prev_idx = idx;
                    prev_elt = *elt;
                }

                handle_rank_ties(&mut ranks, &enumerated, prev, n, tie_breaker);
                ranks
            }
        }
    }
}

impl Min<f64> for [f64] {
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
    /// use statrs::statistics::Min;
    ///
    /// let x: [f64; 0] = [];
    /// assert!(x.min().is_nan());
    ///
    /// let y = [0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.min().is_nan());
    ///
    /// let z = [0.0, 3.0, -2.0];
    /// assert_eq!(z.min(), -2.0);
    /// ```
    fn min(&self) -> f64 {
        Statistics::min(self)
    }
}

impl Max<f64> for [f64] {
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
    /// use statrs::statistics::Max;
    ///
    /// let x: [f64; 0] = [];
    /// assert!(x.max().is_nan());
    ///
    /// let y = [0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.max().is_nan());
    ///
    /// let z = [0.0, 3.0, -2.0];
    /// assert_eq!(z.max(), 3.0);
    /// ```
    fn max(&self) -> f64 {
        Statistics::max(self)
    }
}

impl Mean<f64> for [f64] {
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
    /// use statrs::statistics::Mean;
    ///
    /// # fn main() {
    /// let x = [];
    /// assert!(x.mean().is_nan());
    ///
    /// let y = [0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.mean().is_nan());
    ///
    /// let z = [0.0, 3.0, -2.0];
    /// assert_almost_eq!(z.mean(), 1.0 / 3.0, 1e-15);
    /// # }
    /// ```
    fn mean(&self) -> f64 {
        Statistics::mean(self)
    }
}

impl Variance<f64> for [f64] {
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
    /// use statrs::statistics::Variance;
    ///
    /// let x = [];
    /// assert!(x.variance().is_nan());
    ///
    /// let y = [0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.variance().is_nan());
    ///
    /// let z = [0.0, 3.0, -2.0];
    /// assert_eq!(z.variance(), 19.0 / 3.0);
    /// ```
    fn variance(&self) -> f64 {
        Statistics::variance(self)
    }

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
    /// use statrs::statistics::Variance;
    ///
    /// let x = [];
    /// assert!(x.std_dev().is_nan());
    ///
    /// let y = [0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.std_dev().is_nan());
    ///
    /// let z = [0.0, 3.0, -2.0];
    /// assert_eq!(z.std_dev(), (19f64 / 3.0).sqrt());
    /// ```
    fn std_dev(&self) -> f64 {
        Statistics::std_dev(self)
    }
}

impl Median<f64> for [f64] {
    /// Returns the median value from the data
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Median;
    ///
    /// let x = [];
    /// assert!(x.median().is_nan());
    ///
    /// let y = [0.0, 3.0, -2.0];
    /// assert_eq!(y.median(), 0.0);
    fn median(&self) -> f64 {
        let mut copy = self.to_vec();
        OrderStatistics::median(&mut *copy)
    }
}

fn handle_rank_ties(
    ranks: &mut [f64],
    index: &[(usize, &f64)],
    a: usize,
    b: usize,
    tie_breaker: RankTieBreaker,
) {
    let rank = match tie_breaker {
        // equivalent to (b + a - 1) as f64 / 2.0 + 1.0 but less overflow issues
        RankTieBreaker::Average => b as f64 / 2.0 + a as f64 / 2.0 + 0.5,
        RankTieBreaker::Min => (a + 1) as f64,
        RankTieBreaker::Max => b as f64,
        RankTieBreaker::First => unreachable!(),
    };
    for i in &index[a..b] {
        ranks[i.0] = rank
    }
}

// Selection algorithm from Numerical Recipes
// See: https://en.wikipedia.org/wiki/Selection_algorithm
fn select_inplace(arr: &mut [f64], rank: usize) -> f64 {
    if rank == 0 {
        return arr.min();
    }
    if rank > arr.len() - 1 {
        return arr.max();
    }

    let mut low = 0;
    let mut high = arr.len() - 1;
    loop {
        if high <= low + 1 {
            if high == low + 1 && arr[high] < arr[low] {
                arr.swap(low, high)
            }
            return arr[rank];
        }

        let middle = (low + high) / 2;
        arr.swap(middle, low + 1);

        if arr[low] > arr[high] {
            arr.swap(low, high);
        }
        if arr[low + 1] > arr[high] {
            arr.swap(low + 1, high);
        }
        if arr[low] > arr[low + 1] {
            arr.swap(low, low + 1);
        }

        let mut begin = low + 1;
        let mut end = high;
        let pivot = arr[begin];
        loop {
            loop {
                begin += 1;
                if arr[begin] >= pivot {
                    break;
                }
            }
            loop {
                end -= 1;
                if arr[end] <= pivot {
                    break;
                }
            }
            if end < begin {
                break;
            }
            arr.swap(begin, end);
        }

        arr[low + 1] = arr[end];
        arr[end] = pivot;

        if end >= rank {
            high = end - 1;
        }
        if end <= rank {
            low = begin;
        }
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod test {
    use std::f64;
    use crate::statistics::*;

    #[test]
    fn test_order_statistic_short() {
        let mut data = [-1.0, 5.0, 0.0, -3.0, 10.0, -0.5, 4.0, 1.0, 6.0];
        assert!(data.order_statistic(0).is_nan());
        assert_eq!(data.order_statistic(1), -3.0);
        assert_eq!(data.order_statistic(2), -1.0);
        assert_eq!(data.order_statistic(3), -0.5);
        assert_eq!(data.order_statistic(7), 5.0);
        assert_eq!(data.order_statistic(8), 6.0);
        assert_eq!(data.order_statistic(9), 10.0);
        assert!(data.order_statistic(10).is_nan());
    }

    #[test]
    fn test_quantile_short() {
        let mut data = [-1.0, 5.0, 0.0, -3.0, 10.0, -0.5, 4.0, 0.2, 1.0, 6.0];
        assert_eq!(data.quantile(0.0), -3.0);
        assert_eq!(data.quantile(1.0), 10.0);
        assert_almost_eq!(data.quantile(0.5), 3.0 / 5.0, 1e-15);
        assert_almost_eq!(data.quantile(0.2), -4.0 / 5.0, 1e-15);
        assert_eq!(data.quantile(0.7), 137.0 / 30.0);
        assert_eq!(data.quantile(0.01), -3.0);
        assert_eq!(data.quantile(0.99), 10.0);
        assert_almost_eq!(data.quantile(0.52), 287.0 / 375.0, 1e-15);
        assert_almost_eq!(data.quantile(0.325), -37.0 / 240.0, 1e-15);
    }

    // TODO: need coverage for case where data.length > 10 to cover quick sort
    #[test]
    fn test_ranks() {
        let mut sorted_distinct = [1.0, 2.0, 4.0, 7.0, 8.0, 9.0, 10.0, 12.0];
        let mut sorted_ties = [1.0, 2.0, 2.0, 7.0, 9.0, 9.0, 10.0, 12.0];
        assert_eq!(sorted_distinct.ranks(RankTieBreaker::Average), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(sorted_ties.ranks(RankTieBreaker::Average), [1.0, 2.5, 2.5, 4.0, 5.5, 5.5, 7.0, 8.0]);
        assert_eq!(sorted_distinct.ranks(RankTieBreaker::Min), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(sorted_ties.ranks(RankTieBreaker::Min), [1.0, 2.0, 2.0, 4.0, 5.0, 5.0, 7.0, 8.0]);
        assert_eq!(sorted_distinct.ranks(RankTieBreaker::Max), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(sorted_ties.ranks(RankTieBreaker::Max), [1.0, 3.0, 3.0, 4.0, 6.0, 6.0, 7.0, 8.0]);
        assert_eq!(sorted_distinct.ranks(RankTieBreaker::First), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(sorted_ties.ranks(RankTieBreaker::First), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let distinct = [1.0, 8.0, 12.0, 7.0, 2.0, 9.0, 10.0, 4.0];
        let ties = [1.0, 9.0, 12.0, 7.0, 2.0, 9.0, 10.0, 2.0];
        assert_eq!(distinct.clone().ranks(RankTieBreaker::Average), [1.0, 5.0, 8.0, 4.0, 2.0, 6.0, 7.0, 3.0]);
        assert_eq!(ties.clone().ranks(RankTieBreaker::Average), [1.0, 5.5, 8.0, 4.0, 2.5, 5.5, 7.0, 2.5]);
        assert_eq!(distinct.clone().ranks(RankTieBreaker::Min), [1.0, 5.0, 8.0, 4.0, 2.0, 6.0, 7.0, 3.0]);
        assert_eq!(ties.clone().ranks(RankTieBreaker::Min), [1.0, 5.0, 8.0, 4.0, 2.0, 5.0, 7.0, 2.0]);
        assert_eq!(distinct.clone().ranks(RankTieBreaker::Max), [1.0, 5.0, 8.0, 4.0, 2.0, 6.0, 7.0, 3.0]);
        assert_eq!(ties.clone().ranks(RankTieBreaker::Max), [1.0, 6.0, 8.0, 4.0, 3.0, 6.0, 7.0, 3.0]);
        assert_eq!(distinct.clone().ranks(RankTieBreaker::First), [1.0, 5.0, 8.0, 4.0, 2.0, 6.0, 7.0, 3.0]);
        assert_eq!(ties.clone().ranks(RankTieBreaker::First), [1.0, 5.0, 8.0, 4.0, 2.0, 6.0, 7.0, 3.0]);
    }

    #[test]
    fn test_median_short() {
        let even = [-1.0, 5.0, 0.0, -3.0, 10.0, -0.5, 4.0, 0.2, 1.0, 6.0];
        assert_eq!(even.median(), 0.6);

        let odd = [-1.0, 5.0, 0.0, -3.0, 10.0, -0.5, 4.0, 0.2, 1.0];
        assert_eq!(odd.median(), 0.2);
    }

    #[test]
    fn test_median_long_constant_seq() {
        let even = vec![2.0; 100000];
        assert_eq!(2.0, even.median());

        let odd = vec![2.0; 100001];
        assert_eq!(2.0, odd.median());
    }

    // TODO: test codeplex issue 5667 (Math.NET)

    #[test]
    fn test_median_robust_on_infinities() {
        let mut data3 = [2.0, f64::NEG_INFINITY, f64::INFINITY];
        assert_eq!(data3.median(), 2.0);
        assert_eq!(data3.median(), 2.0);

        data3 = [f64::NEG_INFINITY, 2.0, f64::INFINITY];
        assert_eq!(data3.median(), 2.0);
        assert_eq!(data3.median(), 2.0);

        data3 = [f64::NEG_INFINITY, f64::INFINITY, 2.0];
        assert_eq!(data3.median(), 2.0);
        assert_eq!(data3.median(), 2.0);

        let data4 = [f64::NEG_INFINITY, 2.0, 3.0, f64::INFINITY];
        assert_eq!(data4.median(), 2.5);
        assert_eq!(data4.median(), 2.5);
    }
}
