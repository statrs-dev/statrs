use crate::statistics::*;
use core::ops::{Index, IndexMut};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Data<D>(D);

impl<D, I> std::fmt::Display for Data<D>
where
    D: Clone + IntoIterator<Item = I>,
    I: Clone + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut tee = self.0.clone().into_iter();
        write!(f, "Data([")?;

        if let Some(v) = tee.next() {
            write!(f, "{}", v)?;
        }
        for _ in 1..5 {
            if let Some(v) = tee.next() {
                write!(f, ", {}", v)?;
            }
        }
        if tee.next().is_some() {
            write!(f, "...")?;
        }

        write!(f, "])")
    }
}

impl<D: AsRef<[f64]>> Index<usize> for Data<D> {
    type Output = f64;

    fn index(&self, i: usize) -> &f64 {
        &self.0.as_ref()[i]
    }
}

impl<D: AsMut<[f64]> + AsRef<[f64]>> IndexMut<usize> for Data<D> {
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        &mut self.0.as_mut()[i]
    }
}

impl<D: AsMut<[f64]> + AsRef<[f64]>> Data<D> {
    pub fn new(data: D) -> Self {
        Data(data)
    }

    pub fn swap(&mut self, i: usize, j: usize) {
        self.0.as_mut().swap(i, j)
    }

    pub fn len(&self) -> usize {
        self.0.as_ref().len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.as_ref().len() == 0
    }

    pub fn iter(&self) -> core::slice::Iter<'_, f64> {
        self.0.as_ref().iter()
    }

    // Selection algorithm from Numerical Recipes
    // See: https://en.wikipedia.org/wiki/Selection_algorithm
    fn select_inplace(&mut self, rank: usize) -> f64 {
        if rank == 0 {
            return self.min();
        }
        if rank > self.len() - 1 {
            return self.max();
        }

        let mut low = 0;
        let mut high = self.len() - 1;
        loop {
            if high <= low + 1 {
                if high == low + 1 && self[high] < self[low] {
                    self.swap(low, high)
                }
                return self[rank];
            }

            let middle = (low + high) / 2;
            self.swap(middle, low + 1);

            if self[low] > self[high] {
                self.swap(low, high);
            }
            if self[low + 1] > self[high] {
                self.swap(low + 1, high);
            }
            if self[low] > self[low + 1] {
                self.swap(low, low + 1);
            }

            let mut begin = low + 1;
            let mut end = high;
            let pivot = self[begin];
            loop {
                loop {
                    begin += 1;
                    if self[begin] >= pivot {
                        break;
                    }
                }
                loop {
                    end -= 1;
                    if self[end] <= pivot {
                        break;
                    }
                }
                if end < begin {
                    break;
                }
                self.swap(begin, end);
            }

            self[low + 1] = self[end];
            self[end] = pivot;

            if end >= rank {
                high = end - 1;
            }
            if end <= rank {
                low = begin;
            }
        }
    }
}

#[cfg(feature = "rand")]
impl<D: AsRef<[f64]>> ::rand::distributions::Distribution<f64> for Data<D> {
    fn sample<R: ::rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand::prelude::SliceRandom;

        *self.0.as_ref().choose(rng).unwrap()
    }
}

impl<D: AsMut<[f64]> + AsRef<[f64]>> OrderStatistics<f64> for Data<D> {
    fn order_statistic(&mut self, order: usize) -> f64 {
        let n = self.len();
        match order {
            1 => self.min(),
            _ if order == n => self.max(),
            _ if order < 1 || order > n => f64::NAN,
            _ => self.select_inplace(order - 1),
        }
    }

    fn median(&mut self) -> f64 {
        let k = self.len() / 2;
        if self.len() % 2 != 0 {
            self.select_inplace(k)
        } else {
            (self.select_inplace(k.saturating_sub(1)) + self.select_inplace(k)) / 2.0
        }
    }

    fn quantile(&mut self, tau: f64) -> f64 {
        if !(0.0..=1.0).contains(&tau) || self.is_empty() {
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

        let a = self.select_inplace((hf as usize).saturating_sub(1));
        let b = self.select_inplace(hf as usize);
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

impl<D: AsMut<[f64]> + AsRef<[f64]>> Min<f64> for Data<D> {
    /// Returns the minimum value in the data
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Min;
    /// use statrs::statistics::Data;
    ///
    /// let x = [];
    /// let x = Data::new(x);
    /// assert!(x.min().is_nan());
    ///
    /// let y = [0.0, f64::NAN, 3.0, -2.0];
    /// let y = Data::new(y);
    /// assert!(y.min().is_nan());
    ///
    /// let z = [0.0, 3.0, -2.0];
    /// let z = Data::new(z);
    /// assert_eq!(z.min(), -2.0);
    /// ```
    fn min(&self) -> f64 {
        Statistics::min(self.iter())
    }
}

impl<D: AsMut<[f64]> + AsRef<[f64]>> Max<f64> for Data<D> {
    /// Returns the maximum value in the data
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Max;
    /// use statrs::statistics::Data;
    ///
    /// let x = [];
    /// let x = Data::new(x);
    /// assert!(x.max().is_nan());
    ///
    /// let y = [0.0, f64::NAN, 3.0, -2.0];
    /// let y = Data::new(y);
    /// assert!(y.max().is_nan());
    ///
    /// let z = [0.0, 3.0, -2.0];
    /// let z = Data::new(z);
    /// assert_eq!(z.max(), 3.0);
    /// ```
    fn max(&self) -> f64 {
        Statistics::max(self.iter())
    }
}

impl<D: AsMut<[f64]> + AsRef<[f64]>> Distribution<f64> for Data<D> {
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
    /// use statrs::statistics::Distribution;
    /// use statrs::statistics::Data;
    ///
    /// # fn main() {
    /// let x = [];
    /// let x = Data::new(x);
    /// assert!(x.mean().unwrap().is_nan());
    ///
    /// let y = [0.0, f64::NAN, 3.0, -2.0];
    /// let y = Data::new(y);
    /// assert!(y.mean().unwrap().is_nan());
    ///
    /// let z = [0.0, 3.0, -2.0];
    /// let z = Data::new(z);
    /// assert_almost_eq!(z.mean().unwrap(), 1.0 / 3.0, 1e-15);
    /// # }
    /// ```
    fn mean(&self) -> Option<f64> {
        Some(Statistics::mean(self.iter()))
    }

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
    /// use statrs::statistics::Distribution;
    /// use statrs::statistics::Data;
    ///
    /// let x = [];
    /// let x = Data::new(x);
    /// assert!(x.variance().unwrap().is_nan());
    ///
    /// let y = [0.0, f64::NAN, 3.0, -2.0];
    /// let y = Data::new(y);
    /// assert!(y.variance().unwrap().is_nan());
    ///
    /// let z = [0.0, 3.0, -2.0];
    /// let z = Data::new(z);
    /// assert_eq!(z.variance().unwrap(), 19.0 / 3.0);
    /// ```
    fn variance(&self) -> Option<f64> {
        Some(Statistics::variance(self.iter()))
    }
}

impl<D: AsMut<[f64]> + AsRef<[f64]> + Clone> Median<f64> for Data<D> {
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
    /// use statrs::statistics::Data;
    ///
    /// let x = [];
    /// let x = Data::new(x);
    /// assert!(x.median().is_nan());
    ///
    /// let y = [0.0, 3.0, -2.0];
    /// let y = Data::new(y);
    /// assert_eq!(y.median(), 0.0);
    fn median(&self) -> f64 {
        let mut v = self.clone();
        OrderStatistics::median(&mut v)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statistics::*;

    #[test]
    fn test_order_statistic_short() {
        let data = [-1.0, 5.0, 0.0, -3.0, 10.0, -0.5, 4.0, 1.0, 6.0];
        let mut data = Data::new(data);
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
        let data = [-1.0, 5.0, 0.0, -3.0, 10.0, -0.5, 4.0, 0.2, 1.0, 6.0];
        let mut data = Data::new(data);
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

    #[test]
    fn test_ranks() {
        let sorted_distinct = [1.0, 2.0, 4.0, 7.0, 8.0, 9.0, 10.0, 12.0];
        let mut sorted_distinct = Data::new(sorted_distinct);
        let sorted_ties = [1.0, 2.0, 2.0, 7.0, 9.0, 9.0, 10.0, 12.0];
        let mut sorted_ties = Data::new(sorted_ties);
        assert_eq!(
            sorted_distinct.ranks(RankTieBreaker::Average),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
        assert_eq!(
            sorted_ties.ranks(RankTieBreaker::Average),
            [1.0, 2.5, 2.5, 4.0, 5.5, 5.5, 7.0, 8.0]
        );
        assert_eq!(
            sorted_distinct.ranks(RankTieBreaker::Min),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
        assert_eq!(
            sorted_ties.ranks(RankTieBreaker::Min),
            [1.0, 2.0, 2.0, 4.0, 5.0, 5.0, 7.0, 8.0]
        );
        assert_eq!(
            sorted_distinct.ranks(RankTieBreaker::Max),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
        assert_eq!(
            sorted_ties.ranks(RankTieBreaker::Max),
            [1.0, 3.0, 3.0, 4.0, 6.0, 6.0, 7.0, 8.0]
        );
        assert_eq!(
            sorted_distinct.ranks(RankTieBreaker::First),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
        assert_eq!(
            sorted_ties.ranks(RankTieBreaker::First),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );

        let distinct = [1.0, 8.0, 12.0, 7.0, 2.0, 9.0, 10.0, 4.0];
        let distinct = Data::new(distinct);
        let ties = [1.0, 9.0, 12.0, 7.0, 2.0, 9.0, 10.0, 2.0];
        let ties = Data::new(ties);
        assert_eq!(
            distinct.clone().ranks(RankTieBreaker::Average),
            [1.0, 5.0, 8.0, 4.0, 2.0, 6.0, 7.0, 3.0]
        );
        assert_eq!(
            ties.clone().ranks(RankTieBreaker::Average),
            [1.0, 5.5, 8.0, 4.0, 2.5, 5.5, 7.0, 2.5]
        );
        assert_eq!(
            distinct.clone().ranks(RankTieBreaker::Min),
            [1.0, 5.0, 8.0, 4.0, 2.0, 6.0, 7.0, 3.0]
        );
        assert_eq!(
            ties.clone().ranks(RankTieBreaker::Min),
            [1.0, 5.0, 8.0, 4.0, 2.0, 5.0, 7.0, 2.0]
        );
        assert_eq!(
            distinct.clone().ranks(RankTieBreaker::Max),
            [1.0, 5.0, 8.0, 4.0, 2.0, 6.0, 7.0, 3.0]
        );
        assert_eq!(
            ties.clone().ranks(RankTieBreaker::Max),
            [1.0, 6.0, 8.0, 4.0, 3.0, 6.0, 7.0, 3.0]
        );
        assert_eq!(
            distinct.clone().ranks(RankTieBreaker::First),
            [1.0, 5.0, 8.0, 4.0, 2.0, 6.0, 7.0, 3.0]
        );
        assert_eq!(
            ties.clone().ranks(RankTieBreaker::First),
            [1.0, 5.0, 8.0, 4.0, 2.0, 6.0, 7.0, 3.0]
        );
    }

    #[test]
    fn test_median_short() {
        let even = [-1.0, 5.0, 0.0, -3.0, 10.0, -0.5, 4.0, 0.2, 1.0, 6.0];
        let even = Data::new(even);
        assert_eq!(even.median(), 0.6);

        let odd = [-1.0, 5.0, 0.0, -3.0, 10.0, -0.5, 4.0, 0.2, 1.0];
        let odd = Data::new(odd);
        assert_eq!(odd.median(), 0.2);
    }

    #[test]
    fn test_median_long_constant_seq() {
        let even = vec![2.0; 100000];
        let even = Data::new(even);
        assert_eq!(2.0, even.median());

        let odd = vec![2.0; 100001];
        let odd = Data::new(odd);
        assert_eq!(2.0, odd.median());
    }

    // TODO: test codeplex issue 5667 (Math.NET)

    #[test]
    fn test_median_robust_on_infinities() {
        let data3 = [2.0, f64::NEG_INFINITY, f64::INFINITY];
        let data3 = Data::new(data3);
        assert_eq!(data3.median(), 2.0);
        assert_eq!(data3.median(), 2.0);

        let data3 = [f64::NEG_INFINITY, 2.0, f64::INFINITY];
        let data3 = Data::new(data3);
        assert_eq!(data3.median(), 2.0);
        assert_eq!(data3.median(), 2.0);

        let data3 = [f64::NEG_INFINITY, f64::INFINITY, 2.0];
        let data3 = Data::new(data3);
        assert_eq!(data3.median(), 2.0);
        assert_eq!(data3.median(), 2.0);

        let data4 = [f64::NEG_INFINITY, 2.0, 3.0, f64::INFINITY];
        let data4 = Data::new(data4);
        assert_eq!(data4.median(), 2.5);
        assert_eq!(data4.median(), 2.5);
    }
    #[test]
    fn test_foo() {
        let arr = [0.0, 1.0, 2.0, 3.0];
        let mut arr = Data::new(arr);
        arr.order_statistic(2);
    }
}
