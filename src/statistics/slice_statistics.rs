use crate::statistics::*;
use core::ops::{Index, IndexMut};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Data<D>(D);

impl<D, I> core::fmt::Display for Data<D>
where
    D: Clone + IntoIterator<Item = I>,
    I: Clone + core::fmt::Display,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut tee = self.0.clone().into_iter();
        write!(f, "Data([")?;

        if let Some(v) = tee.next() {
            write!(f, "{v}")?;
        }
        for _ in 1..5 {
            if let Some(v) = tee.next() {
                write!(f, ", {v}")?;
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

    // Introselect via `select_nth_unstable_by`, ordered by `total_cmp` so NaN
    // has a defined position instead of the arbitrary one the old Numerical
    // Recipes partition produced.
    //
    // Already-ordered input is checked first. Median-of-three picked a perfect
    // pivot on such input, so plain introselect was slower there than the code
    // it replaced; indexing is faster than either. The checks stop at the first
    // out-of-order pair, and `median` and `quantile` select twice, so they
    // benefit twice.
    fn select_inplace(&mut self, rank: usize) -> f64 {
        if rank == 0 {
            return self.min();
        }
        if rank > self.len() - 1 {
            return self.max();
        }

        let slice = self.0.as_mut();
        if slice.is_sorted_by(|a, b| a.total_cmp(b).is_le()) {
            return slice[rank];
        }
        if slice.is_sorted_by(|a, b| a.total_cmp(b).is_ge()) {
            return slice[slice.len() - 1 - rank];
        }

        *slice.select_nth_unstable_by(rank, |a, b| a.total_cmp(b)).1
    }
}

#[cfg(feature = "rand")]
impl<D: AsRef<[f64]>> ::rand::distr::Distribution<f64> for Data<D> {
    fn sample<R: ::rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand::prelude::IndexedRandom;

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
        if !self.len().is_multiple_of(2) {
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
        if hf >= self.len() as i64 || crate::prec::ulps_eq!(tau, 1.0) {
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

    #[cfg(feature = "std")]
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
    /// use approx::assert_abs_diff_eq;
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
    /// assert_abs_diff_eq!(z.mean().unwrap(), 1.0 / 3.0, epsilon = 1e-15);
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

#[cfg(feature = "std")]
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

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::prec;

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
        prec::assert_abs_diff_eq!(data.quantile(0.5), 3.0 / 5.0, epsilon = 1e-15);
        prec::assert_abs_diff_eq!(data.quantile(0.2), -4.0 / 5.0, epsilon = 1e-15);
        assert_eq!(data.quantile(0.7), 137.0 / 30.0);
        assert_eq!(data.quantile(0.01), -3.0);
        assert_eq!(data.quantile(0.99), 10.0);
        prec::assert_abs_diff_eq!(data.quantile(0.52), 287.0 / 375.0, epsilon = 1e-15);
        prec::assert_abs_diff_eq!(data.quantile(0.325), -37.0 / 240.0, epsilon = 1e-15);
    }

    #[test]
    #[cfg(feature = "std")]
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
    #[cfg(feature = "std")]
    fn test_median_long_constant_seq() {
        let even = vec![2.0; 100000];
        let even = Data::new(even);
        assert_eq!(2.0, even.median());

        let odd = vec![2.0; 100001];
        let odd = Data::new(odd);
        assert_eq!(2.0, odd.median());
    }

    // TODO: test codeplex issue 5667 (Math.NET)

    /// Selection agrees with sorting, on ordinary values across shapes and
    /// sizes. Compares bit patterns, so a wrong sign of zero is a failure.
    #[test]
    #[cfg(all(feature = "std", feature = "rand"))]
    fn test_order_statistics_match_sorted_reference() {
        use ::rand::{RngExt, SeedableRng, rngs::StdRng};

        fn shaped(shape: &str, n: usize) -> Vec<f64> {
            match shape {
                "scattered" => {
                    let mut rng = StdRng::seed_from_u64(0x5EED);
                    (0..n).map(|_| rng.random_range(-500.0..500.0)).collect()
                }
                "sorted" => (0..n).map(|i| i as f64 * 1.5).collect(),
                "reversed" => (0..n).rev().map(|i| i as f64 * 1.5).collect(),
                "organ_pipe" => (0..n).map(|i| if i < n / 2 { i } else { n - i } as f64).collect(),
                "duplicates" => (0..n).map(|i| (i % 3) as f64).collect(),
                "constant" => vec![4.25; n],
                "negatives" => (0..n).map(|i| -((i * 37 % 100) as f64)).collect(),
                // -0.0 and +0.0 compare equal under `<=` but not under
                // `total_cmp`, so a fast path keyed on `<=` reports this as
                // sorted and returns the wrong zero (statrs-dev/statrs#407).
                "signed_zeros" => (0..n)
                    .map(|i| match i % 4 {
                        0 => -1.0,
                        1 => 0.0,
                        2 => -0.0,
                        _ => 1.0,
                    })
                    .collect(),
                other => panic!("unknown shape {other}"),
            }
        }

        for n in [1usize, 2, 3, 4, 5, 8, 17, 101, 1000] {
            for shape in [
                "scattered", "sorted", "reversed", "organ_pipe", "duplicates", "constant",
                "negatives", "signed_zeros",
            ] {
                let data = shaped(shape, n);
                let mut sorted = data.clone();
                sorted.sort_by(f64::total_cmp);

                // every order statistic, 1-based
                for order in 1..=n {
                    let mut d = Data::new(data.clone());
                    let got = d.order_statistic(order);
                    // bits, not value: -0.0 == +0.0 would hide a wrong zero
                    assert_eq!(
                        got.to_bits(),
                        sorted[order - 1].to_bits(),
                        "{shape}/{n}: order_statistic({order}) = {got}, want {}",
                        sorted[order - 1]
                    );
                }

                // out of range on either side
                let mut d = Data::new(data.clone());
                assert!(d.order_statistic(0).is_nan(), "{shape}/{n}: order 0");
                assert!(d.order_statistic(n + 1).is_nan(), "{shape}/{n}: order n+1");

                // median, including the even case that averages two selections
                let expected_median = if n % 2 == 1 {
                    sorted[n / 2]
                } else {
                    (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
                };
                let mut d = Data::new(data.clone());
                assert_eq!(
                    OrderStatistics::median(&mut d),
                    expected_median,
                    "{shape}/{n}: median"
                );

                // the quantile endpoints, and that interior values stay in range
                let mut d = Data::new(data.clone());
                assert_eq!(d.quantile(0.0), sorted[0], "{shape}/{n}: quantile(0)");
                assert_eq!(d.quantile(1.0), sorted[n - 1], "{shape}/{n}: quantile(1)");
                for &tau in &[0.1, 0.25, 0.5, 0.75, 0.9] {
                    let q = d.quantile(tau);
                    assert!(
                        q >= sorted[0] && q <= sorted[n - 1],
                        "{shape}/{n}: quantile({tau}) = {q} outside [{}, {}]",
                        sorted[0],
                        sorted[n - 1]
                    );
                }
            }
        }
    }

    /// Selection does not depend on input order.
    #[test]
    #[cfg(all(feature = "std", feature = "rand"))]
    fn test_order_statistic_is_permutation_invariant() {
        use ::rand::{RngExt, SeedableRng, rngs::StdRng};

        let mut rng = StdRng::seed_from_u64(0xA11CE);
        let base: Vec<f64> = (0..64).map(|_| rng.random_range(-40.0..40.0)).collect();
        let mut sorted = base.clone();
        sorted.sort_by(f64::total_cmp);

        // a few fixed permutations, including the ones that hit the fast paths
        let mut rotated = base.clone();
        rotated.rotate_left(23);
        let mut ascending = base.clone();
        ascending.sort_by(f64::total_cmp);
        let mut descending = ascending.clone();
        descending.reverse();

        for variant in [base.clone(), rotated, ascending, descending] {
            for order in 1..=variant.len() {
                let mut d = Data::new(variant.clone());
                assert_eq!(d.order_statistic(order), sorted[order - 1]);
            }
        }
    }

    /// The old hand-rolled Numerical Recipes quickselect walked off the end of
    /// the array when the slice contained NaN (statrs-dev/statrs#163).
    #[test]
    fn test_order_statistics_with_nan_do_not_panic() {
        // Arrays rather than `Vec`, so the test also builds under `no_std`.
        let mut v: [f64; 1000] = core::array::from_fn(|i| (i as f64 * 0.7).sin());
        for i in (0..1000).step_by(7) {
            v[i] = f64::NAN;
        }
        let mut d = Data::new(v);
        // the values themselves are unspecified once NaN is present; what is
        // being pinned is that these terminate in bounds
        let _ = d.quantile(0.5);
        let _ = d.quantile(0.0);
        let _ = d.quantile(1.0);
        let _ = d.order_statistic(500);
        let _ = OrderStatistics::median(&mut d);
        let _ = d.interquartile_range();

        // all-NaN, and NaN at the exact positions the old partition tripped on
        let mut all_nan = Data::new([f64::NAN; 64]);
        let _ = all_nan.quantile(0.5);
        for pos in [0usize, 1, 31, 62, 63] {
            let mut v = [0.0f64; 64];
            v[pos] = f64::NAN;
            let _ = Data::new(v).quantile(0.5);
        }
    }

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
