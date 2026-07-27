//! Single-pass (online) statistical accumulators.
//!
//! Unlike [`crate::statistics::Statistics`], which consumes its input once
//! per method call, these types accumulate observations one at a time via
//! [`OnlineMoments::push`] and can be read from repeatedly, or composed with
//! [`Accumulate`] to share a single fold pass across several statistics.

#[cfg(not(feature = "std"))]
use num_traits::Float as _;

/// Single-pass accumulator for central moments via Welford's online algorithm.
///
/// `ORDER` controls which moments are tracked:
/// - `1` mean
/// - `2` + variance
/// - `3` + skewness
/// - above does not presently implement further moments
///
/// Moments are accumulated for `x - offset`, where `offset` is the first value
/// pushed. Central moments are invariant under that shift, and it is what makes
/// the accumulator usable on data with a large offset: Welford's `mean +=
/// delta / n` update cannot represent a small increment against a large running
/// mean, so `1e12 + U(0, 1)` came out with `2.5e-4` relative error in the
/// variance. Referring everything to the first observation keeps the magnitudes
/// small and brings that to `5e-15` at no cost (statrs-dev/statrs#376).
pub struct OnlineMoments<const ORDER: usize> {
    pub count: u64,
    /// The first value pushed; `m` holds the moments of `x - offset`.
    offset: f64,
    m: [f64; ORDER],
}

impl<const ORDER: usize> Default for OnlineMoments<ORDER> {
    fn default() -> Self {
        Self {
            count: 0,
            offset: 0.0,
            m: [0.0; ORDER],
        }
    }
}

impl OnlineMoments<2> {
    /// Returns the mean, or `None` if no observations have been pushed.
    pub fn mean(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.offset + self.m[0])
        }
    }

    /// Returns the sample variance (normalised by `n - 1`), or `None` if
    /// fewer than two observations have been pushed.
    pub fn variance(&self) -> Option<f64> {
        if self.count < 2 {
            None
        } else {
            Some(self.m[1] / (self.count - 1) as f64)
        }
    }

    /// Returns the sample standard deviation, or `None` if fewer than two
    /// observations have been pushed.
    pub fn std_dev(&self) -> Option<f64> {
        self.variance().map(f64::sqrt)
    }

    /// Returns the population variance (normalised by `n`), or `None` if no
    /// observations have been pushed.
    pub fn population_variance(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.m[1] / self.count as f64)
        }
    }

    /// Returns the population standard deviation, or `None` if no
    /// observations have been pushed.
    pub fn population_std_dev(&self) -> Option<f64> {
        self.population_variance().map(f64::sqrt)
    }
}

impl OnlineMoments<3> {
    /// Returns the mean, or `None` if no observations have been pushed.
    pub fn mean(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.offset + self.m[0])
        }
    }

    /// Returns the sample variance (normalised by `n - 1`), or `None` if
    /// fewer than two observations have been pushed.
    pub fn variance(&self) -> Option<f64> {
        if self.count < 2 {
            None
        } else {
            Some(self.m[1] / (self.count - 1) as f64)
        }
    }

    /// Returns the sample standard deviation, or `None` if fewer than two
    /// observations have been pushed.
    pub fn std_dev(&self) -> Option<f64> {
        self.variance().map(f64::sqrt)
    }

    /// Returns the population variance (normalised by `n`), or `None` if no
    /// observations have been pushed.
    pub fn population_variance(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.m[1] / self.count as f64)
        }
    }

    /// Returns the population standard deviation, or `None` if no
    /// observations have been pushed.
    pub fn population_std_dev(&self) -> Option<f64> {
        self.population_variance().map(f64::sqrt)
    }

    /// Returns the skewness, or `None` if fewer than two observations have
    /// been pushed.
    pub fn skewness(&self) -> Option<f64> {
        if self.count < 2 {
            return None;
        }
        let n = self.count as f64;
        let m2_mean = self.m[1] / n;
        let m3_mean = self.m[2] / n;
        let denom = m2_mean.powf(1.5);
        if denom == 0.0 {
            Some(0.0)
        } else {
            Some(m3_mean / denom)
        }
    }
}

impl<const ORDER: usize> OnlineMoments<ORDER> {
    /// Merges two accumulators as if all observations had been pushed into
    /// one, using the pairwise update of Chan, Golub & LeVeque (extended to
    /// the third moment by Pébay, 2008).
    ///
    /// Besides combining accumulators built in parallel, folding into several
    /// accumulators and merging at the end is also slightly *better*
    /// conditioned than one long Welford chain, since each chain's rounding
    /// errors accumulate over fewer updates.
    ///
    /// ```
    /// use statrs::statistics::OnlineVariance;
    /// use statrs::statistics::Accumulate;
    /// let a = [1.0_f64, 2.0].iter().copied().fold(OnlineVariance::default(), OnlineVariance::push);
    /// let b = [3.0_f64, 4.0].iter().copied().fold(OnlineVariance::default(), OnlineVariance::push);
    /// let all = [1.0_f64, 2.0, 3.0, 4.0].iter().copied().fold(OnlineVariance::default(), OnlineVariance::push);
    /// assert_eq!(a.merge(b).variance(), all.variance());
    /// ```
    pub fn merge(self, other: Self) -> Self {
        if other.count == 0 {
            return self;
        }
        if self.count == 0 {
            return other;
        }
        let na = self.count as f64;
        let nb = other.count as f64;
        let n = na + nb;
        // The two accumulators generally have different offsets, so re-express
        // `other`'s mean in `self`'s frame. Grouping the two differences
        // separately keeps this accurate when the offsets are close, which is
        // the common case (both are data values).
        let delta = (other.offset - self.offset) + (other.m[0] - self.m[0]);

        let mut m = [0.0; ORDER];
        m[0] = self.m[0] + delta * nb / n;
        if let (Some(&m2a), Some(&m2b)) = (self.m.get(1), other.m.get(1)) {
            if let (Some(&m3a), Some(&m3b)) = (self.m.get(2), other.m.get(2)) {
                m[2] = m3a
                    + m3b
                    + delta * delta * delta * na * nb * (na - nb) / (n * n)
                    + 3.0 * delta * (na * m2b - nb * m2a) / n;
            }
            m[1] = m2a + m2b + delta * delta * na * nb / n;
        }

        Self {
            count: self.count + other.count,
            offset: self.offset,
            m,
        }
    }
}

/// Single-pass mean accumulator (alias of [`OnlineMoments<2>`]).
pub type OnlineMean = OnlineMoments<2>;

/// Single-pass mean and variance accumulator (alias of [`OnlineMoments<2>`]).
pub type OnlineVariance = OnlineMoments<2>;

/// Single-pass mean, variance, and skewness accumulator (alias of [`OnlineMoments<3>`]).
pub type OnlineSkewness = OnlineMoments<3>;

impl<const ORDER: usize> crate::statistics::Accumulate for OnlineMoments<ORDER> {
    /// Folds one observation into the moments.
    ///
    /// ```
    /// use statrs::statistics::OnlineVariance;
    /// use statrs::statistics::Accumulate;
    /// let s = [1.0_f64, 2.0, 3.0].iter().copied()
    ///     .fold(OnlineVariance::default(), OnlineVariance::push);
    /// ```
    fn push(mut self, x: f64) -> Self {
        if self.count == 0 {
            self.offset = x;
        }
        self.count += 1;
        let n = self.count as f64;
        // work relative to the first observation; see the type-level docs
        let x = x - self.offset;

        // Welford / Pebay (2008) central moment update. Update order: M3
        // before M2 before mean; each step uses the previous observation's
        // lower-order accumulators.
        let delta = x - self.m[0];
        let delta_n = delta / n;
        let new_mean = self.m[0] + delta_n;
        let delta2 = x - new_mean;

        if let Some(&old_m2) = self.m.get(1) {
            if let Some(inc) = self.m.get(2).map(|_| {
                delta * (delta_n * delta_n) * (n - 1.0) * (n - 2.0) - 3.0 * delta_n * old_m2
            }) {
                self.m[2] += inc;
            }
            self.m[1] += delta * delta2;
        }

        self.m[0] = new_mean;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{prec, statistics::Accumulate};

    #[test]
    fn single_element() {
        let s = OnlineMoments::<2>::default().push(5.0);
        assert_eq!(s.count, 1);
        assert_eq!(s.mean(), Some(5.0));
        assert_eq!(s.variance(), None);
        assert_eq!(s.std_dev(), None);
        assert_eq!(s.population_variance(), Some(0.0));
        assert_eq!(s.population_std_dev(), Some(0.0));
    }

    #[test]
    fn known_dataset() {
        // [2,4,4,4,5,5,7,9]: mean=5.0, M2=32, sample variance=32/7,
        // population variance=32/8=4.0
        let data = [2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let s = data
            .iter()
            .copied()
            .fold(OnlineMoments::<2>::default(), OnlineMoments::push);
        prec::assert_abs_diff_eq!(s.mean().unwrap(), 5.0);
        prec::assert_abs_diff_eq!(s.variance().unwrap(), 32.0 / 7.0);
        prec::assert_abs_diff_eq!(s.std_dev().unwrap(), (32.0_f64 / 7.0).sqrt());
        prec::assert_abs_diff_eq!(s.population_variance().unwrap(), 4.0);
        prec::assert_abs_diff_eq!(s.population_std_dev().unwrap(), 2.0);
    }

    /// Welford's `mean += delta / n` cannot represent a small increment against
    /// a large running mean, so `1e12 + U(0, 1)` used to come out with `2.5e-4`
    /// relative error in the variance (statrs-dev/statrs#376). Accumulating
    /// relative to the first observation keeps the magnitudes small.
    ///
    /// The offsets and the step are powers of two and the values need only 51
    /// bits, so every sample is exactly representable at every offset and the
    /// reference variance is exact - otherwise quantisation at `2^40` would
    /// swamp what is being measured.
    #[test]
    fn variance_is_accurate_for_data_with_a_large_offset() {
        const STEP: f64 = 1.0 / 1024.0; // 2^-10
        const PERIOD: usize = 1024;
        const REPEATS: usize = 100;
        let n = (PERIOD * REPEATS) as f64;
        // population variance of {0, STEP, ..., 1023 * STEP}
        let pop = ((PERIOD * PERIOD - 1) as f64 / 12.0) * STEP * STEP;
        let expected_variance = pop * n / (n - 1.0);
        let expected_mean_offset = (PERIOD - 1) as f64 / 2.0 * STEP;

        for exp in [0i32, 10, 20, 30, 40] {
            let offset = f64::powi(2.0, exp);
            let data: Vec<f64> = (0..PERIOD * REPEATS)
                .map(|i| offset + (i % PERIOD) as f64 * STEP)
                .collect();
            let s = data
                .iter()
                .copied()
                .fold(OnlineMoments::<2>::default(), OnlineMoments::push);
            prec::assert_relative_eq!(
                s.variance().unwrap(),
                expected_variance,
                epsilon = 0.0,
                max_relative = 1e-13
            );
            prec::assert_relative_eq!(
                s.mean().unwrap(),
                offset + expected_mean_offset,
                epsilon = 0.0,
                // the accumulated error lives in the shifted mean, so it is
                // largest relative to the total when the offset is small
                max_relative = 1e-14
            );
        }
    }

    /// The shift is per-accumulator, so `merge` has to reconcile two different
    /// offsets; check it still matches a single chain on offset data.
    #[test]
    fn merge_reconciles_different_offsets() {
        let a: Vec<f64> = (0..500).map(|i| 1e12 + i as f64 * 1e-3).collect();
        let b: Vec<f64> = (0..500).map(|i| 1e12 + 5.0 + i as f64 * 1e-3).collect();
        let ma = a
            .iter()
            .copied()
            .fold(OnlineMoments::<2>::default(), OnlineMoments::push);
        let mb = b
            .iter()
            .copied()
            .fold(OnlineMoments::<2>::default(), OnlineMoments::push);
        let whole: Vec<f64> = a.iter().chain(b.iter()).copied().collect();
        let mw = whole
            .iter()
            .copied()
            .fold(OnlineMoments::<2>::default(), OnlineMoments::push);
        prec::assert_relative_eq!(
            ma.merge(mb).variance().unwrap(),
            mw.variance().unwrap(),
            epsilon = 0.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn nan_propagates() {
        let s = [1.0_f64, f64::NAN]
            .iter()
            .copied()
            .fold(OnlineMoments::<2>::default(), OnlineMoments::push);
        assert!(s.mean().unwrap().is_nan());
        assert!(s.variance().unwrap().is_nan());
    }

    #[test]
    fn skewness_known_dataset() {
        // [2,4,4,4,5,5,7,9]: skewness = (M3/n) / (M2/n)^1.5
        // M2 = 32, M3 = 42, n = 8 => (42/8) / (32/8)^1.5 = 5.25 / 8.0 = 0.65625
        let data = [2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let s = data
            .iter()
            .copied()
            .fold(OnlineMoments::<3>::default(), OnlineMoments::push);
        prec::assert_abs_diff_eq!(s.skewness().unwrap(), 0.65625);
    }

    #[test]
    fn merge_matches_single_accumulator() {
        let data = [3.0_f64, -1.0, 4.0, 1.0, -5.0, 9.0, 2.0, 6.0, -3.0];
        for split in 0..=data.len() {
            let (lo, hi) = data.split_at(split);
            let a = lo
                .iter()
                .copied()
                .fold(OnlineMoments::<3>::default(), OnlineMoments::push);
            let b = hi
                .iter()
                .copied()
                .fold(OnlineMoments::<3>::default(), OnlineMoments::push);
            let merged = a.merge(b);
            let whole = data
                .iter()
                .copied()
                .fold(OnlineMoments::<3>::default(), OnlineMoments::push);
            assert_eq!(merged.count, whole.count);
            prec::assert_relative_eq!(
                merged.mean().unwrap(),
                whole.mean().unwrap(),
                max_relative = 1e-14
            );
            prec::assert_relative_eq!(
                merged.variance().unwrap(),
                whole.variance().unwrap(),
                max_relative = 1e-13
            );
            prec::assert_relative_eq!(
                merged.skewness().unwrap(),
                whole.skewness().unwrap(),
                max_relative = 1e-12
            );
        }
    }

    #[test]
    fn merge_with_empty_is_identity() {
        let a = [1.0_f64, 2.0, 3.0]
            .iter()
            .copied()
            .fold(OnlineMoments::<2>::default(), OnlineMoments::push);
        let empty = OnlineMoments::<2>::default();
        assert_eq!(a.merge(empty).variance(), Some(1.0));
        let a = [1.0_f64, 2.0, 3.0]
            .iter()
            .copied()
            .fold(OnlineMoments::<2>::default(), OnlineMoments::push);
        let empty = OnlineMoments::<2>::default();
        assert_eq!(empty.merge(a).variance(), Some(1.0));
    }

    #[test]
    fn order_3_mean_and_variance_match_order_2() {
        let data = [2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let s2 = data
            .iter()
            .copied()
            .fold(OnlineMoments::<2>::default(), OnlineMoments::push);
        let s3 = data
            .iter()
            .copied()
            .fold(OnlineMoments::<3>::default(), OnlineMoments::push);
        prec::assert_abs_diff_eq!(s2.mean().unwrap(), s3.mean().unwrap());
        prec::assert_abs_diff_eq!(s2.variance().unwrap(), s3.variance().unwrap());
    }
}

#[cfg(test)]
mod accumulate_tests {
    use super::*;
    use crate::statistics::Accumulate;

    #[test]
    fn online_moments_impl_accumulate() {
        let s: OnlineMoments<2> = [1.0_f64, 2.0, 3.0]
            .iter()
            .copied()
            .fold(Default::default(), Accumulate::push);
        assert_eq!(s.mean(), Some(2.0));
    }

    #[test]
    fn tuple_composition_matches_separate_folds() {
        let data = [3.0_f64, -1.0, 4.0, 1.0, -5.0, 9.0];

        let (skew, var): (OnlineSkewness, OnlineVariance) = data
            .iter()
            .copied()
            .fold(Default::default(), Accumulate::push);

        let skew_alone = data
            .iter()
            .copied()
            .fold(OnlineSkewness::default(), OnlineSkewness::push);
        let var_alone = data
            .iter()
            .copied()
            .fold(OnlineVariance::default(), OnlineVariance::push);

        assert_eq!(skew.skewness(), skew_alone.skewness());
        assert_eq!(var.variance(), var_alone.variance());
    }
}
