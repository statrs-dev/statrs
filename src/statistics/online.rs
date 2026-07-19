//! Single-pass (online) statistical accumulators.
//!
//! Unlike [`crate::statistics::Statistics`], which consumes its input once
//! per method call, these types accumulate observations one at a time via
//! [`OnlineMoments::push`] and can be read from repeatedly, or composed with
//! [`Accumulate`] to share a single fold pass across several statistics.

/// Single-pass accumulator for central moments via Welford's online algorithm.
///
/// `ORDER` controls which moments are tracked:
/// - `1` mean
/// - `2` + variance
/// - `3` + skewness
/// - above does not presently implement further moments
pub struct OnlineMoments<const ORDER: usize> {
    pub count: u64,
    m: [f64; ORDER],
}

impl<const ORDER: usize> Default for OnlineMoments<ORDER> {
    fn default() -> Self {
        Self {
            count: 0,
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
            Some(self.m[0])
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
            Some(self.m[0])
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
        self.count += 1;
        let n = self.count as f64;

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
