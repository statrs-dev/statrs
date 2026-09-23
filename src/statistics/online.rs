//! Single-pass (online) statistical accumulators.
//!
//! Unlike [`crate::statistics::Statistics`], which consumes its input once
//! per method call, these types accumulate observations one at a time via
//! [`Accumulate::push`] and can be read from repeatedly, or composed with
//! [`Accumulate`] to share a single fold pass across several statistics.

#[cfg(not(feature = "std"))]
use num_traits::Float as _;

use crate::statistics::Accumulate;

/// Statistics that can be accumulated with [`Accumulate`].
pub trait OnlineMoment {
    /// `ORDER` controls which moments are tracked:
    /// - `1`: count + mean
    /// - `2`: `1` + variance
    /// - `3`: `2` + skewness
    /// - values above: not presently implemented
    const ORDER: usize;
    // This method is not meant to be called from user, so we can
    // have less type restrictions. In theory, we can even restrict
    // MS only to the tuple types that include Self.
    /// Convert an [`Accumulate`] to this stat. In most cases, you
    /// are not meant to use this method. Use [`Accumulate::get`]
    /// instead.
    fn from_acc<MS: OnlineMoments>(acc: &Accumulate<MS>) -> Self;
}
/// A collection of [`OnlineMoment`]s.
///
/// This trait is implemented for up to 8-tuple of [`OnlineMoment`]s.
pub trait OnlineMoments: Sized {
    // This will always be compiled into constants.
    // In the future, this can be directly an associated const.
    /// The maximal order of collected [`OnlineMoment`]s.
    fn order() -> usize;
    /// Convert an [`Accumulate`] to the stats. In most cases, you
    /// are not meant to use this method. Use [`Accumulate::get`]
    /// instead.
    fn from_acc(acc: &Accumulate<Self>) -> Self;
}
macro_rules! impl_online_moments_for_tuple {
    ($($M: ident),+) => {
        impl<$($M: OnlineMoment),+> OnlineMoments for ($($M),+,) {
            fn order() -> usize {
                let mut val = 0;
                $(
                    val = ::core::cmp::max(val, <$M as OnlineMoment>::ORDER);
                )+
                val
            }

            fn from_acc(acc: &Accumulate<Self>) -> Self {
                ($(<$M as OnlineMoment>::from_acc(acc)),+,)
            }
        }
    };
}
impl_online_moments_for_tuple!(M1);
impl_online_moments_for_tuple!(M1, M2);
impl_online_moments_for_tuple!(M1, M2, M3);
impl_online_moments_for_tuple!(M1, M2, M3, M4);
impl_online_moments_for_tuple!(M1, M2, M3, M4, M5);
impl_online_moments_for_tuple!(M1, M2, M3, M4, M5, M6);
impl_online_moments_for_tuple!(M1, M2, M3, M4, M5, M6, M7);
impl_online_moments_for_tuple!(M1, M2, M3, M4, M5, M6, M7, M8);

/// Contains the mean, or `None` if no observations have been pushed.
///
/// Can be used with [`Accumulate`].
pub struct OnlineMean(pub Option<f64>);
impl OnlineMoment for OnlineMean {
    const ORDER: usize = 1;
    fn from_acc<MS: OnlineMoments>(acc: &Accumulate<MS>) -> Self {
        if acc.count == 0 {
            Self(None)
        } else {
            Self(Some(acc.m[0]))
        }
    }
}

/// Contains the sample variance (normalised by `n - 1`), or `None` if
/// fewer than two observations have been pushed.
///
/// Can be used with [`Accumulate`].
pub struct OnlineVariance(pub Option<f64>);
impl OnlineMoment for OnlineVariance {
    const ORDER: usize = 2;
    fn from_acc<MS: OnlineMoments>(acc: &Accumulate<MS>) -> Self {
        if acc.count < 2 {
            Self(None)
        } else {
            Self(Some(acc.m[1] / (acc.count - 1) as f64))
        }
    }
}

/// Contains the sample standard deviation, or `None` if fewer than two
/// observations have been pushed.
///
/// Can be used with [`Accumulate`].
pub struct OnlineStdDev(pub Option<f64>);
impl OnlineMoment for OnlineStdDev {
    const ORDER: usize = OnlineVariance::ORDER;
    fn from_acc<MS: OnlineMoments>(acc: &Accumulate<MS>) -> Self {
        let OnlineVariance(variance) = OnlineVariance::from_acc(acc);
        Self(variance.map(f64::sqrt))
    }
}

/// Contains the population variance (normalised by `n`), or `None` if no
/// observations have been pushed.
///
/// Can be used with [`Accumulate`].
pub struct OnlinePopulationVariance(pub Option<f64>);
impl OnlineMoment for OnlinePopulationVariance {
    const ORDER: usize = 2;
    fn from_acc<MS: OnlineMoments>(acc: &Accumulate<MS>) -> Self {
        if acc.count == 0 {
            Self(None)
        } else {
            Self(Some(acc.m[1] / acc.count as f64))
        }
    }
}

/// Contains the population standard deviation, or `None` if no
/// observations have been pushed.
///
/// Can be used with [`Accumulate`].
pub struct OnlinePopulationStdDev(pub Option<f64>);
impl OnlineMoment for OnlinePopulationStdDev {
    const ORDER: usize = OnlinePopulationVariance::ORDER;
    fn from_acc<MS: OnlineMoments>(acc: &Accumulate<MS>) -> Self {
        let OnlinePopulationVariance(population_variance) = OnlinePopulationVariance::from_acc(acc);
        Self(population_variance.map(f64::sqrt))
    }
}

/// Contains the skewness, or `None` if fewer than two observations have
/// been pushed.
///
/// Can be used with [`Accumulate`].
pub struct OnlineSkewness(pub Option<f64>);
impl OnlineMoment for OnlineSkewness {
    const ORDER: usize = 3;
    fn from_acc<MS: OnlineMoments>(acc: &Accumulate<MS>) -> Self {
        if acc.count < 2 {
            return Self(None);
        }
        let n = acc.count as f64;
        let m2_mean = acc.m[1] / n;
        let m3_mean = acc.m[2] / n;
        let denom = m2_mean.powf(1.5);
        if denom == 0.0 {
            Self(Some(0.0))
        } else {
            Self(Some(m3_mean / denom))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{prec, statistics::Accumulate};

    #[test]
    fn single_element() {
        let acc = Accumulate::default().push(5.0);
        let (
            OnlineMean(mean),
            OnlineVariance(variance),
            OnlineStdDev(std_dev),
            OnlinePopulationVariance(population_variance),
            OnlinePopulationStdDev(population_std_dev),
        ) = acc.get();
        assert_eq!(acc.count, 1);
        assert_eq!(mean, Some(5.0));
        assert_eq!(variance, None);
        assert_eq!(std_dev, None);
        assert_eq!(population_variance, Some(0.0));
        assert_eq!(population_std_dev, Some(0.0));
    }

    #[test]
    fn known_dataset() {
        // [2,4,4,4,5,5,7,9]: mean=5.0, M2=32, sample variance=32/7,
        // population variance=32/8=4.0
        let data = [2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let (
            OnlineMean(mean),
            OnlineVariance(variance),
            OnlineStdDev(std_dev),
            OnlinePopulationVariance(population_variance),
            OnlinePopulationStdDev(population_std_dev),
        ) = data
            .iter()
            .copied()
            .fold(Accumulate::default(), Accumulate::push)
            .get();
        prec::assert_abs_diff_eq!(mean.unwrap(), 5.0);
        prec::assert_abs_diff_eq!(variance.unwrap(), 32.0 / 7.0);
        prec::assert_abs_diff_eq!(std_dev.unwrap(), (32.0_f64 / 7.0).sqrt());
        prec::assert_abs_diff_eq!(population_variance.unwrap(), 4.0);
        prec::assert_abs_diff_eq!(population_std_dev.unwrap(), 2.0);
    }

    #[test]
    fn nan_propagates() {
        let (OnlineMean(mean), OnlineVariance(variance)) = [1.0_f64, f64::NAN]
            .iter()
            .copied()
            .fold(Accumulate::default(), Accumulate::push)
            .get();
        assert!(mean.unwrap().is_nan());
        assert!(variance.unwrap().is_nan());
    }

    #[test]
    fn skewness_known_dataset() {
        // [2,4,4,4,5,5,7,9]: skewness = (M3/n) / (M2/n)^1.5
        // M2 = 32, M3 = 42, n = 8 => (42/8) / (32/8)^1.5 = 5.25 / 8.0 = 0.65625
        let data = [2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let (OnlineSkewness(skewness),) = data
            .iter()
            .copied()
            .fold(Accumulate::default(), Accumulate::push)
            .get();
        prec::assert_abs_diff_eq!(skewness.unwrap(), 0.65625);
    }

    #[test]
    fn order_3_mean_and_variance_match_order_2() {
        fn get_order<MS: OnlineMoments>(_acc: &Accumulate<MS>) -> usize {
            MS::order()
        }

        let data = [2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let acc2 = data
            .iter()
            .copied()
            .fold(Accumulate::default(), Accumulate::push);
        assert_eq!(get_order(&acc2), 2);
        let (OnlineMean(s2_mean), OnlineVariance(s2_variance)) = acc2.get();
        let acc3 = data
            .iter()
            .copied()
            .fold(Accumulate::default(), Accumulate::push);
        assert_eq!(get_order(&acc3), 3);
        let (OnlineMean(s3_mean), OnlineVariance(s3_variance), OnlineSkewness(_)) = acc3.get();
        prec::assert_abs_diff_eq!(s2_mean.unwrap(), s3_mean.unwrap());
        prec::assert_abs_diff_eq!(s2_variance.unwrap(), s3_variance.unwrap());
    }
}

#[cfg(test)]
mod accumulate_tests {
    use super::*;
    use crate::statistics::Accumulate;

    #[test]
    fn online_moments_impl_accumulate() {
        let (OnlineMean(mean),) = [1.0_f64, 2.0, 3.0]
            .iter()
            .copied()
            .fold(Accumulate::default(), Accumulate::push)
            .get();
        assert_eq!(mean, Some(2.0));
    }

    #[test]
    fn tuple_composition_matches_separate_folds() {
        let data = [3.0_f64, -1.0, 4.0, 1.0, -5.0, 9.0];

        let (OnlineSkewness(skewness), OnlineVariance(variance)) = data
            .iter()
            .copied()
            .fold(Accumulate::default(), Accumulate::push)
            .get();

        let (OnlineSkewness(skewness_alone),) = data
            .iter()
            .copied()
            .fold(Accumulate::default(), Accumulate::push)
            .get();
        let (OnlineVariance(variance_alone),) = data
            .iter()
            .copied()
            .fold(Accumulate::default(), Accumulate::push)
            .get();

        assert_eq!(skewness, skewness_alone);
        assert_eq!(variance, variance_alone);
    }
}
