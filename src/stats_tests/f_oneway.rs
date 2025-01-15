//! Provides the [one-way ANOVA F-test](https://en.wikipedia.org/wiki/One-way_analysis_of_variance)
//! and related functions

use crate::distribution::{ContinuousCDF, FisherSnedecor};
use crate::stats_tests::NaNPolicy;

/// Represents the errors that occur when computing the f_oneway function
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum FOneWayTestError {
    /// must be at least two samples
    NotEnoughSamples,
    /// one sample must be length greater than 1
    SampleTooSmall,
    /// samples must not contain all of the same values
    SampleContainsSameConstants,
    /// samples can not contain NaN when `nan_policy` is set to `NaNPolicy::Error`
    SampleContainsNaN,
}

impl std::fmt::Display for FOneWayTestError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            FOneWayTestError::NotEnoughSamples => write!(f, "must be at least two samples"),
            FOneWayTestError::SampleTooSmall => {
                write!(f, "one sample must be length greater than 1")
            }
            FOneWayTestError::SampleContainsSameConstants => {
                write!(f, "samples must not contain all of the same values")
            }
            FOneWayTestError::SampleContainsNaN => {
                write!(
                    f,
                    "samples can not contain NaN when `nan_policy` is set to `NaNPolicy::Error`"
                )
            }
        }
    }
}

impl std::error::Error for FOneWayTestError {}

/// Perform a one-way Analysis of Variance (ANOVA) F-test
///
/// Takes in a set (outer vector) of samples (inner vector) and returns the F-statistic and p-value
///
/// # Remarks
/// Implementation based on [statsdirect](https://www.statsdirect.com/help/analysis_of_variance/one_way.htm)
/// and [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html#scipy.stats.f_oneway)
///
/// `samples` needs to be mutable in case needing to filter out NaNs for NaNPolicy::Emit
///
/// # Examples
///
/// ```
/// use statrs::stats_tests::f_oneway::f_oneway;
/// use statrs::stats_tests::NaNPolicy;
///
/// // based on wikipedia example
/// let a1 = Vec::from([6f64, 8f64, 4f64, 5f64, 3f64, 4f64]);
/// let a2 = Vec::from([8f64, 12f64, 9f64, 11f64, 6f64, 8f64]);
/// let a3 = Vec::from([13f64, 9f64, 11f64, 8f64, 7f64, 12f64]);
/// let sample_input = Vec::from([a1, a2, a3]);
/// let (statistic, pvalue) = f_oneway(sample_input, NaNPolicy::Error).unwrap(); // (9.3, 0.002)
/// ```
pub fn f_oneway(
    mut samples: Vec<Vec<f64>>,
    nan_policy: NaNPolicy,
) -> Result<(f64, f64), FOneWayTestError> {
    let k = samples.len();

    // initial input validation
    if k < 2 {
        return Err(FOneWayTestError::NotEnoughSamples);
    }

    let has_nans = samples.iter().flatten().any(|x| x.is_nan());
    if has_nans {
        match nan_policy {
            NaNPolicy::Propogate => {
                return Ok((f64::NAN, f64::NAN));
            }
            NaNPolicy::Error => {
                return Err(FOneWayTestError::SampleContainsNaN);
            }
            NaNPolicy::Emit => {
                samples = samples
                    .into_iter()
                    .map(|v| v.into_iter().filter(|x| !x.is_nan()).collect::<Vec<_>>())
                    .collect::<Vec<_>>();
            }
        }
    }

    // do remaining input validation after potential subset from Emit
    let n_i: Vec<usize> = samples.iter().map(|v| v.len()).collect();
    if !n_i.iter().all(|x| *x >= 1) || !n_i.iter().any(|x| *x >= 2) {
        return Err(FOneWayTestError::SampleTooSmall);
    }

    if samples.iter().any(|v| {
        if v.len() > 1 {
            let mut it = v.iter();
            let first = it.next().unwrap();
            it.all(|x| x == first)
        } else {
            false
        }
    }) {
        return Err(FOneWayTestError::SampleContainsSameConstants);
    }

    let n = n_i.iter().sum::<usize>();
    let g = samples.iter().flatten().sum::<f64>();

    let tsq = samples
        .iter()
        .map(|v| v.iter().sum::<f64>().powi(2) / v.len() as f64)
        .sum::<f64>();
    let ysq = samples.iter().flatten().map(|x| x.powi(2)).sum::<f64>();

    // Sum of Squares (SS) and Mean Square (MS) between and within groups
    let sst = tsq - (g.powi(2) / n as f64);
    let mst = sst / (k - 1) as f64;

    let sse = ysq - tsq;
    let mse = sse / (n - k) as f64;

    let fstat = mst / mse;

    // degrees of freedom for between groups (t) and within groups (e)
    let dft = (k - 1) as f64;
    let dfe = (n - k) as f64;
    // k >= 2 meaning dft = (k-1) > 0 or Err(NotEnoughSamples)
    // one group must be at least 2 and all other groups must be at least 1 or Err(SampleTooSmall)
    // meaning that the minimum value of n will always be at least one greater than k so dfe must
    // be > 0
    let f_dist = FisherSnedecor::new(dft, dfe).expect("degrees of freedom should always be >0 ");
    let pvalue = 1.0 - f_dist.cdf(fstat);

    Ok((fstat, pvalue))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prec;

    #[test]
    fn test_scipy_example() {
        // Test against the scipy example
        // https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html#scipy.stats.f_oneway
        let tillamook = Vec::from([
            0.0571, 0.0813, 0.0831, 0.0976, 0.0817, 0.0859, 0.0735, 0.0659, 0.0923, 0.0836,
        ]);
        let newport = Vec::from([
            0.0873, 0.0662, 0.0672, 0.0819, 0.0749, 0.0649, 0.0835, 0.0725,
        ]);
        let petersburg = Vec::from([0.0974, 0.1352, 0.0817, 0.1016, 0.0968, 0.1064, 0.105]);
        let magadan = Vec::from([
            0.1033, 0.0915, 0.0781, 0.0685, 0.0677, 0.0697, 0.0764, 0.0689,
        ]);
        let tvarminne = Vec::from([0.0703, 0.1026, 0.0956, 0.0973, 0.1039, 0.1045]);
        let sample_input = Vec::from([tillamook, newport, petersburg, magadan, tvarminne]);
        let (statistic, pvalue) = f_oneway(sample_input, NaNPolicy::Error).unwrap();

        assert!(prec::almost_eq(statistic, 7.121019471642447, 1e-1));
        assert!(prec::almost_eq(pvalue, 0.0002812242314534544, 1e-12));
    }
    #[test]
    fn test_nan_in_data_w_emit() {
        // same as scipy example above with NaNs added should give same result
        let tillamook = Vec::from([
            0.0571,
            0.0813,
            0.0831,
            0.0976,
            0.0817,
            0.0859,
            0.0735,
            0.0659,
            0.0923,
            0.0836,
            f64::NAN,
        ]);
        let newport = Vec::from([
            0.0873, 0.0662, 0.0672, 0.0819, 0.0749, 0.0649, 0.0835, 0.0725,
        ]);
        let petersburg = Vec::from([0.0974, 0.1352, 0.0817, 0.1016, 0.0968, 0.1064, 0.105]);
        let magadan = Vec::from([
            0.1033,
            0.0915,
            0.0781,
            0.0685,
            0.0677,
            0.0697,
            0.0764,
            0.0689,
            f64::NAN,
        ]);
        let tvarminne = Vec::from([0.0703, 0.1026, 0.0956, 0.0973, 0.1039, 0.1045]);
        let sample_input = Vec::from([tillamook, newport, petersburg, magadan, tvarminne]);
        let (statistic, pvalue) = f_oneway(sample_input, NaNPolicy::Emit).unwrap();

        assert!(prec::almost_eq(statistic, 7.121019471642447, 1e-1));
        assert!(prec::almost_eq(pvalue, 0.0002812242314534544, 1e-12));
    }
    #[test]
    fn test_group_length_one_ok() {
        // group length 1 doesn't result in error
        let group1 = Vec::from([0.5]);
        let group2 = Vec::from([0.25, 0.75]);
        let sample_input = Vec::from([group1, group2]);
        let (statistic, pvalue) = f_oneway(sample_input, NaNPolicy::Propogate).unwrap();
        assert!(prec::almost_eq(statistic, 0.0, 1e-1));
        assert!(prec::almost_eq(pvalue, 1.0, 1e-12));
    }
    #[test]
    fn test_nan_in_data_w_propogate() {
        let group1 = Vec::from([0.0571, 0.0813, f64::NAN, 0.0836]);
        let group2 = Vec::from([0.0873, 0.0662, 0.0672, 0.0819, 0.0749]);
        let sample_input = Vec::from([group1, group2]);
        let (statistic, pvalue) = f_oneway(sample_input, NaNPolicy::Propogate).unwrap();
        assert!(statistic.is_nan());
        assert!(pvalue.is_nan());
    }
    #[test]
    fn test_nan_in_data_w_error() {
        let group1 = Vec::from([0.0571, 0.0813, f64::NAN, 0.0836]);
        let group2 = Vec::from([0.0873, 0.0662, 0.0672, 0.0819, 0.0749]);
        let sample_input = Vec::from([group1, group2]);
        let result = f_oneway(sample_input, NaNPolicy::Error);
        assert_eq!(result, Err(FOneWayTestError::SampleContainsNaN));
    }
    #[test]
    fn test_bad_data_not_enough_samples() {
        let group1 = Vec::from([0.0, 0.0]);
        let sample_input = Vec::from([group1]);
        let result = f_oneway(sample_input, NaNPolicy::Propogate);
        assert_eq!(result, Err(FOneWayTestError::NotEnoughSamples))
    }
    #[test]
    fn test_bad_data_sample_too_small() {
        let group1 = Vec::new();
        let group2 = Vec::from([0.0873, 0.0662]);
        let sample_input = Vec::from([group1, group2]);
        let result = f_oneway(sample_input, NaNPolicy::Propogate);
        assert_eq!(result, Err(FOneWayTestError::SampleTooSmall));

        let group1 = Vec::from([f64::NAN]);
        let group2 = Vec::from([0.0873, 0.0662]);
        let sample_input = Vec::from([group1, group2]);
        let result = f_oneway(sample_input, NaNPolicy::Emit);
        assert_eq!(result, Err(FOneWayTestError::SampleTooSmall));

        let group1 = Vec::from([1.0]);
        let group2 = Vec::from([0.0873]);
        let sample_input = Vec::from([group1, group2]);
        let result = f_oneway(sample_input, NaNPolicy::Propogate);
        assert_eq!(result, Err(FOneWayTestError::SampleTooSmall));

        let group1 = Vec::from([1.0, f64::NAN]);
        let group2 = Vec::from([0.0873, f64::NAN]);
        let sample_input = Vec::from([group1, group2]);
        let result = f_oneway(sample_input, NaNPolicy::Emit);
        assert_eq!(result, Err(FOneWayTestError::SampleTooSmall));
    }
    #[test]
    fn test_bad_data_sample_contains_same_constants() {
        let group1 = Vec::from([1.0, 1.0]);
        let group2 = Vec::from([2.0, 2.0]);
        let sample_input = Vec::from([group1, group2]);
        let result = f_oneway(sample_input, NaNPolicy::Error);
        assert_eq!(result, Err(FOneWayTestError::SampleContainsSameConstants));

        let group1 = Vec::from([1.0, 1.0, 1.0]);
        let group2 = Vec::from([0.0873, 0.0662, 0.0342]);
        let sample_input = Vec::from([group1, group2]);
        let result = f_oneway(sample_input, NaNPolicy::Error);
        assert_eq!(result, Err(FOneWayTestError::SampleContainsSameConstants));
    }
}
