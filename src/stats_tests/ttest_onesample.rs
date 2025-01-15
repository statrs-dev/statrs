//! Provides the [one-sample t-test](https://en.wikipedia.org/wiki/Student%27s_t-test#One-sample_t-test)
//! and related functions

use crate::distribution::{ContinuousCDF, StudentsT};
use crate::stats_tests::{Alternative, NaNPolicy};

/// Represents the errors that can occur when computing the ttest_onesample function
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum TTestOneSampleError {
    /// sample must be greater than length 1
    SampleTooSmall,
    /// samples can not contain NaN when `nan_policy` is set to `NaNPolicy::Error`
    SampleContainsNaN,
}

impl std::fmt::Display for TTestOneSampleError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            TTestOneSampleError::SampleTooSmall => write!(f, "sample must be len > 1"),
            TTestOneSampleError::SampleContainsNaN => {
                write!(
                    f,
                    "samples can not contain NaN when nan_policy is set to NaNPolicy::Error"
                )
            }
        }
    }
}

impl std::error::Error for TTestOneSampleError {}

/// Perform a one sample t-test
///
/// Returns the t-statistic and p-value
///
/// # Remarks
///
/// Implementation based on [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html).
///
/// `a` needs to be mutable in case needing to filter out NaNs for NaNPolicy::Emit
///
/// # Examples
///
/// ```
/// use statrs::stats_tests::ttest_onesample::ttest_onesample;
/// use statrs::stats_tests::{Alternative, NaNPolicy};
/// let data = Vec::from([13f64, 9f64, 11f64, 8f64, 7f64, 12f64]);
/// let (statistic, pvalue) = ttest_onesample(data, 13f64, Alternative::TwoSided, NaNPolicy::Error).unwrap();
/// ```
pub fn ttest_onesample(
    mut a: Vec<f64>,
    popmean: f64,
    alternative: Alternative,
    nan_policy: NaNPolicy,
) -> Result<(f64, f64), TTestOneSampleError> {
    let has_nans = a.iter().any(|x| x.is_nan());
    if has_nans {
        match nan_policy {
            NaNPolicy::Propogate => {
                return Ok((f64::NAN, f64::NAN));
            }
            NaNPolicy::Error => {
                return Err(TTestOneSampleError::SampleContainsNaN);
            }
            NaNPolicy::Emit => {
                a = a.into_iter().filter(|x| !x.is_nan()).collect::<Vec<_>>();
            }
        }
    }

    let n = a.len();
    if n < 2 {
        return Err(TTestOneSampleError::SampleTooSmall);
    }
    let samplemean = a.iter().sum::<f64>() / (n as f64);
    let df = (n - 1) as f64;
    let s = a.iter().map(|x| (x - samplemean).powi(2)).sum::<f64>() / df;
    let se = (s / n as f64).sqrt();

    let tstat = (samplemean - popmean) / se;

    let t_dist =
        StudentsT::new(0.0, 1.0, df).expect("df should always be non NaN and greater than 0");

    let pvalue = match alternative {
        Alternative::TwoSided => 2.0 * (1.0 - t_dist.cdf(tstat.abs())),
        Alternative::Less => t_dist.cdf(tstat),
        Alternative::Greater => 1.0 - t_dist.cdf(tstat),
    };

    Ok((tstat, pvalue))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prec;

    /// Test one sample t-test comparing to
    #[test]
    fn test_jmp_example() {
        // Test against an example from jmp.com
        // https://www.jmp.com/en_us/statistics-knowledge-portal/t-test/one-sample-t-test.html
        let data = Vec::from([
            20.70f64, 27.46f64, 22.15f64, 19.85f64, 21.29f64, 24.75f64, 20.75f64, 22.91f64,
            25.34f64, 20.33f64, 21.54f64, 21.08f64, 22.14f64, 19.56f64, 21.10f64, 18.04f64,
            24.12f64, 19.95f64, 19.72f64, 18.28f64, 16.26f64, 17.46f64, 20.53f64, 22.12f64,
            25.06f64, 22.44f64, 19.08f64, 19.88f64, 21.39f64, 22.33f64, 25.79f64,
        ]);
        let (statistic, pvalue) =
            ttest_onesample(data.clone(), 20.0, Alternative::TwoSided, NaNPolicy::Error).unwrap();
        assert!(prec::almost_eq(statistic, 3.066831635284081, 1e-1));
        assert!(prec::almost_eq(pvalue, 0.004552621060635401, 1e-12));

        let (statistic, pvalue) =
            ttest_onesample(data.clone(), 20.0, Alternative::Greater, NaNPolicy::Error).unwrap();
        assert!(prec::almost_eq(statistic, 3.066831635284081, 1e-1));
        assert!(prec::almost_eq(pvalue, 0.0022763105303177005, 1e-12));

        let (statistic, pvalue) =
            ttest_onesample(data.clone(), 20.0, Alternative::Less, NaNPolicy::Error).unwrap();
        assert!(prec::almost_eq(statistic, 3.066831635284081, 1e-1));
        assert!(prec::almost_eq(pvalue, 0.9977236894696823, 1e-12));
    }
    #[test]
    fn test_nan_in_data_w_emit() {
        // results should be the same as the example above since the NaNs should be filtered out
        let data = Vec::from([
            20.70f64,
            27.46f64,
            22.15f64,
            19.85f64,
            21.29f64,
            24.75f64,
            20.75f64,
            22.91f64,
            25.34f64,
            20.33f64,
            21.54f64,
            21.08f64,
            22.14f64,
            19.56f64,
            21.10f64,
            18.04f64,
            24.12f64,
            19.95f64,
            19.72f64,
            18.28f64,
            16.26f64,
            17.46f64,
            20.53f64,
            22.12f64,
            25.06f64,
            22.44f64,
            19.08f64,
            19.88f64,
            21.39f64,
            22.33f64,
            25.79f64,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
        ]);
        let (statistic, pvalue) =
            ttest_onesample(data.clone(), 20.0, Alternative::TwoSided, NaNPolicy::Emit).unwrap();
        assert!(prec::almost_eq(statistic, 3.066831635284081, 1e-1));
        assert!(prec::almost_eq(pvalue, 0.004552621060635401, 1e-12));
    }
    #[test]
    fn test_nan_in_data_w_propogate() {
        let sample_input = Vec::from([1.3, f64::NAN]);
        let (statistic, pvalue) = ttest_onesample(
            sample_input,
            20.0,
            Alternative::TwoSided,
            NaNPolicy::Propogate,
        )
        .unwrap();
        assert!(statistic.is_nan());
        assert!(pvalue.is_nan());
    }
    #[test]
    fn test_nan_in_data_w_error() {
        let sample_input = Vec::from([0.0571, 0.0813, f64::NAN, 0.0836]);
        let result = ttest_onesample(sample_input, 20.0, Alternative::TwoSided, NaNPolicy::Error);
        assert_eq!(result, Err(TTestOneSampleError::SampleContainsNaN));
    }
    #[test]
    fn test_bad_data_sample_too_small() {
        let sample_input = Vec::new();
        let result = ttest_onesample(sample_input, 20.0, Alternative::TwoSided, NaNPolicy::Error);
        assert_eq!(result, Err(TTestOneSampleError::SampleTooSmall));

        let sample_input = Vec::from([1.0]);
        let result = ttest_onesample(sample_input, 20.0, Alternative::TwoSided, NaNPolicy::Error);
        assert_eq!(result, Err(TTestOneSampleError::SampleTooSmall));
    }
}
