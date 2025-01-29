//! Provides the [skewtest](https://docs.scipy.org/doc/scipy-1.15.0/reference/generated/scipy.stats.skewtest.html)
//! to test whether or not provided data is different than a normal distribution

use crate::distribution::{ContinuousCDF, Normal};
use crate::stats_tests::{Alternative, NaNPolicy};

/// Represents the errors that can occur when computing the skewtest function
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum SkewTestError {
    /// sample must contain at least 8 observations
    SampleTooSmall,
    /// samples can not contain NaN when `nan_policy` is set to `NaNPolicy::Error`
    SampleContainsNaN,
}

impl std::fmt::Display for SkewTestError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            SkewTestError::SampleTooSmall => {
                write!(f, "sample must contain at least 8 observations")
            }
            SkewTestError::SampleContainsNaN => {
                write!(
                    f,
                    "samples can not contain NaN when nan_policy is set to NaNPolicy::Error"
                )
            }
        }
    }
}

impl std::error::Error for SkewTestError {}

fn calc_root_b1(data: &[f64]) -> f64 {
    // Fisher's moment coefficient of skewness
    // https://en.wikipedia.org/wiki/Skewness#Definition
    let n = data.len() as f64;
    let mu = data.iter().sum::<f64>() / n;

    // NOTE: population not sample skewness
    (data.iter().map(|x_i| (x_i - mu).powi(3)).sum::<f64>() / n)
        / (data.iter().map(|x_i| (x_i - mu).powi(2)).sum::<f64>() / n).powf(1.5)
}

/// Perform a skewness test for whether the skew of the sample provided is different than a normal
/// distribution
///
/// Returns the z-score and p-value
///
/// # Remarks
///
/// `a` needs to be mutable in case needing to filter out NaNs for NaNPolicy::Emit
///
/// Implementation based on [fintools.com](https://www.fintools.com/docs/normality_correlation.pdf)
/// which indirectly uses [D'Agostino, (1970)](https://doi.org/10.2307/2684359)
/// while aligning to [scipy's](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewtest.html#scipy.stats.skewtest)
/// function header where possible. The scipy implementation was also used for testing and validation.
/// Includes the use of [Shapiro & Wilk (1965)](https://doi.org/10.2307/2333709) for
/// testing and validation.
///
/// # Examples
///
/// ```
/// use statrs::stats_tests::skewtest::skewtest;
/// use statrs::stats_tests::{Alternative, NaNPolicy};
/// let data = Vec::from([ 1.0f64, 2.0f64, 3.0f64, 4.0f64, 5.0f64, 6.0f64, 7.0f64, 8.0f64, ]);
/// let (statistic, pvalue) = skewtest(data, Alternative::TwoSided, NaNPolicy::Error).unwrap();
/// ```
pub fn skewtest(
    mut a: Vec<f64>,
    alternative: Alternative,
    nan_policy: NaNPolicy,
) -> Result<(f64, f64), SkewTestError> {
    let has_nans = a.iter().any(|x| x.is_nan());
    if has_nans {
        match nan_policy {
            NaNPolicy::Propogate => {
                return Ok((f64::NAN, f64::NAN));
            }
            NaNPolicy::Error => {
                return Err(SkewTestError::SampleContainsNaN);
            }
            NaNPolicy::Emit => {
                a = a.into_iter().filter(|x| !x.is_nan()).collect::<Vec<_>>();
            }
        }
    }

    let n = a.len();
    if n < 8 {
        return Err(SkewTestError::SampleTooSmall);
    }
    let n = n as f64;

    let root_b1 = calc_root_b1(&a);
    let mut y = root_b1 * ((n + 1.0) * (n + 3.0) / (6.0 * (n - 2.0))).sqrt();
    let beta2_root_b1 = 3.0 * (n.powi(2) + 27.0 * n - 70.0) * (n + 1.0) * (n + 3.0)
        / ((n - 2.0) * (n + 5.0) * (n + 7.0) * (n + 9.0));
    let w_sq = -1.0 + (2.0 * (beta2_root_b1 - 1.0)).sqrt();
    let delta = 1.0 / (0.5 * w_sq.ln()).sqrt();
    let alpha = (2.0 / (w_sq - 1.0)).sqrt();
    // correction from scipy version to`match scipy example results
    if y == 0.0 {
        y = 1.0;
    }
    let zscore = delta * (y / alpha + ((y / alpha).powi(2) + 1.0).sqrt()).ln();

    let norm_dist = Normal::default();

    let pvalue = match alternative {
        Alternative::TwoSided => 2.0 * (1.0 - norm_dist.cdf(zscore.abs())),
        Alternative::Less => norm_dist.cdf(zscore),
        Alternative::Greater => 1.0 - norm_dist.cdf(zscore),
    };

    Ok((zscore, pvalue))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prec;

    #[test]
    fn test_scipy_example() {
        let data = Vec::from([
            148.0f64, 154.0f64, 158.0f64, 160.0f64, 161.0f64, 162.0f64, 166.0f64, 170.0f64,
            182.0f64, 195.0f64, 236.0f64,
        ]);
        let (statistic, pvalue) =
            skewtest(data.clone(), Alternative::TwoSided, NaNPolicy::Error).unwrap();
        assert!(prec::almost_eq(statistic, 2.7788579769903414, 1e-1));
        assert!(prec::almost_eq(pvalue, 0.005455036974740185, 1e-9));

        let (statistic, pvalue) = skewtest(
            Vec::from([
                1.0f64, 2.0f64, 3.0f64, 4.0f64, 5.0f64, 6.0f64, 7.0f64, 8.0f64,
            ]),
            Alternative::TwoSided,
            NaNPolicy::Error,
        )
        .unwrap();
        assert!(prec::almost_eq(statistic, 1.0108048609177787, 1e-1));
        assert!(prec::almost_eq(pvalue, 0.3121098361421897, 1e-9));
        let (statistic, pvalue) = skewtest(
            Vec::from([
                2.0f64, 8.0f64, 0.0f64, 4.0f64, 1.0f64, 9.0f64, 9.0f64, 0.0f64,
            ]),
            Alternative::TwoSided,
            NaNPolicy::Error,
        )
        .unwrap();
        assert!(prec::almost_eq(statistic, 0.44626385374196975, 1e-1));
        assert!(prec::almost_eq(pvalue, 0.6554066631275459, 1e-9));
        let (statistic, pvalue) = skewtest(
            Vec::from([
                1.0f64, 2.0f64, 3.0f64, 4.0f64, 5.0f64, 6.0f64, 7.0f64, 8000.0f64,
            ]),
            Alternative::TwoSided,
            NaNPolicy::Error,
        )
        .unwrap();
        assert!(prec::almost_eq(statistic, 3.571773510360407, 1e-1));
        assert!(prec::almost_eq(pvalue, 0.0003545719905823133, 1e-9));
        let (statistic, pvalue) = skewtest(
            Vec::from([
                100.0f64, 100.0f64, 100.0f64, 100.0f64, 100.0f64, 100.0f64, 100.0f64, 101.0f64,
            ]),
            Alternative::TwoSided,
            NaNPolicy::Error,
        )
        .unwrap();
        assert!(prec::almost_eq(statistic, 3.5717766638478072, 1e-1));
        assert!(prec::almost_eq(pvalue, 0.000354567720281634, 1e012));
        let (statistic, pvalue) = skewtest(
            Vec::from([
                1.0f64, 2.0f64, 3.0f64, 4.0f64, 5.0f64, 6.0f64, 7.0f64, 8.0f64,
            ]),
            Alternative::Less,
            NaNPolicy::Error,
        )
        .unwrap();
        assert!(prec::almost_eq(statistic, 1.0108048609177787, 1e-1));
        assert!(prec::almost_eq(pvalue, 0.8439450819289052, 1e-9));
        let (statistic, pvalue) = skewtest(
            Vec::from([
                1.0f64, 2.0f64, 3.0f64, 4.0f64, 5.0f64, 6.0f64, 7.0f64, 8.0f64,
            ]),
            Alternative::Greater,
            NaNPolicy::Error,
        )
        .unwrap();
        assert!(prec::almost_eq(statistic, 1.0108048609177787, 1e-1));
        assert!(prec::almost_eq(pvalue, 0.15605491807109484, 1e-9));
    }
    #[test]
    fn test_nan_in_data_w_emit() {
        // results should be the same as the example above since the NaNs should be filtered out
        let data = Vec::from([
            148.0f64,
            154.0f64,
            158.0f64,
            160.0f64,
            161.0f64,
            162.0f64,
            166.0f64,
            170.0f64,
            182.0f64,
            195.0f64,
            236.0f64,
            f64::NAN,
        ]);
        let (statistic, pvalue) =
            skewtest(data.clone(), Alternative::TwoSided, NaNPolicy::Emit).unwrap();
        assert!(prec::almost_eq(statistic, 2.7788579769903414, 1e-1));
        assert!(prec::almost_eq(pvalue, 0.005455036974740185, 1e-9));
    }
    #[test]
    fn test_nan_in_data_w_propogate() {
        let sample_input = Vec::from([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, f64::NAN]);
        let (statistic, pvalue) =
            skewtest(sample_input, Alternative::TwoSided, NaNPolicy::Propogate).unwrap();
        assert!(statistic.is_nan());
        assert!(pvalue.is_nan());
    }
    #[test]
    fn test_nan_in_data_w_error() {
        let sample_input = Vec::from([0.0571, 0.0813, f64::NAN, 0.0836]);
        let result = skewtest(sample_input, Alternative::TwoSided, NaNPolicy::Error);
        assert_eq!(result, Err(SkewTestError::SampleContainsNaN));
    }
    #[test]
    fn test_bad_data_sample_too_small() {
        let sample_input = Vec::new();
        let result = skewtest(sample_input, Alternative::TwoSided, NaNPolicy::Error);
        assert_eq!(result, Err(SkewTestError::SampleTooSmall));

        let sample_input = Vec::from([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, f64::NAN]);
        let result = skewtest(sample_input, Alternative::TwoSided, NaNPolicy::Emit);
        assert_eq!(result, Err(SkewTestError::SampleTooSmall));
    }
    #[test]
    fn test_calc_root_b1() {
        // compare to https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html
        // since no wikipedia examples
        let sample_input = Vec::from([1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(calc_root_b1(&sample_input), 0.0);

        let sample_input = Vec::from([2.0, 8.0, 0.0, 4.0, 1.0, 9.0, 9.0, 0.0]);
        let result = calc_root_b1(&sample_input);
        assert!(prec::almost_eq(result, 0.2650554122698573, 1e-1));
    }
}
