//! Provides the functions related to [Chi-Squared tests](https://en.wikipedia.org/wiki/Chi-squared_test)

use crate::distribution::{ChiSquared, ContinuousCDF};

/// Represents the errors that can occur when computing the chisquare function
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum ChiSquareTestError {
    /// `f_obs` must have a length (or number of categories) greater than 1
    FObsInvalid,
    /// `f_exp` must have same length and sum as `f_obs`
    FExpInvalid,
    /// for the p-value to be meaningful, `ddof` must be at least two less
    /// than the number of categories, k, which is the length of `f_obs`
    DdofInvalid,
}

impl std::fmt::Display for ChiSquareTestError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ChiSquareTestError::FObsInvalid => {
                write!(f, "`f_obs` must have a length greater than 1")
            }
            ChiSquareTestError::FExpInvalid => {
                write!(f, "`f_exp` must have same length and sum as `f_obs`")
            }
            ChiSquareTestError::DdofInvalid => {
                write!(f, "for the p-value to be meaningful, `ddof` must be at least two less than the number of categories, k, which is the length of `f_obs`")
            }
        }
    }
}

impl std::error::Error for ChiSquareTestError {}

/// Perform a Pearson's chi-square test
///
/// Returns the chi-square test statistic and p-value
///
/// # Remarks
///
/// `ddof` represents an adjustment that can be made to the degrees of freedom where the unadjusted
/// degrees of freedom is `f_obs.len() - 1`.
///
/// Implementation based on [wikipedia](https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test)
/// while aligning to [scipy's](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html)
/// function header where possible. The scipy implementation was also used for testing and validation.
///
/// # Examples
///
/// ```
/// use statrs::stats_tests::chisquare::chisquare;
/// let (statistic, pvalue) = chisquare(&[16, 18, 16, 14, 12, 12], None, None).unwrap();
/// let (statistic, pvalue) = chisquare(&[16, 18, 16, 14, 12, 12], None, Some(1)).unwrap();
/// let (statistic, pvalue) = chisquare(
///     &[16, 18, 16, 14, 12, 12],
///     Some(&[16.0, 16.0, 16.0, 16.0, 16.0, 8.0]),
///     None,
/// )
/// .unwrap();
/// ```
pub fn chisquare(
    f_obs: &[usize],
    f_exp: Option<&[f64]>,
    ddof: Option<usize>,
) -> Result<(f64, f64), ChiSquareTestError> {
    let n: usize = f_obs.len();
    if n <= 1 {
        return Err(ChiSquareTestError::FObsInvalid);
    }
    let total_samples = f_obs.iter().sum();
    let f_obs: Vec<f64> = f_obs.iter().map(|x| *x as f64).collect();

    let f_exp = match f_exp {
        Some(f_to_validate) => {
            // same length check
            if f_to_validate.len() != n {
                return Err(ChiSquareTestError::FExpInvalid);
            }
            // same sum check
            if f_to_validate.iter().sum::<f64>() as usize != total_samples {
                return Err(ChiSquareTestError::FExpInvalid);
            }
            f_to_validate.to_vec()
        }
        None => {
            // make the expected assuming equal frequency
            vec![total_samples as f64 / n as f64; n]
        }
    };

    let ddof = match ddof {
        Some(ddof_to_validate) => {
            if ddof_to_validate >= (n - 1) {
                return Err(ChiSquareTestError::DdofInvalid);
            }
            ddof_to_validate
        }
        None => 0,
    };
    let dof = n - 1 - ddof;

    let stat = f_obs
        .into_iter()
        .zip(f_exp)
        .map(|(o, e)| (o - e).powi(2) / e)
        .sum::<f64>();

    let chi_dist = ChiSquared::new(dof as f64).expect("ddof validity should already be checked");
    let pvalue = 1.0 - chi_dist.cdf(stat);

    Ok((stat, pvalue))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prec;

    #[test]
    fn test_scipy_example() {
        let (statistic, pvalue) = chisquare(&[16, 18, 16, 14, 12, 12], None, None).unwrap();
        assert!(prec::almost_eq(statistic, 2.0, 1e-1));
        assert!(prec::almost_eq(pvalue, 0.84914503608460956, 1e-9));

        let (statistic, pvalue) = chisquare(
            &[16, 18, 16, 14, 12, 12],
            Some(&[16.0, 16.0, 16.0, 16.0, 16.0, 8.0]),
            None,
        )
        .unwrap();
        assert!(prec::almost_eq(statistic, 3.5, 1e-1));
        assert!(prec::almost_eq(pvalue, 0.62338762774958223, 1e-9));

        let (statistic, pvalue) = chisquare(&[16, 18, 16, 14, 12, 12], None, Some(1)).unwrap();
        assert!(prec::almost_eq(statistic, 2.0, 1e-1));
        assert!(prec::almost_eq(pvalue, 0.7357588823428847, 1e-9));
    }
    #[test]
    fn test_wiki_example() {
        // fairness of dice - p-value not provided
        let (statistic, _) = chisquare(&[5, 8, 9, 8, 10, 20], None, None).unwrap();
        assert!(prec::almost_eq(statistic, 13.4, 1e-1));

        let (statistic, _) = chisquare(&[5, 8, 9, 8, 10, 20], Some(&[10.0; 6]), None).unwrap();
        assert!(prec::almost_eq(statistic, 13.4, 1e-1));

        // chi-squared goodness of fit test
        let (statistic, pvalue) = chisquare(&[44, 56], Some(&[50.0, 50.0]), None).unwrap();
        assert!(prec::almost_eq(statistic, 1.44, 1e-2));
        assert!(prec::almost_eq(pvalue, 0.24, 1e-2));
    }

    #[test]
    fn test_bad_data_f_obs_invalid() {
        let result = chisquare(&[16], None, None);
        assert_eq!(result, Err(ChiSquareTestError::FObsInvalid));
        let f_exp: &[usize] = &[];
        let result = chisquare(f_exp, None, None);
        assert_eq!(result, Err(ChiSquareTestError::FObsInvalid));
    }
    #[test]
    fn test_bad_data_f_exp_invalid() {
        let result = chisquare(&[16, 18, 16, 14, 12, 12], Some(&[1.0, 2.0, 3.0]), None);
        assert_eq!(result, Err(ChiSquareTestError::FExpInvalid));
        let result = chisquare(&[16, 18, 16, 14, 12, 12], Some(&[16.0; 6]), None);
        assert_eq!(result, Err(ChiSquareTestError::FExpInvalid));
    }
    #[test]
    fn test_bad_data_ddof_invalid() {
        let result = chisquare(&[16, 18, 16, 14, 12, 12], None, Some(5));
        assert_eq!(result, Err(ChiSquareTestError::DdofInvalid));
        let result = chisquare(&[16, 18, 16, 14, 12, 12], None, Some(100));
        assert_eq!(result, Err(ChiSquareTestError::DdofInvalid));
    }
}
