//! Provides the [Mann-Whitney U test](https://en.wikipedia.org/wiki/Mann–Whitney_U_test#) and related
//! functions

use num_traits::clamp;

use crate::distribution::{ContinuousCDF, Normal};
use crate::stats_tests::Alternative;

/// Represents the errors that can occur when computing the mannwhitneyu function
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum MannWhitneyUError {
    /// at least one element of the input data can not be compared to another element (possibly due
    /// to float NaNs)
    UncomparableData,
    /// the samples for both `x` and `y` must be at least length 1
    SampleTooSmall,
    /// `MannWhitneyUMethod::Exact` is not implemented for data where ties exist
    ExactMethodWithTiesInData,
}

impl std::fmt::Display for MannWhitneyUError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            MannWhitneyUError::UncomparableData => {
                write!(f, "elements in the data are not comparable")
            }
            MannWhitneyUError::SampleTooSmall => write!(
                f,
                "the samples for both `x` and `y` must be at least length 1"
            ),
            MannWhitneyUError::ExactMethodWithTiesInData => write!(
                f,
                "using the Exact method with ties in input data is not supported"
            ),
        }
    }
}

impl std::error::Error for MannWhitneyUError {}

/// Represents the different methods that can be used when calculating the p-value for the
/// mannwhitneyu function
pub enum MannWhitneyUMethod {
    /// determine method based on input data provided in `x` and `y`. Will use `Exact` for smaller
    /// sample sizes and `AsymptoticInclContinuityCorrection` for larger samples and when there
    /// are ties in the data
    Automatic,
    /// calculate the exact p-value
    Exact,
    /// calculate an approximated (via normal distribution) p-value including a continuity
    /// correction
    AsymptoticInclContinuityCorrection,
    /// calculate an approximated (via normal distribution) p-value excluding a continuity
    /// correction
    AsymptoticExclContinuityCorrection,
}

/// ranks data and accounts for ties to calculate the U statistic
fn rankdata_mwu<T: PartialOrd>(xy: Vec<T>) -> Result<(Vec<f64>, Vec<usize>), MannWhitneyUError> {
    let mut j = (0..xy.len()).collect::<Vec<usize>>();
    let mut y = xy;

    // check to make sure data can be compared to generate the ranks
    for i in 0..y.len() {
        for k in i + 1..y.len() {
            if y[i].partial_cmp(&y[k]).is_none() {
                return Err(MannWhitneyUError::UncomparableData);
            }
        }
    }

    // calculate the ordinal rank minus 1 (ordinal index) in j which is roughly equivalent to
    // np.argsort. Additionally sort xy at the same time
    let mut zipped: Vec<_> = j.into_iter().zip(y).collect();
    zipped.sort_by(|(_, a), (_, b)| {
        a.partial_cmp(b)
            .expect("NaN should not exist or be filtered out by this point")
    });
    (j, y) = zipped.into_iter().unzip();

    let mut ranks_sorted: Vec<f64> = vec![999.0; y.len()];
    let mut t: Vec<usize> = vec![999; y.len()];

    let mut k = 0;
    let mut count = 1;
    let n = y.len();

    for i in 1..n {
        if y[i] != y[i - 1] {
            let ordinal_rank = k + 1;
            let rank = ordinal_rank as f64 + (count as f64 - 1.0) / 2.0;
            // repeat the rank in the event of ties
            ranks_sorted[k..i].fill(rank);
            // for ties, match scipy logic and have first occurrence be the count
            // and all additional occurrences be 0
            t[k] = count;
            t[(k + 1)..i].fill(0);

            // reset to handle next occurrence of a unique value
            k = i;
            count = 0;
        }
        count += 1;
    }

    // handle from the last set of unique values to the end
    // same logic as above except goes until n (instead of i) including the last count increment
    let ordinal_rank = k + 1;
    let rank = ordinal_rank as f64 + (count as f64 - 1.0) / 2.0;
    ranks_sorted[k..n].fill(rank);
    t[k] = count;
    t[(k + 1)..n].fill(0);

    // leverage the ordinal indices from j to reverse into to the original ordering
    let mut ranks = ranks_sorted;
    let mut zipped: Vec<_> = j.into_iter().zip(ranks).collect();
    zipped.sort_by(|(i, _), (j, _)| i.partial_cmp(j).unwrap());
    (_, ranks) = zipped.into_iter().unzip::<usize, f64, Vec<_>, Vec<_>>();

    Ok((ranks, t))
}

/// based on https://github.com/scipy/scipy/blob/92d2a8592782ee19a1161d0bf3fc2241ba78bb63/scipy/stats/_mannwhitneyu.py#L149
fn calc_mwu_asymptotic_pvalue(
    u: f64,
    n1: usize,
    n2: usize,
    t: Vec<usize>,
    continuity: bool,
) -> f64 {
    let mu = ((n1 * n2) as f64) / 2.0;

    let tie_term = t.iter().map(|x| x.pow(3) - x).sum::<usize>();

    let n1 = n1 as f64;
    let n2 = n2 as f64;
    let n = n1 + n2;

    let s: f64 = (n1 * n2 / 12.0 * ((n + 1.0) - tie_term as f64 / (n * (n - 1.0)))).sqrt();

    let mut numerator = u - mu;
    if continuity {
        numerator -= 0.5;
    }

    let z = numerator / s;

    // NOTE: z could be infinity (if all input values are the same for example)
    // but the Normal CDF should handle this in a consistent way with scipy
    let norm_dist = Normal::default();
    1.0 - norm_dist.cdf(z)
}

fn calc_mwu_exact_pvalue(u: f64, n1: usize, n2: usize) -> f64 {
    let n = n1 + n2;
    let k = n1.min(n2); // use the smaller of the two for less combinations to go through
    let mut a: Vec<usize> = (0..n).collect();

    // placeholder for number of times U (observed) is smaller than the universe of U values
    let mut numerator = 0;
    let mut total = 0; // total combinations (universe of U values)

    loop {
        // calculate the number of times the hypothesis is rejected
        //
        // add k since index 0 all the indices need to be shifted by 1 to represent ranks
        let r1 = a[0..k].iter().sum::<usize>() + k;
        let u_generic = r1 - (k * (k + 1)) / 2;
        if u <= (u_generic as f64) {
            numerator += 1;
        }
        total += 1;

        // handle generating the next combination of n choose k (non-recursively)
        //
        // figure out the right most index g
        let mut i = k;
        while i > 0 {
            i -= 1;
            if a[i] != i + n - k {
                break;
            }
        }

        // all combinations have been generated since the first index is at its max value
        if i == 0 && a[i] == n - k {
            break;
        }

        a[i] += 1;

        for j in i + 1..k {
            a[j] = a[j - 1] + 1;
        }
    }

    if k == n1 {
        1.0 - numerator as f64 / total as f64
    } else {
        // if k was set to n2, return back the compliment p-value
        numerator as f64 / total as f64
    }
}

/// Perform a Mann-Whitney U (Wilcoxon rank-sum) test
///
/// Returns the U statistic (based on `x`) and p-value
///
/// # Remarks
///
/// For larger sample sizes, the Exact method can become computationally expensive. Per Wikipedia,
/// samples sizes (length of `x` + length of `y`) above 20 are approximated fairly well using the
/// asymptotic (normal) methods.
///
/// Implementation was largely based on the [scipy version](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html#scipy.stats.mannwhitneyu).
/// There are a few deviations including, not supporting calculation of the value via permutation
/// tests, not supporting calculation of the exact p-value where input data includes ties, and not
/// supporting the NaN policy due to being generic on T which might not have NaN values.
///
/// # Examples
///
/// ```
/// use statrs::stats_tests::mannwhitneyu::{mannwhitneyu, MannWhitneyUMethod};
/// use statrs::stats_tests::Alternative;
///
/// // based on scipy example
/// let male = Vec::from([19, 22, 16, 29, 24]);
/// let female = Vec::from([20, 11, 17, 12]);
///
/// let (statistic, pvalue) = mannwhitneyu(
///     &male,
///     &female,
///     MannWhitneyUMethod::Automatic,
///     Alternative::TwoSided,
/// )
/// .unwrap();
/// ```
pub fn mannwhitneyu<T: PartialOrd + Clone>(
    x: &[T],
    y: &[T],
    method: MannWhitneyUMethod,
    alternative: Alternative,
) -> Result<(f64, f64), MannWhitneyUError> {
    let n1 = x.len();
    let n2 = y.len();

    if n1 == 0 || n2 == 0 {
        return Err(MannWhitneyUError::SampleTooSmall);
    }

    let mut x = x.to_vec();
    let mut y = y.to_vec();
    x.append(&mut y);

    let (ranks, t) = rankdata_mwu(x)?;
    // NOTE: in the case of ties (eg: x = &[1, 2, 3] and y = &[3, 4, 5]), the U statistic can be a float
    // (being #.5). When there are no ties, U will always be a whole number
    let r1 = ranks[..n1].iter().sum::<f64>();
    let u1 = r1 - (n1 * (n1 + 1) / 2) as f64;
    let u2 = (n1 * n2) as f64 - u1;

    // f is a factor to apply to the p-value in a two-sided test
    let (u, f) = match alternative {
        Alternative::Greater => (u1, 1),
        Alternative::Less => (u2, 1),
        Alternative::TwoSided => (u1.max(u2), 2),
    };

    let mut pvalue = match method {
        MannWhitneyUMethod::Automatic => {
            if (n1 > 8 && n2 > 8) || t.iter().any(|x| x > &1usize) {
                calc_mwu_asymptotic_pvalue(u, n1, n2, t, true)
            } else {
                calc_mwu_exact_pvalue(u, n1, n2)
            }
        }
        MannWhitneyUMethod::Exact => {
            if t.iter().any(|x| x > &1usize) {
                return Err(MannWhitneyUError::ExactMethodWithTiesInData);
            }
            calc_mwu_exact_pvalue(u, n1, n2)
        }
        MannWhitneyUMethod::AsymptoticInclContinuityCorrection => {
            calc_mwu_asymptotic_pvalue(u, n1, n2, t, true)
        }
        MannWhitneyUMethod::AsymptoticExclContinuityCorrection => {
            calc_mwu_asymptotic_pvalue(u, n1, n2, t, false)
        }
    };

    pvalue *= f as f64;
    pvalue = clamp(pvalue, 0.0, 1.0);

    Ok((u1, pvalue))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prec;

    #[test]
    fn test_wikipedia_example() {
        // Replicate example from https://en.wikipedia.org/wiki/Mann–Whitney_U_test#Illustration_of_calculation_methods
        let data = "THHHHHTTTTTH";
        let mut x = Vec::new();
        let mut y = Vec::new();

        for (i, c) in data.chars().enumerate() {
            if c == 'T' {
                x.push(i + 1)
            } else {
                y.push(i + 1)
            }
        }
        let (statistic, _) = mannwhitneyu(
            &x,
            &y,
            MannWhitneyUMethod::AsymptoticInclContinuityCorrection,
            Alternative::Less,
        )
        .unwrap();
        assert_eq!(statistic, 25.0);

        let (statistic, _) = mannwhitneyu(
            &y,
            &x,
            MannWhitneyUMethod::AsymptoticInclContinuityCorrection,
            Alternative::Greater,
        )
        .unwrap();
        assert_eq!(statistic, 11.0);
    }

    #[test]
    fn test_scipy_example() {
        // Test against scipy function including the documentation example
        // https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
        // as well as additional validations comparing to examples run in python
        let male = Vec::from([19, 22, 16, 29, 24]);
        let female = Vec::from([20, 11, 17, 12]);

        let (statistic, pvalue) = mannwhitneyu(
            &male,
            &female,
            MannWhitneyUMethod::Automatic,
            Alternative::TwoSided,
        )
        .unwrap();
        assert_eq!(statistic, 17.0);
        assert!(prec::almost_eq(pvalue, 0.1111111111111111, 1e-9));

        let (statistic, _) = mannwhitneyu(
            &female,
            &male,
            MannWhitneyUMethod::Automatic,
            Alternative::TwoSided,
        )
        .unwrap();
        assert_eq!(statistic, 3.0);

        let (statistic, pvalue) = mannwhitneyu(
            &male,
            &female,
            MannWhitneyUMethod::AsymptoticInclContinuityCorrection,
            Alternative::TwoSided,
        )
        .unwrap();
        assert_eq!(statistic, 17.0);
        assert!(prec::almost_eq(pvalue, 0.11134688653314041, 1e-9));

        // not in scipy's official example but testing other variations against python output
        let (_, pvalue) = mannwhitneyu(
            &male,
            &female,
            MannWhitneyUMethod::AsymptoticExclContinuityCorrection,
            Alternative::Less,
        )
        .unwrap();
        assert!(prec::almost_eq(pvalue, 0.95679463351315, 1e-9));

        let (_, pvalue) =
            mannwhitneyu(&male, &female, MannWhitneyUMethod::Exact, Alternative::Less).unwrap();
        assert!(prec::almost_eq(pvalue, 0.9682539682539683, 1e-9));

        let (_, pvalue) = mannwhitneyu(
            &male,
            &female,
            MannWhitneyUMethod::AsymptoticInclContinuityCorrection,
            Alternative::Greater,
        )
        .unwrap();
        assert!(prec::almost_eq(pvalue, 0.055673443266570206, 1e-9));

        let (statistic, pvalue) = mannwhitneyu(
            &[1],
            &[2],
            MannWhitneyUMethod::AsymptoticInclContinuityCorrection,
            Alternative::Less,
        )
        .unwrap();
        assert_eq!(statistic, 0.0);
        assert!(prec::almost_eq(pvalue, 0.5, 1e-9));

        // larger deviation from scipy logic for exact so double check here
        // also check usage with floats
        let x = &[5.0, 2.0, 7.0, 8.0, 9.0, 3.0, 11.0, 12.0];
        let y = &[1.0, 6.0, 10.0, 4.0];

        let (statistic, pvalue) =
            mannwhitneyu(x, y, MannWhitneyUMethod::Exact, Alternative::Greater).unwrap();
        assert_eq!(statistic, 21.0);
        assert!(prec::almost_eq(pvalue, 0.23030303030303031, 1e-9));

        let (statistic, pvalue) =
            mannwhitneyu(x, y, MannWhitneyUMethod::Exact, Alternative::Less).unwrap();
        assert_eq!(statistic, 21.0);
        assert!(prec::almost_eq(pvalue, 0.8161616161616161, 1e-9));

        let (statistic, pvalue) =
            mannwhitneyu(x, y, MannWhitneyUMethod::Exact, Alternative::TwoSided).unwrap();
        assert_eq!(statistic, 21.0);
        assert!(prec::almost_eq(pvalue, 0.46060606060606063, 1e-9));

        let (statistic, pvalue) = mannwhitneyu(
            &[1, 1],
            &[1, 1, 1],
            MannWhitneyUMethod::AsymptoticInclContinuityCorrection,
            Alternative::TwoSided,
        )
        .unwrap();
        assert_eq!(statistic, 3.0);
        assert!(prec::almost_eq(pvalue, 1.0, 1e-9));
    }

    #[test]
    fn test_bad_data_nan() {
        let male = Vec::from([19.0, 22.0, 16.0, 29.0, 24.0, f64::NAN]);
        let female = Vec::from([20.0, 11.0, 17.0, 12.0]);

        let result = mannwhitneyu(
            &male,
            &female,
            MannWhitneyUMethod::Automatic,
            Alternative::TwoSided,
        );
        assert_eq!(result, Err(MannWhitneyUError::UncomparableData));
    }
    #[test]
    fn test_bad_data_sample_too_small() {
        let result = mannwhitneyu(
            &[],
            &[1, 2, 3],
            MannWhitneyUMethod::Automatic,
            Alternative::TwoSided,
        );
        assert_eq!(result, Err(MannWhitneyUError::SampleTooSmall));

        let result = mannwhitneyu::<i32>(
            &[],
            &[],
            MannWhitneyUMethod::Automatic,
            Alternative::TwoSided,
        );
        assert_eq!(result, Err(MannWhitneyUError::SampleTooSmall));
    }
    #[test]
    fn test_bad_data_exact_with_ties() {
        let result = mannwhitneyu(
            &[1, 2],
            &[1, 2, 3],
            MannWhitneyUMethod::Exact,
            Alternative::TwoSided,
        );
        assert_eq!(result, Err(MannWhitneyUError::ExactMethodWithTiesInData));
    }
    #[test]
    fn test_rankdata_mwu() {
        let data = Vec::from([1, 4, 3]);
        let (rank, t) = rankdata_mwu(data).expect("data is good");
        assert_eq!(rank, Vec::from([1.0, 3.0, 2.0]));
        assert_eq!(t, Vec::from([1, 1, 1]));

        let data = Vec::from([4.0, 2.0, 2.0, 1.0]);
        let (rank, t) = rankdata_mwu(data).expect("data is good");
        assert_eq!(rank, Vec::from([4.0, 2.5, 2.5, 1.0]));
        assert_eq!(t, Vec::from([1, 2, 0, 1,]));

        let data = Vec::from([1, 2, 2, 2, 3]);
        let (rank, t) = rankdata_mwu(data).expect("data is good");
        assert_eq!(rank, Vec::from([1.0, 3.0, 3.0, 3.0, 5.0]));
        assert_eq!(t, Vec::from([1, 3, 0, 0, 1]));
    }
    #[test]
    fn test_calc_mwu_exact_pvalue() {
        let pvalue = calc_mwu_exact_pvalue(4.0, 3, 2);
        assert!(prec::almost_eq(pvalue, 0.4, 1e-9));
        let pvalue = calc_mwu_exact_pvalue(4.0, 2, 3);
        assert!(prec::almost_eq(pvalue, 0.6, 1e-9));
    }
}
