//! Provides the [Kolmogorov-Smirnov (KS) test](https://en.wikipedia.org/wiki/Kolmogorov–Smirnov_test) and related
//! functions

use core::f64;
use std::iter::zip;

use num_traits::clamp;

use crate::distribution::ContinuousCDF;

use crate::function::factorial;

use super::NaNPolicy;

/// Represents the errors that can occur when computing the ks_test functions
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum KSTestError {
    /// sample must be greater than length 1
    SampleTooSmall,
    /// samples can not contain NaN when `nan_policy` is set to `NaNPolicy::Error`
    SampleContainsNaN,
    /// `KSOneSampleAlternativeMethod::TwoSidedExact`selected with ties in data
    ExactAndTies,
    /// `KSOneSampleAlternativeMethod::TwoSidedExact`selected with the size of the data (`n`) being
    /// too large
    ExactAndTooLarge,
}

impl std::fmt::Display for KSTestError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            KSTestError::SampleTooSmall => write!(f, "sample must be len > 1"),
            KSTestError::SampleContainsNaN => {
                write!(
                    f,
                    "samples can not contain NaN when nan_policy is set to NaNPolicy::Error"
                )
            }
            KSTestError::ExactAndTies => write!(f, "`KSOneSampleAlternativeMethod::TwoSidedExact`selected with ties in data"),
            KSTestError::ExactAndTooLarge => write!(f, "`KSOneSampleAlternativeMethod::TwoSidedExact`selected with the size of the data (`n`) being too large"),
        }
    }
}

impl std::error::Error for KSTestError {}

/// Represents the different methods that can be used when calculating the p-value for the
/// one sample KS test.
///
/// There are numerous algorithms for calculation the p-value of for the KS
/// test with various trade-offs related to speed and precision for when to use them (see
/// [Simard & L’Ecuyer (2011)](doi.org/10.18637/jss.v039.i11) for an overview of some of the
/// different options related to two-sided p-value calculation). The implementation here does not
/// currently provide functionality that accounts for all the trade-offs. Instead, it aims to be
/// somewhat serviceable while leaving the door open for future enhancements. The `TwoSidedExact`,
/// while possibly on the slower side, for `n` < 140 will produce the exact p-value.
/// `TwoSidedAsymptotic`, for `n` > 140, should have roughly 5 digits of precision which should be
/// sufficient for the majority of use cases.
///
/// Eventually, an `Automatic` option could be added that would choose the best method
/// based on the size of the data and the value of the statistic.
#[non_exhaustive]
pub enum KSOneSampleAlternativeMethod {
    /// uses [Birnbaum & Tingey (1951)](doi.org/10.1214/aoms/1177729550) to calculate the p-value
    /// for the one-sided hypothesis test.
    Less,
    /// uses [Birnbaum & Tingey (1951)](doi.org/10.1214/aoms/1177729550) to calculate the p-value
    /// for the one-sided hypothesis test.
    Greater,
    /// uses [Marsaglia, Tsang & Wang (2003)](doi.org/10.18637/jss.v008.i18) to calculate the
    /// p-value for the two-sided hypothesis test. This implementation can become slow for larger
    /// `n`s and will error with if there are ties in the input data or the input data is too
    /// large. The threshold for too large is data with length 170 lining up with the
    /// implementation of [`factorial::factorial`] being used. Exact calculation requires the use of
    /// [`nalgebra`] crate/feature.
    #[cfg(feature = "nalgebra")]
    TwoSidedExact,
    /// calculates an approximated p-value based on asymptotic approximation described in
    /// Kolmogorov (1933). The asymptotic approximation is commonly used in other languages when
    /// the exact form is not used.
    TwoSidedAsymptotic,
    /// calculates an approximated p-value based on 2 times the one sided p-value (same algorithm
    /// used for `Less` and `Greater` calculations).
    TwoSidedApproximate,
}

fn onesample_birnbaum_tingey_onesided_pvalue(d: f64, n: f64) -> f64 {
    // Birnbaum & Tingey (1951)
    let mut sum = 0.0;
    for j in 0..=(n * (1.0 - d)).floor() as u64 {
        sum += factorial::binomial(n as u64, j)
            * (j as f64 / n + d).powi(j as i32 - 1)
            * (1.0 - d - j as f64 / n).powi(n as i32 - j as i32);
    }
    d * sum
}

fn onesample_kolmogorov_twosided_pvalue(d: f64, n: f64) -> f64 {
    // Kolmogorov (1933)
    // https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test#Kolmogorov_distribution
    // 1-2\sum _{k=1}^{\infty }(-1)^{k-1}e^{-2k^{2}x^{2}}
    let x = d * n.sqrt();

    let mut sum = 0.0;
    let mut k: f64 = 1.0;
    loop {
        let term = (-2.0 * k * k * x * x).exp();
        sum += (-1.0f64).powf(k - 1.0) * term;
        if term.abs() < 1e-10 {
            break; // break when added term is relatively small
        }
        k += 1.0;
    }

    2.0 * sum
}

#[cfg(feature = "nalgebra")]
fn onesample_marsaglia_et_al_twosided_pvalue(d: f64, n: f64) -> Result<f64, KSTestError> {
    use nalgebra::DMatrix;
    // Marsaglia, Tsang & Wang (2003)
    // `factorial` can only handle up to 170... could use ln factorial
    if n as usize >= 170 {
        return Err(KSTestError::ExactAndTooLarge);
    }

    let k = (n * d).ceil();
    let m = 2 * k as usize - 1;
    let h = k - n * d;

    let mut mm = DMatrix::<f64>::zeros(m, m);

    // PERF: definitely a better way to fill the matrix. Also could cache the
    // factorial calculations to save time as well
    for j in 0..m {
        for i in 0..m {
            if j == 0 {
                mm[(i, j)] = (1.0 - h.powi(i as i32 + 1)) / factorial::factorial(i as u64 + 1);
                if i == (m - 1) {
                    // bottom left corner
                    mm[(i, j)] = (1.0 - 2.0 * h.powi(m as i32)
                        + (2.0 * h - 1.0).powi(m as i32).max(0.0))
                        / factorial::factorial(m as u64);
                }
            } else if i == (m - 1) {
                mm[(i, j)] = mm[(m - j - 1, 0)]
            } else if (i as isize - j as isize + 1) >= 0 {
                mm[(i, j)] = 1.0 / factorial::factorial((i as isize - j as isize + 1) as u64)
            } else {
                continue;
            }
        }
    }

    let mut t = mm.clone();
    for _ in 0..(n as usize - 1) {
        t = &mm * t;
    }

    Ok(t[(k as usize - 1, k as usize - 1)] * factorial::factorial(n as u64) / n.powi(n as i32))
}

/// Kolmogorov-Smirnov (KS) Test for one sample against [`ContinuousCDF`]
///
/// Returns the statistic and p-value
///
///
/// # Remarks
///
/// see [`KSOneSampleAlternativeMethod`] for additional remarks related to implementation
///
/// # Examples
///
/// ```
/// use statrs::stats_tests::ks_test::{ks_onesample, KSOneSampleAlternativeMethod};
/// use statrs::distribution::Normal;
/// use statrs::stats_tests::NaNPolicy;
///
/// let data: Vec<f64> = (-150..=150).map(|i| i as f64 * 0.01).collect();
///
/// let (statistic, pvalue) = ks_onesample(
///     data.clone(),
///     &Normal::default(),
///     KSOneSampleAlternativeMethod::TwoSidedAsymptotic,
///     NaNPolicy::Error,
/// )
/// .unwrap();
/// ```
pub fn ks_onesample<T>(
    mut data: Vec<f64>,
    distribution: &T,
    method: KSOneSampleAlternativeMethod,
    nan_policy: NaNPolicy,
) -> Result<(f64, f64), KSTestError>
where
    T: ContinuousCDF<f64, f64>,
{
    let has_nans = data.iter().any(|x| x.is_nan());
    if has_nans {
        match nan_policy {
            NaNPolicy::Propogate => {
                return Ok((f64::NAN, f64::NAN));
            }
            NaNPolicy::Error => {
                return Err(KSTestError::SampleContainsNaN);
            }
            NaNPolicy::Emit => {
                data = data.into_iter().filter(|x| !x.is_nan()).collect::<Vec<_>>();
            }
        }
    }

    let n = data.len() as f64;
    if (n as usize) < 1 {
        return Err(KSTestError::SampleTooSmall);
    }

    data.sort_by(|a, b| {
        a.partial_cmp(b)
            .expect("nans should be filtered out by this point so it should always work")
    });

    let theoretical_cdf = data
        .iter()
        .map(|x| distribution.cdf(*x))
        .collect::<Vec<f64>>();

    let d_minus: f64 = zip(&theoretical_cdf, 1..=n as usize)
        .map(|(e, o)| o as f64 / n - e)
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));

    let d_plus: f64 = zip(&theoretical_cdf, 0..n as usize)
        .map(|(e, o)| e - o as f64 / n)
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));

    let (statistic, pvalue) = match method {
        KSOneSampleAlternativeMethod::Less => {
            let statistic = d_plus;
            let pvalue = onesample_birnbaum_tingey_onesided_pvalue(statistic, n);
            (statistic, pvalue)
        }
        KSOneSampleAlternativeMethod::Greater => {
            let statistic = d_minus;
            let pvalue = onesample_birnbaum_tingey_onesided_pvalue(statistic, n);
            (statistic, pvalue)
        }
        #[cfg(feature = "nalgebra")]
        KSOneSampleAlternativeMethod::TwoSidedExact => {
            let mut duplicate_check = data.clone(); // should be for small n so not a big deal
            duplicate_check.dedup(); // data should already be sorted above
            if duplicate_check.len() < n as usize {
                return Err(KSTestError::ExactAndTies);
            }
            let statistic = d_plus.max(d_minus);
            let pvalue = 1.0 - onesample_marsaglia_et_al_twosided_pvalue(statistic, n)?;
            (statistic, pvalue)
        }
        KSOneSampleAlternativeMethod::TwoSidedApproximate => {
            let statistic = d_plus.max(d_minus);
            let pvalue = onesample_birnbaum_tingey_onesided_pvalue(statistic, n) * 2.0;
            (statistic, pvalue)
        }
        KSOneSampleAlternativeMethod::TwoSidedAsymptotic => {
            let statistic = d_plus.max(d_minus);
            let pvalue = onesample_kolmogorov_twosided_pvalue(statistic, n);
            (statistic, pvalue)
        }
    };
    let pvalue = clamp(pvalue, 0.0, 1.0);

    Ok((statistic, pvalue))
}

#[non_exhaustive]
/// Represents the different methods that can be used when calculating the p-value for the
/// two sample KS test.
///
/// Between R and scipy results seem to be different (especially in the non-exact implementation).
/// The one-sided asymptotic methods align closer to the scipy results while the two-sided methods
/// will align closer to R. `TwoSidedExact` should be consistent with both R and scipy (up to a
/// certain sized input).
///
/// The scipy implementation offers the ability to calculate exact p-values for the one-sided test.
/// That functionality is not implemented here as the asymptotic approximation should be sufficient
/// for most use cases.
pub enum KSTwoSampleAlternativeMethod {
    /// uses [Hodges (1957)](doi.org/10.1007/BF02589501) (specifically equation 5.3) to calculate
    /// the p-value for the two-sample, one-sided test.
    LessAsymptotic,
    /// see `LessAsymptotic` for more information
    /// uses [Hodges (1957)](doi.org/10.1007/BF02589501) (specifically equation 5.3) to calculate
    /// the p-value for the two-sample, one-sided test.
    GreaterAsymptotic,
    /// uses [Schröer and Trenkler (1995)](https://doi.org/10.1016/0167-9473(94)00040-P) to
    /// calculate the exact p-value for the two-sample, two-sided test. This paper builds on top of
    /// Hodges (1957) accounting for ties. There are some special edge cases (like the sample sizes
    /// being the same length) where more straightforward and/or efficient solutions could be used,
    /// but those are currently considered too niche to be implemented here.
    TwoSidedExact,
    /// calculates an approximated p-value based on asymptotic approximation described in
    /// Kolmogorov (1933). This approximation takes in a single sample size parameter `n` instead
    /// of the two-sample approaches (which take in `m` and `n`). What is supplied is `m` * `n` /
    /// (`m` + `n`).
    TwoSidedAsymptotic,
}

fn twosample_hodge_equation_53_onesided_pvalue(d: f64, m: f64, n: f64) -> f64 {
    let z = d * ((m * n) / (m + n)).sqrt();
    (-2.0 * z.powi(2) - 2.0 * z / 3.0 * (m + 2.0 * n) / ((m * n) * (m + n)).sqrt()).exp()
}

fn twosample_schroer_and_trenkler_twosided_pvalue(d: f64, m: usize, n: usize) -> f64 {
    let (m, n) = if m > n { (n, m) } else { (m, n) };

    let md = m as f64;
    let nd = n as f64;

    // scale + adjustment for rounding
    let d_scaled = (0.5 + (d * md * nd - 1e-7).floor()) / (md * nd);
    let total_paths = factorial::binomial((m + n) as u64, m as u64);

    let mut a = vec![vec![0.0; n + 1]; m + 1];
    a[0][0] = 1.0;

    for x in 0..=m {
        for y in 0..=n {
            if x == 0 && y == 0 {
                continue;
            }

            // outside of constraint
            if (x as f64 / md - y as f64 / nd).abs() > d_scaled {
                a[x][y] = 0.0;
            } else {
                a[x][y] = (if x > 0 { a[x - 1][y] } else { 0.0 })
                    + (if y > 0 { a[x][y - 1] } else { 0.0 });
            }
        }
    }
    let valid_paths = a[m][n];
    1.0 - valid_paths / total_paths
}

/// Kolmogorov-Smirnov (KS) Test for two data samples
///
/// Returns the statistic and p-value
///
///
/// # Remarks
///
/// see [`KSTwoSampleAlternativeMethod`] for additional remarks related to implementation
///
/// # Examples
///
/// ```
/// use statrs::stats_tests::ks_test::{ks_twosample, KSTwoSampleAlternativeMethod};
/// use statrs::stats_tests::NaNPolicy;
///
/// let data1: Vec<f64> = (0..2000i32).map(|x| x.pow(2) as f64).collect();
/// let data2: Vec<f64> = (-150..2000i32).map(|x| x.pow(2) as f64).collect();
///
/// let (statistic, pvalue) = ks_twosample(
///   data1.clone(),
///   data2.clone(),
///   KSTwoSampleAlternativeMethod::TwoSidedAsymptotic,
///   NaNPolicy::Error,
/// ).unwrap();
/// ```
pub fn ks_twosample(
    mut data1: Vec<f64>,
    mut data2: Vec<f64>,
    method: KSTwoSampleAlternativeMethod,
    nan_policy: NaNPolicy,
) -> Result<(f64, f64), KSTestError> {
    let has_nans1 = data1.iter().any(|x| x.is_nan());
    if has_nans1 {
        match nan_policy {
            NaNPolicy::Propogate => {
                return Ok((f64::NAN, f64::NAN));
            }
            NaNPolicy::Error => {
                return Err(KSTestError::SampleContainsNaN);
            }
            NaNPolicy::Emit => {
                data1 = data1
                    .into_iter()
                    .filter(|x| !x.is_nan())
                    .collect::<Vec<_>>();
            }
        }
    }
    let has_nans2 = data2.iter().any(|x| x.is_nan());
    if has_nans2 {
        match nan_policy {
            NaNPolicy::Propogate => {
                return Ok((f64::NAN, f64::NAN));
            }
            NaNPolicy::Error => {
                return Err(KSTestError::SampleContainsNaN);
            }
            NaNPolicy::Emit => {
                data2 = data2
                    .into_iter()
                    .filter(|x| !x.is_nan())
                    .collect::<Vec<_>>();
            }
        }
    }
    let n1 = data1.len() as f64;
    let n2 = data2.len() as f64;
    if (n1 as usize) < 1 || (n2 as usize) < 1 {
        return Err(KSTestError::SampleTooSmall);
    }
    let n = (n1 as usize).min(n2 as usize);
    let m = (n1 as usize).max(n2 as usize);

    // calculate the test statistic
    data1.sort_by(|a, b| {
        a.partial_cmp(b)
            .expect("nans should be filtered out by this point so it should always work")
    });
    data2.sort_by(|a, b| {
        a.partial_cmp(b)
            .expect("nans should be filtered out by this point so it should always work")
    });
    let mut data_all = [data1.clone(), data2.clone()].concat();
    data_all.sort_by(|a, b| {
        a.partial_cmp(b)
            .expect("nans should be filtered out by this point so it should always work")
    });
    data_all.dedup();

    let mut i = 0;
    let mut j = 0;
    let mut f1 = 0.0;
    let mut f2 = 0.0;
    let mut d_plus: f64 = 0.0;
    let mut d_minus: f64 = 0.0;

    for x in data_all.iter() {
        while i < n1 as usize && &data1[i] == x {
            f1 += 1.0 / n1;
            i += 1;
        }
        while j < n2 as usize && &data2[j] == x {
            f2 += 1.0 / n2;
            j += 1;
        }
        d_plus = d_plus.max(f1 - f2);
        d_minus = d_minus.max(f2 - f1);
    }

    let (statistic, pvalue) = match method {
        KSTwoSampleAlternativeMethod::LessAsymptotic => {
            let statistic = d_minus;
            let pvalue = twosample_hodge_equation_53_onesided_pvalue(statistic, m as f64, n as f64);
            (statistic, pvalue)
        }
        KSTwoSampleAlternativeMethod::GreaterAsymptotic => {
            let statistic = d_plus;
            let pvalue = twosample_hodge_equation_53_onesided_pvalue(statistic, m as f64, n as f64);
            (statistic, pvalue)
        }
        KSTwoSampleAlternativeMethod::TwoSidedExact => {
            if (m * n) > 10000 {
                return Err(KSTestError::ExactAndTooLarge);
            }
            let statistic = d_plus.max(d_minus);
            let pvalue = twosample_schroer_and_trenkler_twosided_pvalue(statistic, m, n);
            (statistic, pvalue)
        }
        KSTwoSampleAlternativeMethod::TwoSidedAsymptotic => {
            let statistic = d_plus.max(d_minus);
            let en = m as f64 * n as f64 / (m as f64 + n as f64);
            let pvalue = onesample_kolmogorov_twosided_pvalue(statistic, en);
            (statistic, pvalue)
        }
    };

    Ok((statistic, pvalue))
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::distribution::{Exp, Normal, Uniform};
    use crate::{prec, statistics::Statistics};

    #[test]
    fn test_ks_onesample_against_scipy() {
        let data = Vec::from([
            0.7, 0.8, 1.1, 2.0, 3.9, 4.2, 4.3, 4.9, 5.1, 5.2, 5.3, 5.5, 5.7, 5.8, 6.0,
        ]);
        let mean = data.iter().mean();

        let (statistic, pvalue) = ks_onesample(
            data.clone(),
            &Exp::new(1.0 / mean).unwrap(),
            KSOneSampleAlternativeMethod::Less,
            NaNPolicy::Error,
        )
        .unwrap();
        // exp_cdf = scipy.stats.expon(scale=mean).cdf
        // scipy.stats.ks_1samp(x=data, cdf=exp_cdf, alternative="less")
        assert!(prec::almost_eq(statistic, 0.35308934158478106, 1e-9));
        assert!(prec::almost_eq(pvalue, 0.01768990758651141, 1e-9));

        let (statistic, pvalue) = ks_onesample(
            data.clone(),
            &Exp::new(1.0 / mean).unwrap(),
            KSOneSampleAlternativeMethod::Greater,
            NaNPolicy::Error,
        )
        .unwrap();
        // scipy.stats.ks_1samp(x=data, cdf=exp_cdf, alternative="greater")
        assert!(prec::almost_eq(statistic, 0.22591345268298602, 1e-9));
        assert!(prec::almost_eq(pvalue, 0.18683781649758202, 1e-9));

        let (statistic, pvalue) = ks_onesample(
            data.clone(),
            &Exp::new(1.0 / mean).unwrap(),
            KSOneSampleAlternativeMethod::TwoSidedAsymptotic,
            NaNPolicy::Error,
        )
        .unwrap();
        // scipy.stats.ks_1samp(x=data, cdf=exp_cdf, alternative="two-sided", method="approx")
        assert!(prec::almost_eq(statistic, 0.35308934158478106, 1e-9));
        assert!(prec::almost_eq(pvalue, 0.047499850721610656, 1e-9));

        let (statistic, pvalue) = ks_onesample(
            data.clone(),
            &Exp::new(1.0 / mean).unwrap(),
            KSOneSampleAlternativeMethod::TwoSidedApproximate,
            NaNPolicy::Error,
        )
        .unwrap();
        // scipy.stats.ks_1samp(x=data, cdf=exp_cdf, alternative="two-sided", method="asymp")
        assert!(prec::almost_eq(statistic, 0.35308934158478106, 1e-9));
        assert!(prec::almost_eq(pvalue, 0.03537981517302282, 1e-9));

        let (statistic, pvalue) = ks_onesample(
            data.clone(),
            &Exp::new(1.0 / mean).unwrap(),
            KSOneSampleAlternativeMethod::TwoSidedExact,
            NaNPolicy::Error,
        )
        .unwrap();
        // scipy.stats.ks_1samp(x=data, cdf=exp_cdf, alternative="two-sided", method="exact")
        assert!(prec::almost_eq(statistic, 0.35308934158478106, 1e-9));
        assert!(prec::almost_eq(pvalue, 0.03537978433644373, 1e-9));
    }
    #[test]
    fn test_ks_onesample_against_r() {
        let data: Vec<f64> = (-150..=150).map(|i| i as f64 * 0.01).collect();

        let (statistic, pvalue) = ks_onesample(
            data.clone(),
            &Normal::default(),
            KSOneSampleAlternativeMethod::Less,
            NaNPolicy::Error,
        )
        .unwrap();
        // ks.test(-150:150/100, "pnorm", alternative="less", exact=TRUE)
        assert!(prec::almost_eq(statistic, 0.066807, 1e-6));
        assert!(prec::almost_eq(pvalue, 0.06508, 1e-3));

        let (statistic, pvalue) = ks_onesample(
            data.clone(),
            &Normal::default(),
            KSOneSampleAlternativeMethod::TwoSidedAsymptotic,
            NaNPolicy::Error,
        )
        .unwrap();
        // ks.test(-150:150/100, "pnorm", alternative="two", exact=NULL)
        assert!(prec::almost_eq(statistic, 0.066807, 1e-6));
        assert!(prec::almost_eq(pvalue, 0.1361, 1e-3));

        // can't test this since n would be too large
        // let (statistic, pvalue) = ks_onesample(
        //     data.clone(),
        //     &Normal::default(),
        //     KSOneSampleAlternativeMethod::TwoSidedExact,
        //     NaNPolicy::Error,
        // )
        // .unwrap();
        // ks.test(-150:150/100, "pnorm", alternative="two", exact=TRUE)
        // assert!(prec::almost_eq(statistic, 0.066807, 1e-6));
        // assert!(prec::almost_eq(pvalue, 0.1301, 1e-3));

        // ensure that the ks test can handle non trivial small sizes
        let data_small_enough: Vec<f64> = (0..140).map(|i| i as f64 * 0.01).collect();
        let (statistic, pvalue) = ks_onesample(
            data_small_enough,
            &Uniform::default(),
            KSOneSampleAlternativeMethod::TwoSidedExact,
            NaNPolicy::Error,
        )
        .unwrap();
        assert!(prec::almost_eq(statistic, 0.28571, 1e-5));
        assert!(prec::almost_eq(pvalue, 1.311e-10, 1e-12));

        let (statistic, pvalue) = ks_onesample(
            data.clone(),
            &Uniform::default(),
            KSOneSampleAlternativeMethod::TwoSidedAsymptotic,
            NaNPolicy::Error,
        )
        .unwrap();
        // ks.test(-150:150/100, "punif", alternative="two", exact=NULL)
        assert!(prec::almost_eq(statistic, 0.50166, 1e-5));
        assert!(prec::almost_eq(pvalue, 0.0, 1e-9));
    }
    #[test]
    fn test_ks_onesample_marsaglia_tsang_wang_2003_exact() {
        // In their example, the value of h appears to be a typo and should really be 0.26
        // which is what is calcualted within the function
        let d = 0.274;
        let n = 10;

        let pvalue = onesample_marsaglia_et_al_twosided_pvalue(d, n as f64).unwrap();
        assert!(prec::almost_eq(pvalue, 0.6284796154565043, 1e-9));
    }
    #[test]
    fn test_ks_onesample_bad_data_data_too_small() {
        let data: Vec<f64> = Vec::new();
        let result = ks_onesample(
            data,
            &Normal::default(),
            KSOneSampleAlternativeMethod::TwoSidedExact,
            NaNPolicy::Error,
        );
        assert_eq!(result, Err(KSTestError::SampleTooSmall));

        let data: Vec<f64> = Vec::from([f64::NAN, f64::NAN]);
        let result = ks_onesample(
            data,
            &Normal::default(),
            KSOneSampleAlternativeMethod::TwoSidedExact,
            NaNPolicy::Emit,
        );
        assert_eq!(result, Err(KSTestError::SampleTooSmall));
    }
    #[test]
    fn test_ks_onesample_bad_data_exact_too_large() {
        let data: Vec<f64> = (-150..=150).map(|i| i as f64 * 0.01).collect();
        let result = ks_onesample(
            data,
            &Normal::default(),
            KSOneSampleAlternativeMethod::TwoSidedExact,
            NaNPolicy::Error,
        );
        assert_eq!(result, Err(KSTestError::ExactAndTooLarge));
    }
    #[test]
    fn test_ks_onesample_bad_data_exact_with_ties() {
        let mut data: Vec<f64> = (-10..=10).map(|i| i as f64 * 0.01).collect();
        data[0] = data[1];
        let result = ks_onesample(
            data,
            &Normal::default(),
            KSOneSampleAlternativeMethod::TwoSidedExact,
            NaNPolicy::Error,
        );
        assert_eq!(result, Err(KSTestError::ExactAndTies));
    }
    #[test]
    fn test_ks_onesample_nan_in_data_w_emit() {
        let data = Vec::from([
            0.7,
            0.8,
            1.1,
            2.0,
            3.9,
            4.2,
            4.3,
            4.9,
            5.1,
            5.2,
            5.3,
            5.5,
            5.7,
            5.8,
            6.0,
            f64::NAN,
        ]);
        let mean = data.iter().filter(|x| !x.is_nan()).mean();

        let (statistic, pvalue) = ks_onesample(
            data.clone(),
            &Exp::new(1.0 / mean).unwrap(),
            KSOneSampleAlternativeMethod::Less,
            NaNPolicy::Emit,
        )
        .unwrap();
        // exp_cdf = scipy.stats.expon(scale=mean).cdf
        // scipy.stats.ks_1samp(x=data, cdf=exp_cdf, alternative="less")
        assert!(prec::almost_eq(statistic, 0.35308934158478106, 1e-9));
        assert!(prec::almost_eq(pvalue, 0.01768990758651141, 1e-9));
    }
    #[test]
    fn test_ks_onesample_nan_in_data_w_propogate() {
        let mut data: Vec<f64> = (-10..=10).map(|i| i as f64 * 0.01).collect();
        data[0] = f64::NAN;
        let (statistic, pvalue) = ks_onesample(
            data,
            &Normal::default(),
            KSOneSampleAlternativeMethod::TwoSidedExact,
            NaNPolicy::Propogate,
        )
        .unwrap();
        assert!(statistic.is_nan());
        assert!(pvalue.is_nan());
    }
    #[test]
    fn test_ks_onesample_nan_in_data_w_error() {
        let mut data: Vec<f64> = (-10..=10).map(|i| i as f64 * 0.01).collect();
        data[0] = f64::NAN;
        let result = ks_onesample(
            data,
            &Normal::default(),
            KSOneSampleAlternativeMethod::TwoSidedExact,
            NaNPolicy::Error,
        );
        assert_eq!(result, Err(KSTestError::SampleContainsNaN));
    }

    #[test]
    fn test_ks_twosample_against_scipy() {
        let data1 = Vec::from([
            0.75857220,
            0.45485367,
            -1.79747176,
            0.01034235,
            0.99762664,
            0.93219930,
            0.11124772,
            -0.01541150,
            -1.16067678,
            -0.49210878,
        ]);
        let data2 = Vec::from([
            -0.009876332,
            0.119263550,
            -2.048604274,
            0.997550468,
            -0.419749716,
            -0.352510481,
            1.196767584,
            0.726644239,
            -0.329687578,
            0.275964060,
            -0.170640773,
            1.834959167,
            -1.083563713,
            1.665032060,
            1.636287642,
        ]);

        let (statistic, pvalue) = ks_twosample(
            data1.clone(),
            data2.clone(),
            KSTwoSampleAlternativeMethod::TwoSidedExact,
            NaNPolicy::Error,
        )
        .unwrap();
        // scipy.stats.ks_2samp(data1, data2)
        assert!(prec::almost_eq(statistic, 0.26666666666666666, 1e-9));
        assert!(prec::almost_eq(pvalue, 0.7315422361996597, 1e-9));
        let (statistic, pvalue) = ks_twosample(
            data2.clone(),
            data1.clone(),
            KSTwoSampleAlternativeMethod::TwoSidedExact,
            NaNPolicy::Error,
        )
        .unwrap();
        // scipy.stats.ks_2samp(data2, data1)
        assert!(prec::almost_eq(statistic, 0.26666666666666666, 1e-9));
        assert!(prec::almost_eq(pvalue, 0.7315422361996597, 1e-9));

        let (statistic, pvalue) = ks_twosample(
            data1.clone(),
            data2.clone(),
            KSTwoSampleAlternativeMethod::LessAsymptotic,
            NaNPolicy::Error,
        )
        .unwrap();
        // scipy.stats.ks_2samp(data1, data2, method="asymp", alternative="less")
        assert!(prec::almost_eq(statistic, 0.1, 1e-9));
        assert!(prec::almost_eq(pvalue, 0.8078867967299911, 1e-9));
        let (statistic, pvalue) = ks_twosample(
            data2.clone(),
            data1.clone(),
            KSTwoSampleAlternativeMethod::LessAsymptotic,
            NaNPolicy::Error,
        )
        .unwrap();
        // scipy.stats.ks_2samp(data2, data1, method="asymp", alternative="less")
        assert!(prec::almost_eq(statistic, 0.26666666666666666, 1e-9));
        assert!(prec::almost_eq(pvalue, 0.33213219147418116, 1e-9));

        let (statistic, pvalue) = ks_twosample(
            data1.clone(),
            data2.clone(),
            KSTwoSampleAlternativeMethod::GreaterAsymptotic,
            NaNPolicy::Error,
        )
        .unwrap();
        // scipy.stats.ks_2samp(data1, data2, method="asymp", alternative="greater")
        assert!(prec::almost_eq(statistic, 0.26666666666666666, 1e-9));
        assert!(prec::almost_eq(pvalue, 0.33213219147418116, 1e-9));
        let (statistic, pvalue) = ks_twosample(
            data2.clone(),
            data1.clone(),
            KSTwoSampleAlternativeMethod::GreaterAsymptotic,
            NaNPolicy::Error,
        )
        .unwrap();
        // scipy.stats.ks_2samp(data2, data1, method="asymp", alternative="greater")
        assert!(prec::almost_eq(statistic, 0.1, 1e-9));
        assert!(prec::almost_eq(pvalue, 0.8078867967299911, 1e-9));

        // NOTE: scipy two-sided asymptotic basically defaults to the one
        // sample "automatic" implementation which may be an exact calculation
        // for smaller `n`s and certain `D` values
        //
        let data1: Vec<f64> = (0..2000i32).map(|x| x.pow(2) as f64).collect();
        let data2: Vec<f64> = (-150..2000i32).map(|x| x.pow(2) as f64).collect();

        let (statistic, pvalue) = ks_twosample(
            data1.clone(),
            data2.clone(),
            KSTwoSampleAlternativeMethod::TwoSidedAsymptotic,
            NaNPolicy::Error,
        )
        .unwrap();
        // scipy.stats.ks_2samp(data1, data2, method="asymp")
        assert!(prec::almost_eq(statistic, 0.06450000000000002, 1e-9));
        assert!(prec::almost_eq(pvalue, 0.0003435848163318721, 1e-4));
        let (statistic, pvalue) = ks_twosample(
            data2.clone(),
            data1.clone(),
            KSTwoSampleAlternativeMethod::TwoSidedAsymptotic,
            NaNPolicy::Error,
        )
        .unwrap();
        // scipy.stats.ks_2samp(data2, data1, method="asymp"")
        assert!(prec::almost_eq(statistic, 0.06450000000000002, 1e-9));
        assert!(prec::almost_eq(pvalue, 0.0003435848163318721, 1e-4));
    }
    #[test]
    fn test_ks_twosample_hodges() {
        // Hodges used for one sided implementaiton and as a
        // foundation for the implementation used, but example tested
        // does not directly related to the implementation. Is an exact test
        // so result should be the same regardless
        let data = "xyxyxxyyxx";
        let mut x = Vec::new();
        let mut y = Vec::new();

        for (i, c) in data.chars().enumerate() {
            if c == 'x' {
                x.push((i + 1) as f64)
            } else {
                y.push((i + 1) as f64)
            }
        }
        let (statistic, pvalue) = ks_twosample(
            x.clone(),
            y.clone(),
            KSTwoSampleAlternativeMethod::TwoSidedExact,
            NaNPolicy::Error,
        )
        .unwrap();
        assert!(prec::almost_eq(statistic, 1.0 / 3.0, 1e-9));
        assert!(prec::almost_eq(pvalue, 97.0 / 105.0, 1e-9));

        let pvalue = twosample_schroer_and_trenkler_twosided_pvalue(1.0 / 3.0, 6, 4);
        assert!(prec::almost_eq(pvalue, 97.0 / 105.0, 1e-9));

        let pvalue = twosample_schroer_and_trenkler_twosided_pvalue(1.0 / 3.0, 4, 6);
        assert!(prec::almost_eq(pvalue, 97.0 / 105.0, 1e-9));
    }
    #[test]
    fn test_ks_twosample_against_r() {
        let data1: Vec<f64> = (0..2000i32).map(|x| x.pow(2) as f64).collect();
        let data2: Vec<f64> = (-150..2000i32).map(|x| x.pow(2) as f64).collect();

        let (statistic, pvalue) = ks_twosample(
            data1.clone(),
            data2.clone(),
            KSTwoSampleAlternativeMethod::TwoSidedAsymptotic,
            NaNPolicy::Error,
        )
        .unwrap();
        // ks.test(data1, data2)
        assert!(prec::almost_eq(statistic, 0.06450000000000002, 1e-9));
        assert!(prec::almost_eq(pvalue, 0.0003604729, 1e-9));
        let (statistic, pvalue) = ks_twosample(
            data2.clone(),
            data1.clone(),
            KSTwoSampleAlternativeMethod::TwoSidedAsymptotic,
            NaNPolicy::Error,
        )
        .unwrap();
        //ks.test(data2, data1)
        assert!(prec::almost_eq(statistic, 0.06450000000000002, 1e-9));
        assert!(prec::almost_eq(pvalue, 0.0003604729, 1e-9));

        // test against R's `chickwts` built-in dataset from an annon source
        let casein = Vec::from([
            368.0, 390.0, 379.0, 260.0, 404.0, 318.0, 352.0, 359.0, 216.0, 222.0, 283.0, 332.0,
        ]);
        let meatmeal = Vec::from([
            325.0, 257.0, 303.0, 315.0, 380.0, 153.0, 263.0, 242.0, 206.0, 344.0, 258.0,
        ]);
        let (statistic, pvalue) = ks_twosample(
            casein.clone(),
            meatmeal.clone(),
            KSTwoSampleAlternativeMethod::TwoSidedExact,
            NaNPolicy::Error,
        )
        .unwrap();
        //ks.test(casein, meatmeal)
        assert!(prec::almost_eq(statistic, 0.4090909, 1e-6));
        assert!(prec::almost_eq(pvalue, 0.1956825, 1e-6));

        let (statistic, pvalue) = ks_twosample(
            meatmeal.clone(),
            casein.clone(),
            KSTwoSampleAlternativeMethod::TwoSidedExact,
            NaNPolicy::Error,
        )
        .unwrap();
        //ks.test(meatmeal, casein)
        assert!(prec::almost_eq(statistic, 0.4090909, 1e-6));
        assert!(prec::almost_eq(pvalue, 0.1956825, 1e-6));
    }
    #[test]
    fn test_ks_twosample_bad_data_exact_too_large() {
        let data1: Vec<f64> = (0..2000i32).map(|x| x.pow(2) as f64).collect();
        let data2: Vec<f64> = (-150..2000i32).map(|x| x.pow(2) as f64).collect();

        let result = ks_twosample(
            data1.clone(),
            data2.clone(),
            KSTwoSampleAlternativeMethod::TwoSidedExact,
            NaNPolicy::Error,
        );
        assert_eq!(result, Err(KSTestError::ExactAndTooLarge));
    }
    #[test]
    fn test_ks_twosample_bad_data_data_too_small() {
        let data1: Vec<f64> = Vec::new();
        let data2 = Vec::from([-0.009876332, 0.119263550, -2.048604274]);
        let result = ks_twosample(
            data1,
            data2,
            KSTwoSampleAlternativeMethod::TwoSidedExact,
            NaNPolicy::Error,
        );
        assert_eq!(result, Err(KSTestError::SampleTooSmall));

        let data1: Vec<f64> = Vec::from([f64::NAN]);
        let data2 = Vec::from([-0.009876332, 0.119263550, -2.048604274]);
        let result = ks_twosample(
            data1,
            data2,
            KSTwoSampleAlternativeMethod::TwoSidedExact,
            NaNPolicy::Emit,
        );
        assert_eq!(result, Err(KSTestError::SampleTooSmall));

        let data1 = Vec::from([-0.009876332, 0.119263550, -2.048604274]);
        let data2: Vec<f64> = Vec::new();
        let result = ks_twosample(
            data1,
            data2,
            KSTwoSampleAlternativeMethod::TwoSidedExact,
            NaNPolicy::Error,
        );
        assert_eq!(result, Err(KSTestError::SampleTooSmall));

        let data1 = Vec::from([-0.009876332, 0.119263550, -2.048604274]);
        let data2: Vec<f64> = Vec::from([f64::NAN]);
        let result = ks_twosample(
            data1,
            data2,
            KSTwoSampleAlternativeMethod::TwoSidedExact,
            NaNPolicy::Emit,
        );
        assert_eq!(result, Err(KSTestError::SampleTooSmall));
    }
    #[test]
    fn test_ks_twosample_nan_in_data_w_emit() {
        let data1 = Vec::from([
            0.75857220,
            0.45485367,
            -1.79747176,
            0.01034235,
            0.99762664,
            0.93219930,
            0.11124772,
            -0.01541150,
            -1.16067678,
            -0.49210878,
            f64::NAN,
        ]);
        let data2 = Vec::from([
            -0.009876332,
            0.119263550,
            -2.048604274,
            0.997550468,
            -0.419749716,
            -0.352510481,
            1.196767584,
            0.726644239,
            -0.329687578,
            0.275964060,
            -0.170640773,
            1.834959167,
            -1.083563713,
            1.665032060,
            1.636287642,
            f64::NAN,
        ]);

        let (statistic, pvalue) = ks_twosample(
            data1.clone(),
            data2.clone(),
            KSTwoSampleAlternativeMethod::TwoSidedExact,
            NaNPolicy::Emit,
        )
        .unwrap();
        assert!(prec::almost_eq(statistic, 0.26666666666666666, 1e-9));
        assert!(prec::almost_eq(pvalue, 0.7315422361996597, 1e-9));
    }
    #[test]
    fn test_ks_twosample_nan_in_data_w_propogate() {
        let data1 = Vec::from([0.75857220, -0.01541150, -1.16067678, -0.49210878, f64::NAN]);
        let data2 = Vec::from([-0.009876332, 0.119263550, -2.048604274]);
        let (statistic, pvalue) = ks_twosample(
            data1.clone(),
            data2.clone(),
            KSTwoSampleAlternativeMethod::TwoSidedExact,
            NaNPolicy::Propogate,
        )
        .unwrap();
        assert!(statistic.is_nan());
        assert!(pvalue.is_nan());

        let data1 = Vec::from([-0.009876332, 0.119263550, -2.048604274]);
        let data2 = Vec::from([0.75857220, -0.01541150, -1.16067678, -0.49210878, f64::NAN]);
        let (statistic, pvalue) = ks_twosample(
            data1.clone(),
            data2.clone(),
            KSTwoSampleAlternativeMethod::TwoSidedExact,
            NaNPolicy::Propogate,
        )
        .unwrap();
        assert!(statistic.is_nan());
        assert!(pvalue.is_nan());
    }
    #[test]
    fn test_ks_twosample_nan_in_data_w_error() {
        let data1 = Vec::from([0.75857220, -0.01541150, -1.16067678, -0.49210878, f64::NAN]);
        let data2 = Vec::from([-0.009876332, 0.119263550, -2.048604274]);
        let result = ks_twosample(
            data1.clone(),
            data2.clone(),
            KSTwoSampleAlternativeMethod::TwoSidedExact,
            NaNPolicy::Error,
        );
        assert_eq!(result, Err(KSTestError::SampleContainsNaN));

        let data1 = Vec::from([-0.009876332, 0.119263550, -2.048604274]);
        let data2 = Vec::from([0.75857220, -0.01541150, -1.16067678, -0.49210878, f64::NAN]);
        let result = ks_twosample(
            data1.clone(),
            data2.clone(),
            KSTwoSampleAlternativeMethod::TwoSidedExact,
            NaNPolicy::Error,
        );
        assert_eq!(result, Err(KSTestError::SampleContainsNaN));
    }
}
