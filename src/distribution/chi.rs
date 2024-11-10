use crate::distribution::{Continuous, ContinuousCDF};
use crate::function::gamma;
use crate::statistics::*;
use std::f64;
use std::num::NonZeroU64;

/// Implements the [Chi](https://en.wikipedia.org/wiki/Chi_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Chi, Continuous};
/// use statrs::statistics::Distribution;
/// use statrs::prec;
///
/// let n = Chi::new(2).unwrap();
/// assert!(prec::almost_eq(n.mean().unwrap(), 1.25331413731550025121, 1e-14));
/// assert!(prec::almost_eq(n.pdf(1.0), 0.60653065971263342360, 1e-15));
/// ```
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Chi {
    freedom: NonZeroU64,
}

/// Represents the errors that can occur when creating a [`Chi`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum ChiError {
    /// The degrees of freedom are zero.
    FreedomInvalid,
}

impl std::fmt::Display for ChiError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ChiError::FreedomInvalid => {
                write!(f, "Degrees of freedom are zero")
            }
        }
    }
}

impl std::error::Error for ChiError {}

impl Chi {
    /// Constructs a new chi distribution
    /// with `freedom` degrees of freedom
    ///
    /// # Errors
    ///
    /// Returns an error if `freedom` is equal to `0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Chi;
    ///
    /// let mut result = Chi::new(2);
    /// assert!(result.is_ok());
    ///
    /// result = Chi::new(0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(freedom: u64) -> Result<Chi, ChiError> {
        match NonZeroU64::new(freedom) {
            Some(freedom) => Ok(Self { freedom }),
            None => Err(ChiError::FreedomInvalid),
        }
    }

    /// Returns the degrees of freedom of the chi distribution.
    /// Guaranteed to be non-zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Chi;
    ///
    /// let n = Chi::new(2).unwrap();
    /// assert_eq!(n.freedom(), 2);
    /// ```
    pub fn freedom(&self) -> u64 {
        self.freedom.get()
    }
}

impl std::fmt::Display for Chi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "χ_{}", self.freedom)
    }
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl ::rand::distributions::Distribution<f64> for Chi {
    fn sample<R: ::rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        (0..self.freedom())
            .fold(0.0, |acc, _| {
                acc + super::normal::sample_unchecked(rng, 0.0, 1.0).powf(2.0)
            })
            .sqrt()
    }
}

impl ContinuousCDF<f64, f64> for Chi {
    /// Calculates the cumulative distribution function for the chi
    /// distribution at `x`.
    ///
    /// # Formula
    ///
    /// ```text
    /// P(k / 2, x^2 / 2)
    /// ```
    ///
    /// where `k` is the degrees of freedom and `P` is
    /// the regularized lower incomplete Gamma function
    fn cdf(&self, x: f64) -> f64 {
        if x == f64::INFINITY {
            1.0
        } else if x <= 0.0 {
            0.0
        } else {
            gamma::gamma_lr(self.freedom() as f64 / 2.0, x * x / 2.0)
        }
    }

    /// Calculates the survival function for the chi
    /// distribution at `x`.
    ///
    /// # Formula
    ///
    /// ```text
    /// P(k / 2, x^2 / 2)
    /// ```
    ///
    /// where `k` is the degrees of freedom and `P` is
    /// the regularized upper incomplete Gamma function
    fn sf(&self, x: f64) -> f64 {
        if x == f64::INFINITY {
            0.0
        } else if x <= 0.0 {
            1.0
        } else {
            gamma::gamma_ur(self.freedom() as f64 / 2.0, x * x / 2.0)
        }
    }
}

impl Min<f64> for Chi {
    /// Returns the minimum value in the domain of the chi distribution
    /// representable by a double precision float
    ///
    /// # Formula
    ///
    /// ```text
    /// 0
    /// ```
    fn min(&self) -> f64 {
        0.0
    }
}

impl Max<f64> for Chi {
    /// Returns the maximum value in the domain of the chi distribution
    /// representable by a double precision float
    ///
    /// # Formula
    ///
    /// ```text
    /// f64::INFINITY
    /// ```
    fn max(&self) -> f64 {
        f64::INFINITY
    }
}

impl Distribution<f64> for Chi {
    /// Returns the mean of the chi distribution
    ///
    /// # Remarks
    ///
    /// Returns `NaN` if `freedom` is `INF`
    ///
    /// # Formula
    ///
    /// ```text
    /// sqrt2 * Γ((k + 1) / 2) / Γ(k / 2)
    /// ```
    ///
    /// where `k` is degrees of freedom and `Γ` is the gamma function
    fn mean(&self) -> Option<f64> {
        let freedom = self.freedom() as f64;

        if self.freedom() > 300 {
            // Large n approximation based on the Stirling series approximation to the Gamma function
            // This avoids call the Gamma function with large arguments and returning NaN
            //
            // Relative accuracy follows O(1/n^4) and at 300 d.o.f. is better than 1e-12
            // For a f32 impl the threshold should be changed to 150
            Some(
                (freedom.sqrt())
                    / ((1.0 + 0.25 / freedom)
                        * (1.0 + 0.03125 / (freedom * freedom))
                        * (1.0 - 0.046875 / (freedom * freedom * freedom))),
            )
        } else {
            let mean = f64::consts::SQRT_2 * gamma::gamma((freedom + 1.0) / 2.0)
                / gamma::gamma(freedom / 2.0);
            Some(mean)
        }
    }

    /// Returns the variance of the chi distribution
    ///
    /// # Remarks
    ///
    /// Returns `NaN` if `freedom` is `INF`
    ///
    /// # Formula
    ///
    /// ```text
    /// k - μ^2
    /// ```
    ///
    /// where `k` is degrees of freedom and `μ` is the mean
    /// of the distribution
    fn variance(&self) -> Option<f64> {
        let mean = self.mean()?;
        Some(self.freedom() as f64 - mean * mean)
    }

    /// Returns the entropy of the chi distribution
    ///
    /// # Remarks
    ///
    /// Returns `None` if `freedom` is `INF`
    ///
    /// # Formula
    ///
    /// ```text
    /// ln(Γ(k / 2)) + 0.5 * (k - ln2 - (k - 1) * ψ(k / 2))
    /// ```
    ///
    /// where `k` is degrees of freedom, `Γ` is the gamma function,
    /// and `ψ` is the digamma function
    fn entropy(&self) -> Option<f64> {
        let freedom = self.freedom() as f64;
        let entr = gamma::ln_gamma(freedom / 2.0)
            + (freedom - (2.0f64).ln() - (freedom - 1.0) * gamma::digamma(freedom / 2.0)) / 2.0;
        Some(entr)
    }

    /// Returns the skewness of the chi distribution
    ///
    /// # Remarks
    ///
    /// Returns `NaN` if `freedom` is `INF`
    ///
    /// # Formula
    ///
    /// ```text
    /// (μ / σ^3) * (1 - 2σ^2)
    /// ```
    /// where `μ` is the mean and `σ` the standard deviation
    /// of the distribution
    fn skewness(&self) -> Option<f64> {
        let sigma = self.std_dev()?;
        let skew = self.mean()? * (1.0 - 2.0 * sigma * sigma) / (sigma * sigma * sigma);
        Some(skew)
    }
}

impl Mode<Option<f64>> for Chi {
    /// Returns the mode for the chi distribution
    ///
    /// # Panics
    ///
    /// If `freedom < 1.0`
    ///
    /// # Formula
    ///
    /// ```text
    /// sqrt(k - 1)
    /// ```
    ///
    /// where `k` is the degrees of freedom
    fn mode(&self) -> Option<f64> {
        Some(((self.freedom() - 1) as f64).sqrt())
    }
}

impl Continuous<f64, f64> for Chi {
    /// Calculates the probability density function for the chi
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// (2^(1 - (k / 2)) * x^(k - 1) * e^(-x^2 / 2)) / Γ(k / 2)
    /// ```
    ///
    /// where `k` is the degrees of freedom and `Γ` is the gamma function
    fn pdf(&self, x: f64) -> f64 {
        if x == f64::INFINITY || x <= 0.0 {
            0.0
        } else if self.freedom() > 160 {
            self.ln_pdf(x).exp()
        } else {
            let freedom = self.freedom() as f64;
            (2.0f64).powf(1.0 - freedom / 2.0) * x.powf(freedom - 1.0) * (-x * x / 2.0).exp()
                / gamma::gamma(freedom / 2.0)
        }
    }

    /// Calculates the log probability density function for the chi distribution
    /// at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// ln((2^(1 - (k / 2)) * x^(k - 1) * e^(-x^2 / 2)) / Γ(k / 2))
    /// ```
    fn ln_pdf(&self, x: f64) -> f64 {
        if x == f64::INFINITY || x <= 0.0 {
            f64::NEG_INFINITY
        } else {
            let freedom = self.freedom() as f64;
            (1.0 - freedom / 2.0) * (2.0f64).ln() + ((freedom - 1.0) * x.ln())
                - x * x / 2.0
                - gamma::ln_gamma(freedom / 2.0)
        }
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::internal::*;
    use crate::testing_boiler;

    testing_boiler!(freedom: u64; Chi; ChiError);

    #[test]
    fn test_create() {
        create_ok(1);
        create_ok(3);
    }

    #[test]
    fn test_bad_create() {
        create_err(0);
    }

    #[test]
    fn test_mean() {
        let mean = |x: Chi| x.mean().unwrap();
        test_absolute(1, 0.7978845608028653558799, 1e-15, mean);
        test_absolute(2, 1.25331413731550025121, 1e-14, mean);
        test_absolute(5, 2.12769216214097428235, 1e-14, mean);
        test_absolute(336, 18.31666925443713, 1e-12, mean);
    }

    #[test]
    fn test_large_dof_mean_not_nan() {
        for i in 1..1000 {
            let mean = Chi::new(i).unwrap().mean().unwrap();
            assert!(!mean.is_nan(), "Chi mean for {i} dof was {mean}");
        }
    }

    #[test]
    fn test_variance() {
        let variance = |x: Chi| x.variance().unwrap();
        test_absolute(1, 0.3633802276324186569245, 1e-15, variance);
        test_absolute(2, 0.42920367320510338077, 1e-14, variance);
        test_absolute(3, 0.4535209105296746277, 1e-14, variance);
    }

    #[test]
    fn test_entropy() {
        let entropy = |x: Chi| x.entropy().unwrap();
        test_absolute(1, 0.7257913526447274323631, 1e-15, entropy);
        test_absolute(2, 0.9420342421707937755946, 1e-15, entropy);
        test_absolute(3, 0.99615419810620560239, 1e-14, entropy);
    }

    #[test]
    fn test_skewness() {
        let skewness = |x: Chi| x.skewness().unwrap();
        test_absolute(1, 0.995271746431156042444, 1e-14, skewness);
        test_absolute(3, 0.485692828049590809, 1e-12, skewness);
    }

    #[test]
    fn test_mode() {
        let mode = |x: Chi| x.mode().unwrap();
        test_exact(1, 0.0, mode);
        test_exact(2, 1.0, mode);
        test_exact(3, f64::consts::SQRT_2, mode);
    }

    #[test]
    fn test_min_max() {
        let min = |x: Chi| x.min();
        let max = |x: Chi| x.max();
        test_exact(1, 0.0, min);
        test_exact(2, 0.0, min);
        test_exact(2, 0.0, min);
        test_exact(3, 0.0, min);
        test_exact(1, f64::INFINITY, max);
        test_exact(2, f64::INFINITY, max);
        test_exact(2, f64::INFINITY, max);
        test_exact(3, f64::INFINITY, max);
    }

    #[test]
    fn test_pdf() {
        let pdf = |arg: f64| move |x: Chi| x.pdf(arg);
        test_exact(1, 0.0, pdf(0.0));
        test_absolute(1, 0.79390509495402353102, 1e-15, pdf(0.1));
        test_absolute(1, 0.48394144903828669960, 1e-15, pdf(1.0));
        test_absolute(1, 2.1539520085086552718e-7, 1e-22, pdf(5.5));
        test_exact(1, 0.0, pdf(f64::INFINITY));
        test_exact(2, 0.0, pdf(0.0));
        test_absolute(2, 0.099501247919268231335, 1e-16, pdf(0.1));
        test_absolute(2, 0.60653065971263342360, 1e-15, pdf(1.0));
        test_absolute(2, 1.4847681768496578863e-6, 1e-21, pdf(5.5));
        test_exact(2, 0.0, pdf(f64::INFINITY));
        test_exact(2, 0.0, pdf(0.0));
        test_exact(2, 0.0, pdf(f64::INFINITY));
        test_absolute(170, 0.5644678498668440878, 1e-13, pdf(13.0));
    }

    #[test]
    fn test_neg_pdf() {
        let pdf = |arg: f64| move |x: Chi| x.pdf(arg);
        test_exact(1, 0.0, pdf(-1.0));
    }

    #[test]
    fn test_ln_pdf() {
        let ln_pdf = |arg: f64| move |x: Chi| x.ln_pdf(arg);
        test_exact(1, f64::NEG_INFINITY, ln_pdf(0.0));
        test_absolute(1, -0.23079135264472743236, 1e-15, ln_pdf(0.1));
        test_absolute(1, -0.72579135264472743236, 1e-15, ln_pdf(1.0));
        test_absolute(1, -15.350791352644727432, 1e-14, ln_pdf(5.5));
        test_exact(1, f64::NEG_INFINITY, ln_pdf(f64::INFINITY));
        test_exact(2, f64::NEG_INFINITY, ln_pdf(0.0));
        test_absolute(2, -2.3075850929940456840, 1e-15, ln_pdf(0.1));
        test_absolute(2, -0.5, 1e-15, ln_pdf(1.0));
        test_absolute(2, -13.420251907761574765, 1e-15, ln_pdf(5.5));
        test_exact(2, f64::NEG_INFINITY, ln_pdf(f64::INFINITY));
        test_exact(2, f64::NEG_INFINITY, ln_pdf(0.0));
        test_exact(2, f64::NEG_INFINITY, ln_pdf(f64::INFINITY));
        test_absolute(170, -0.57187185030600516424237, 1e-13, ln_pdf(13.0));
    }

    #[test]
    fn test_neg_ln_pdf() {
        let ln_pdf = |arg: f64| move |x: Chi| x.ln_pdf(arg);
        test_exact(1, f64::NEG_INFINITY, ln_pdf(-1.0));
    }

    #[test]
    fn test_cdf() {
        let cdf = |arg: f64| move |x: Chi| x.cdf(arg);
        test_exact(1, 0.0, cdf(0.0));
        test_absolute(1, 0.079655674554057962931, 1e-16, cdf(0.1));
        test_absolute(1, 0.68268949213708589717, 1e-15, cdf(1.0));
        test_exact(1, 0.99999996202087506822, cdf(5.5));
        test_exact(1, 1.0, cdf(f64::INFINITY));
        test_exact(2, 0.0, cdf(0.0));
        test_absolute(2, 0.0049875208073176866474, 1e-17, cdf(0.1));
        test_exact(2, 1.0, cdf(f64::INFINITY));
        test_exact(2, 0.0, cdf(0.0));
        test_exact(2, 1.0, cdf(f64::INFINITY));
    }

    #[test]
    fn test_sf() {
        let sf = |arg: f64| move |x: Chi| x.sf(arg);
        test_exact(1, 1.0, sf(0.0));
        test_absolute(1, 0.920344325445942, 1e-16, sf(0.1));
        test_absolute(1, 0.31731050786291404, 1e-15, sf(1.0));
        test_absolute(1, 3.797912493177544e-8, 1e-15, sf(5.5));
        test_exact(1, 0.0, sf(f64::INFINITY));
        test_exact(2, 1.0, sf(0.0));
        test_absolute(2, 0.9950124791926823, 1e-17, sf(0.1));
        test_absolute(2, 0.6065306597126333, 1e-15, sf(1.0));
        test_absolute(2, 2.699578503363014e-7, 1e-15, sf(5.5));
        test_exact(2, 0.0, sf(f64::INFINITY));
        test_exact(2, 1.0, sf(0.0));
        test_exact(2, 0.0, sf(f64::INFINITY));
    }

    #[test]
    fn test_neg_cdf() {
        let cdf = |arg: f64| move |x: Chi| x.cdf(arg);
        test_exact(1, 0.0, cdf(-1.0));
    }

    #[test]
    fn test_neg_sf() {
        let sf = |arg: f64| move |x: Chi| x.sf(arg);
        test_exact(1, 1.0, sf(-1.0));
    }

    #[test]
    fn test_continuous() {
        test::check_continuous_distribution(&create_ok(1), 0.0, 10.0);
        test::check_continuous_distribution(&create_ok(2), 0.0, 10.0);
        test::check_continuous_distribution(&create_ok(5), 0.0, 10.0);
    }
}
