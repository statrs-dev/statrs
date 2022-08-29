use crate::distribution::{Continuous, ContinuousCDF};
use crate::function::{beta, gamma};
use crate::is_zero;
use crate::statistics::*;
use crate::{Result, StatsError};
use rand::Rng;
use std::f64;

/// Implements the [Student's
/// T](https://en.wikipedia.org/wiki/Student%27s_t-distribution) distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{StudentsT, Continuous};
/// use statrs::statistics::Distribution;
/// use statrs::prec;
///
/// let n = StudentsT::new(0.0, 1.0, 2.0).unwrap();
/// assert_eq!(n.mean().unwrap(), 0.0);
/// assert!(prec::almost_eq(n.pdf(0.0), 0.353553390593274, 1e-15));
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct StudentsT {
    location: f64,
    scale: f64,
    freedom: f64,
}

impl StudentsT {
    /// Constructs a new student's t-distribution with location `location`,
    /// scale `scale`,
    /// and `freedom` freedom.
    ///
    /// # Errors
    ///
    /// Returns an error if any of `location`, `scale`, or `freedom` are `NaN`.
    /// Returns an error if `scale <= 0.0` or `freedom <= 0.0`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::StudentsT;
    ///
    /// let mut result = StudentsT::new(0.0, 1.0, 2.0);
    /// assert!(result.is_ok());
    ///
    /// result = StudentsT::new(0.0, 0.0, 0.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(location: f64, scale: f64, freedom: f64) -> Result<StudentsT> {
        let is_nan = location.is_nan() || scale.is_nan() || freedom.is_nan();
        if is_nan || scale <= 0.0 || freedom <= 0.0 {
            Err(StatsError::BadParams)
        } else {
            Ok(StudentsT {
                location,
                scale,
                freedom,
            })
        }
    }

    /// Returns the location of the student's t-distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::StudentsT;
    ///
    /// let n = StudentsT::new(0.0, 1.0, 2.0).unwrap();
    /// assert_eq!(n.location(), 0.0);
    /// ```
    pub fn location(&self) -> f64 {
        self.location
    }

    /// Returns the scale of the student's t-distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::StudentsT;
    ///
    /// let n = StudentsT::new(0.0, 1.0, 2.0).unwrap();
    /// assert_eq!(n.scale(), 1.0);
    /// ```
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Returns the freedom of the student's t-distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::StudentsT;
    ///
    /// let n = StudentsT::new(0.0, 1.0, 2.0).unwrap();
    /// assert_eq!(n.freedom(), 2.0);
    /// ```
    pub fn freedom(&self) -> f64 {
        self.freedom
    }
}

impl ::rand::distributions::Distribution<f64> for StudentsT {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> f64 {
        // based on method 2, section 5 in chapter 9 of L. Devroye's
        // "Non-Uniform Random Variate Generation"
        let gamma = super::gamma::sample_unchecked(r, 0.5 * self.freedom, 0.5);
        super::normal::sample_unchecked(
            r,
            self.location,
            self.scale * (self.freedom / gamma).sqrt(),
        )
    }
}

impl ContinuousCDF<f64, f64> for StudentsT {
    /// Calculates the cumulative distribution function for the student's
    /// t-distribution
    /// at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// if x < μ {
    ///     (1 / 2) * I(t, v / 2, 1 / 2)
    /// } else {
    ///     1 - (1 / 2) * I(t, v / 2, 1 / 2)
    /// }
    /// ```
    ///
    /// where `t = v / (v + k^2)`, `k = (x - μ) / σ`, `μ` is the location,
    /// `σ` is the scale, `v` is the freedom, and `I` is the regularized
    /// incomplete beta function
    fn cdf(&self, x: f64) -> f64 {
        if self.freedom.is_infinite() {
            super::normal::cdf_unchecked(x, self.location, self.scale)
        } else {
            let k = (x - self.location) / self.scale;
            let h = self.freedom / (self.freedom + k * k);
            let ib = 0.5 * beta::beta_reg(self.freedom / 2.0, 0.5, h);
            if x <= self.location {
                ib
            } else {
                1.0 - ib
            }
        }
    }

    /// Calculates the cumulative distribution function for the student's
    /// t-distribution
    /// at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// if x < μ {
    ///     1 - (1 / 2) * I(t, v / 2, 1 / 2)
    /// } else {
    ///     (1 / 2) * I(t, v / 2, 1 / 2)
    /// }
    /// ```
    ///
    /// where `t = v / (v + k^2)`, `k = (x - μ) / σ`, `μ` is the location,
    /// `σ` is the scale, `v` is the freedom, and `I` is the regularized
    /// incomplete beta function
    fn sf(&self, x: f64) -> f64 {
        if self.freedom.is_infinite() {
            super::normal::sf_unchecked(x, self.location, self.scale)
        } else {
            let k = (x - self.location) / self.scale;
            let h = self.freedom / (self.freedom + k * k);
            let ib = 0.5 * beta::beta_reg(self.freedom / 2.0, 0.5, h);
            if x <= self.location {
                1.0 - ib
            } else {
                ib
            }
        }
    }

    /// Calculates the inverse cumulative distribution function for the
    /// Student's T-distribution at `x`
    fn inverse_cdf(&self, x: f64) -> f64 {
        // first calculate inverse_cdf for normal Student's T
        assert!((0.0..=1.0).contains(&x));
        let x1 = if x >= 0.5 { 1.0 - x } else { x };
        let a = 0.5 * self.freedom;
        let b = 0.5;
        let mut y = beta::inv_beta_reg(a, b, 2.0 * x1);
        y = (self.freedom * (1. - y) / y).sqrt();
        y = if x >= 0.5 { y } else { -y };
        // generalised Student's T is related to normal Student's T by `Y = μ + σ X`
        // where `X` is distributed as Student's T, so this result has to be scaled and shifted back
        // formally: F_Y(t) = P(Y <= t) = P(X <= (t - μ) / σ) = F_X((t - μ) / σ)
        // F_Y^{-1}(p) = inf { t' | F_Y(t') >= p } = inf { t' = μ + σ t | F_X((t' - μ) / σ) >= p }
        // because scale is positive: loc + scale * t is strictly monotonic function
        // = μ + σ inf { t | F_X(t) >= p } = μ + σ F_X^{-1}(p)
        self.location + self.scale * y
    }
}

impl Min<f64> for StudentsT {
    /// Returns the minimum value in the domain of the student's t-distribution
    /// representable by a double precision float
    ///
    /// # Formula
    ///
    /// ```ignore
    /// -INF
    /// ```
    fn min(&self) -> f64 {
        f64::NEG_INFINITY
    }
}

impl Max<f64> for StudentsT {
    /// Returns the maximum value in the domain of the student's t-distribution
    /// representable by a double precision float
    ///
    /// # Formula
    ///
    /// ```ignore
    /// INF
    /// ```
    fn max(&self) -> f64 {
        f64::INFINITY
    }
}

impl Distribution<f64> for StudentsT {
    /// Returns the mean of the student's t-distribution
    ///
    /// # None
    ///
    /// If `freedom <= 1.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// μ
    /// ```
    ///
    /// where `μ` is the location
    fn mean(&self) -> Option<f64> {
        if self.freedom <= 1.0 {
            None
        } else {
            Some(self.location)
        }
    }
    /// Returns the variance of the student's t-distribution
    ///
    /// # None
    ///
    /// If `freedom <= 2.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// if v == INF {
    ///     Some(σ^2)
    /// } else if freedom > 2.0 {
    ///     Some(v * σ^2 / (v - 2))
    /// } else {
    ///     None
    /// }
    /// ```
    ///
    /// where `σ` is the scale and `v` is the freedom
    fn variance(&self) -> Option<f64> {
        if self.freedom.is_infinite() {
            Some(self.scale * self.scale)
        } else if self.freedom > 2.0 {
            Some(self.freedom * self.scale * self.scale / (self.freedom - 2.0))
        } else {
            None
        }
    }
    /// Returns the entropy for the student's t-distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// - ln(σ) + (v + 1) / 2 * (ψ((v + 1) / 2) - ψ(v / 2)) + ln(sqrt(v) * B(v / 2, 1 /
    /// 2))
    /// ```
    ///
    /// where `σ` is the scale, `v` is the freedom, `ψ` is the digamma function, and `B` is the
    /// beta function
    fn entropy(&self) -> Option<f64> {
        // generalised Student's T is related to normal Student's T by `Y = μ + σ X`
        // where `X` is distributed as Student's T, plugging into the definition
        // of entropy shows scaling affects the entropy by an additive constant `- ln σ`
        let shift = -self.scale.ln();
        let result = (self.freedom + 1.0) / 2.0
            * (gamma::digamma((self.freedom + 1.0) / 2.0) - gamma::digamma(self.freedom / 2.0))
            + (self.freedom.sqrt() * beta::beta(self.freedom / 2.0, 0.5)).ln();
        Some(result + shift)
    }
    /// Returns the skewness of the student's t-distribution
    ///
    /// # None
    ///
    /// If `x <= 3.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 0
    /// ```
    fn skewness(&self) -> Option<f64> {
        if self.freedom <= 3.0 {
            None
        } else {
            Some(0.0)
        }
    }
}

impl Median<f64> for StudentsT {
    /// Returns the median of the student's t-distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// μ
    /// ```
    ///
    /// where `μ` is the location
    fn median(&self) -> f64 {
        self.location
    }
}

impl Mode<Option<f64>> for StudentsT {
    /// Returns the mode of the student's t-distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// μ
    /// ```
    ///
    /// where `μ` is the location
    fn mode(&self) -> Option<f64> {
        Some(self.location)
    }
}

impl Continuous<f64, f64> for StudentsT {
    /// Calculates the probability density function for the student's
    /// t-distribution
    /// at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// Γ((v + 1) / 2) / (sqrt(vπ) * Γ(v / 2) * σ) * (1 + k^2 / v)^(-1 / 2 * (v
    /// + 1))
    /// ```
    ///
    /// where `k = (x - μ) / σ`, `μ` is the location, `σ` is the scale, `v` is
    /// the freedom,
    /// and `Γ` is the gamma function
    fn pdf(&self, x: f64) -> f64 {
        if x.is_infinite() {
            0.0
        } else if self.freedom >= 1e8 {
            super::normal::pdf_unchecked(x, self.location, self.scale)
        } else {
            let d = (x - self.location) / self.scale;
            (gamma::ln_gamma((self.freedom + 1.0) / 2.0) - gamma::ln_gamma(self.freedom / 2.0))
                .exp()
                * (1.0 + d * d / self.freedom).powf(-0.5 * (self.freedom + 1.0))
                / (self.freedom * f64::consts::PI).sqrt()
                / self.scale
        }
    }

    /// Calculates the log probability density function for the student's
    /// t-distribution
    /// at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln(Γ((v + 1) / 2) / (sqrt(vπ) * Γ(v / 2) * σ) * (1 + k^2 / v)^(-1 / 2 *
    /// (v + 1)))
    /// ```
    ///
    /// where `k = (x - μ) / σ`, `μ` is the location, `σ` is the scale, `v` is
    /// the freedom,
    /// and `Γ` is the gamma function
    fn ln_pdf(&self, x: f64) -> f64 {
        if x.is_infinite() {
            f64::NEG_INFINITY
        } else if self.freedom >= 1e8 {
            super::normal::ln_pdf_unchecked(x, self.location, self.scale)
        } else {
            let d = (x - self.location) / self.scale;
            gamma::ln_gamma((self.freedom + 1.0) / 2.0)
                - 0.5 * ((self.freedom + 1.0) * (1.0 + d * d / self.freedom).ln())
                - gamma::ln_gamma(self.freedom / 2.0)
                - 0.5 * (self.freedom * f64::consts::PI).ln()
                - self.scale.ln()
        }
    }
}

#[cfg(all(test, feature = "nightly"))]
mod tests {
    use crate::consts::ACC;
    use crate::distribution::internal::*;
    use crate::distribution::{Continuous, ContinuousCDF, StudentsT};
    use crate::statistics::*;
    use crate::testing_boiler;
    use std::panic;

    testing_boiler!((f64, f64, f64), StudentsT);

    #[test]
    fn test_create() {
        try_create((0.0, 0.1, 1.0));
        try_create((0.0, 1.0, 1.0));
        try_create((-5.0, 1.0, 3.0));
        try_create((10.0, 10.0, f64::INFINITY));
    }

    // #[test]
    // fn foo() {
    //     let dist = StudentsT::new(0.0,1.0,1.0).unwrap();
    //     dbg!(dist.mean());
    // }

    #[test]
    fn test_bad_create() {
        bad_create_case((f64::NAN, 1.0, 1.0));
        bad_create_case((0.0, f64::NAN, 1.0));
        bad_create_case((0.0, 1.0, f64::NAN));
        bad_create_case((0.0, -10.0, 1.0));
        bad_create_case((0.0, 10.0, -1.0));
    }

    #[test]
    fn test_mean() {
        let mean = |x: StudentsT| x.mean().unwrap();
        test_case((0.0, 1.0, 3.0), 0.0, mean);
        test_case((0.0, 10.0, 2.0), 0.0, mean);
        test_case((0.0, 10.0, f64::INFINITY), 0.0, mean);
        test_case((-5.0, 100.0, 1.5), -5.0, mean);
        let mean = |x: StudentsT| x.mean();
        test_none((0.0, 1.0, 1.0), mean);
        test_none((0.0, 0.1, 1.0), mean);
        test_none((0.0, 10.0, 1.0), mean);
        test_none((10.0, 1.0, 1.0), mean);
        test_none((0.0, f64::INFINITY, 1.0), mean);
    }

    #[test]
    #[should_panic]
    fn test_mean_freedom_lte_1() {
        let mean = |x: StudentsT| x.mean().unwrap();
        get_value((1.0, 1.0, 0.5), mean);
    }

    #[test]
    fn test_variance() {
        let variance = |x: StudentsT| x.variance().unwrap();
        test_case((0.0, 1.0, 3.0), 3.0, variance);
        test_case((0.0, 10.0, 2.5), 500.0, variance);
        test_case((10.0, 1.0, 2.5), 5.0, variance);
        let variance = |x: StudentsT| x.variance();
        test_none((0.0, 10.0, 2.0), variance);
        test_none((0.0, 1.0, 1.0), variance);
        test_none((0.0, 0.1, 1.0), variance);
        test_none((0.0, 10.0, 1.0), variance);
        test_none((10.0, 1.0, 1.0), variance);
        test_none((-5.0, 100.0, 1.5), variance);
        test_none((0.0, f64::INFINITY, 1.0), variance);
    }

    #[test]
    #[should_panic]
    fn test_variance_freedom_lte1() {
        let variance = |x: StudentsT| x.variance().unwrap();
        get_value((1.0, 1.0, 0.5), variance);
    }

    // TODO: valid skewness tests
    #[test]
    #[should_panic]
    fn test_skewness_freedom_lte_3() {
        let skewness = |x: StudentsT| x.skewness().unwrap();
        get_value((1.0, 1.0, 1.0), skewness);
    }

    #[test]
    fn test_mode() {
        let mode = |x: StudentsT| x.mode().unwrap();
        test_case((0.0, 1.0, 1.0), 0.0, mode);
        test_case((0.0, 0.1, 1.0), 0.0, mode);
        test_case((0.0, 1.0, 3.0), 0.0, mode);
        test_case((0.0, 10.0, 1.0), 0.0, mode);
        test_case((0.0, 10.0, 2.0), 0.0, mode);
        test_case((0.0, 10.0, 2.5), 0.0, mode);
        test_case((0.0, 10.0, f64::INFINITY), 0.0, mode);
        test_case((10.0, 1.0, 1.0), 10.0, mode);
        test_case((10.0, 1.0, 2.5), 10.0, mode);
        test_case((-5.0, 100.0, 1.5), -5.0, mode);
        test_case((0.0, f64::INFINITY, 1.0), 0.0, mode);
    }

    #[test]
    fn test_median() {
        let median = |x: StudentsT| x.median();
        test_case((0.0, 1.0, 1.0), 0.0, median);
        test_case((0.0, 0.1, 1.0), 0.0, median);
        test_case((0.0, 1.0, 3.0), 0.0, median);
        test_case((0.0, 10.0, 1.0), 0.0, median);
        test_case((0.0, 10.0, 2.0), 0.0, median);
        test_case((0.0, 10.0, 2.5), 0.0, median);
        test_case((0.0, 10.0, f64::INFINITY), 0.0, median);
        test_case((10.0, 1.0, 1.0), 10.0, median);
        test_case((10.0, 1.0, 2.5), 10.0, median);
        test_case((-5.0, 100.0, 1.5), -5.0, median);
        test_case((0.0, f64::INFINITY, 1.0), 0.0, median);
    }

    #[test]
    fn test_min_max() {
        let min = |x: StudentsT| x.min();
        let max = |x: StudentsT| x.max();
        test_case((0.0, 1.0, 1.0), f64::NEG_INFINITY, min);
        test_case((2.5, 100.0, 1.5), f64::NEG_INFINITY, min);
        test_case((10.0, f64::INFINITY, 3.5), f64::NEG_INFINITY, min);
        test_case((0.0, 1.0, 1.0), f64::INFINITY, max);
        test_case((2.5, 100.0, 1.5), f64::INFINITY, max);
        test_case((10.0, f64::INFINITY, 5.5), f64::INFINITY, max);
    }

    #[test]
    fn test_pdf() {
        let pdf = |arg: f64| move |x: StudentsT| x.pdf(arg);
        test_case((0.0, 1.0, 1.0), 0.318309886183791, pdf(0.0));
        test_case((0.0, 1.0, 1.0), 0.159154943091895, pdf(1.0));
        test_case((0.0, 1.0, 1.0), 0.159154943091895, pdf(-1.0));
        test_case((0.0, 1.0, 1.0), 0.063661977236758, pdf(2.0));
        test_case((0.0, 1.0, 1.0), 0.063661977236758, pdf(-2.0));
        test_case((0.0, 1.0, 2.0), 0.353553390593274, pdf(0.0));
        test_case((0.0, 1.0, 2.0), 0.192450089729875, pdf(1.0));
        test_case((0.0, 1.0, 2.0), 0.192450089729875, pdf(-1.0));
        test_case((0.0, 1.0, 2.0), 0.068041381743977, pdf(2.0));
        test_case((0.0, 1.0, 2.0), 0.068041381743977, pdf(-2.0));
        test_case((0.0, 1.0, f64::INFINITY), 0.398942280401433, pdf(0.0));
        test_case((0.0, 1.0, f64::INFINITY), 0.241970724519143, pdf(1.0));
        test_case((0.0, 1.0, f64::INFINITY), 0.053990966513188, pdf(2.0));
    }

    #[test]
    fn test_ln_pdf() {
        let ln_pdf = |arg: f64| move |x: StudentsT| x.ln_pdf(arg);
        test_case((0.0, 1.0, 1.0), -1.144729885849399, ln_pdf(0.0));
        test_case((0.0, 1.0, 1.0), -1.837877066409348, ln_pdf(1.0));
        test_case((0.0, 1.0, 1.0), -1.837877066409348, ln_pdf(-1.0));
        test_case((0.0, 1.0, 1.0), -2.754167798283503, ln_pdf(2.0));
        test_case((0.0, 1.0, 1.0), -2.754167798283503, ln_pdf(-2.0));
        test_case((0.0, 1.0, 2.0), -1.039720770839917, ln_pdf(0.0));
        test_case((0.0, 1.0, 2.0), -1.647918433002166, ln_pdf(1.0));
        test_case((0.0, 1.0, 2.0), -1.647918433002166, ln_pdf(-1.0));
        test_case((0.0, 1.0, 2.0), -2.687639203842085, ln_pdf(2.0));
        test_case((0.0, 1.0, 2.0), -2.687639203842085, ln_pdf(-2.0));
        test_case((0.0, 1.0, f64::INFINITY), -0.918938533204672, ln_pdf(0.0));
        test_case((0.0, 1.0, f64::INFINITY), -1.418938533204674, ln_pdf(1.0));
        test_case((0.0, 1.0, f64::INFINITY), -2.918938533204674, ln_pdf(2.0));
    }

    #[test]
    fn test_cdf() {
        let cdf = |arg: f64| move |x: StudentsT| x.cdf(arg);
        test_case((0.0, 1.0, 1.0), 0.5, cdf(0.0));
        test_case((0.0, 1.0, 1.0), 0.75, cdf(1.0));
        test_case((0.0, 1.0, 1.0), 0.25, cdf(-1.0));
        test_case((0.0, 1.0, 1.0), 0.852416382349567, cdf(2.0));
        test_case((0.0, 1.0, 1.0), 0.147583617650433, cdf(-2.0));
        test_case((0.0, 1.0, 2.0), 0.5, cdf(0.0));
        test_case((0.0, 1.0, 2.0), 0.788675134594813, cdf(1.0));
        test_case((0.0, 1.0, 2.0), 0.211324865405187, cdf(-1.0));
        test_case((0.0, 1.0, 2.0), 0.908248290463863, cdf(2.0));
        test_case((0.0, 1.0, 2.0), 0.091751709536137, cdf(-2.0));
        test_case((0.0, 1.0, f64::INFINITY), 0.5, cdf(0.0));

        // TODO: these are curiously low accuracy and should be re-examined
        test_case((0.0, 1.0, f64::INFINITY), 0.841344746068543, cdf(1.0));
        test_case((0.0, 1.0, f64::INFINITY), 0.977249868051821, cdf(2.0));
    }


    #[test]
    fn test_sf() {
        let sf = |arg: f64| move |x: StudentsT| x.sf(arg);
        test_case((0.0, 1.0, 1.0), 0.5, sf(0.0));
        test_case((0.0, 1.0, 1.0), 0.25, sf(1.0));
        test_case((0.0, 1.0, 1.0), 0.75, sf(-1.0));
        test_case((0.0, 1.0, 1.0), 0.147583617650433, sf(2.0));
        test_case((0.0, 1.0, 1.0), 0.852416382349566, sf(-2.0));
        test_case((0.0, 1.0, 2.0), 0.5, sf(0.0));
        test_case((0.0, 1.0, 2.0), 0.211324865405186, sf(1.0));
        test_case((0.0, 1.0, 2.0), 0.788675134594813, sf(-1.0));
        test_case((0.0, 1.0, 2.0), 0.091751709536137, sf(2.0));
        test_case((0.0, 1.0, 2.0), 0.908248290463862, sf(-2.0));
        test_case((0.0, 1.0, f64::INFINITY), 0.5, sf(0.0));

        // TODO: these are curiously low accuracy and should be re-examined
        test_case((0.0, 1.0, f64::INFINITY), 0.158655253945057, sf(1.0));
        test_case((0.0, 1.0, f64::INFINITY), 0.022750131947162, sf(2.0));
    }

    #[test]
    fn test_continuous() {
        test::check_continuous_distribution(&try_create((0.0, 1.0, 3.0)), -30.0, 30.0);
        test::check_continuous_distribution(&try_create((0.0, 1.0, 10.0)), -10.0, 10.0);
        test::check_continuous_distribution(&try_create((20.0, 0.5, 10.0)), 10.0, 30.0);
    }

    #[test]
    fn test_inv_cdf() {
        let test = |x: f64, freedom: f64, expected: f64| {
            use approx::*;
            let d = StudentsT::new(0., 1., freedom).unwrap();
            // Checks that left == right to 4 significant figures, unlike
            // test_almost() which uses decimal places
            assert_relative_eq!(d.inverse_cdf(x), expected, max_relative = 0.001);
        };

        // This test checks our implementation against the whole t-table
        // copied from https://en.wikipedia.org/wiki/Student's_t-distribution

        test(0.75, 1.0, 1.000);
        test(0.8, 1.0, 1.376);
        test(0.85, 1.0, 1.963);
        test(0.9, 1.0, 3.078);
        test(0.95, 1.0, 6.314);
        test(0.975, 1.0, 12.71);
        test(0.99, 1.0, 31.82);
        test(0.995, 1.0, 63.66);
        test(0.9975, 1.0, 127.3);
        test(0.999, 1.0, 318.3);
        test(0.9995, 1.0, 636.6);

        test(0.75, 002.0, 0.816);
        // TODO: investigate
        // test(0.8, 002.0, 1.080);  // We get 1.061 for some reason...
        test(0.85, 002.0, 1.386);
        test(0.9, 002.0, 1.886);
        test(0.95, 002.0, 2.920);
        test(0.975, 002.0, 4.303);
        test(0.99, 002.0, 6.965);
        test(0.995, 002.0, 9.925);
        test(0.9975, 002.0, 14.09);
        test(0.999, 002.0, 22.33);
        test(0.9995, 002.0, 31.60);

        test(0.75, 003.0, 0.765);
        test(0.8, 003.0, 0.978);
        test(0.85, 003.0, 1.250);
        test(0.9, 003.0, 1.638);
        test(0.95, 003.0, 2.353);
        test(0.975, 003.0, 3.182);
        test(0.99, 003.0, 4.541);
        test(0.995, 003.0, 5.841);
        test(0.9975, 003.0, 7.453);
        test(0.999, 003.0, 10.21);
        test(0.9995, 003.0, 12.92);

        test(0.75, 004.0, 0.741);
        test(0.8, 004.0, 0.941);
        test(0.85, 004.0, 1.190);
        test(0.9, 004.0, 1.533);
        test(0.95, 004.0, 2.132);
        test(0.975, 004.0, 2.776);
        test(0.99, 004.0, 3.747);
        test(0.995, 004.0, 4.604);
        test(0.9975, 004.0, 5.598);
        test(0.999, 004.0, 7.173);
        test(0.9995, 004.0, 8.610);

        test(0.75, 005.0, 0.727);
        test(0.8, 005.0, 0.920);
        test(0.85, 005.0, 1.156);
        test(0.9, 005.0, 1.476);
        test(0.95, 005.0, 2.015);
        test(0.975, 005.0, 2.571);
        test(0.99, 005.0, 3.365);
        test(0.995, 005.0, 4.032);
        test(0.9975, 005.0, 4.773);
        test(0.999, 005.0, 5.893);
        test(0.9995, 005.0, 6.869);

        test(0.75, 006.0, 0.718);
        test(0.8, 006.0, 0.906);
        test(0.85, 006.0, 1.134);
        test(0.9, 006.0, 1.440);
        test(0.95, 006.0, 1.943);
        test(0.975, 006.0, 2.447);
        test(0.99, 006.0, 3.143);
        test(0.995, 006.0, 3.707);
        test(0.9975, 006.0, 4.317);
        test(0.999, 006.0, 5.208);
        test(0.9995, 006.0, 5.959);

        test(0.75, 007.0, 0.711);
        test(0.8, 007.0, 0.896);
        test(0.85, 007.0, 1.119);
        test(0.9, 007.0, 1.415);
        test(0.95, 007.0, 1.895);
        test(0.975, 007.0, 2.365);
        test(0.99, 007.0, 2.998);
        test(0.995, 007.0, 3.499);
        test(0.9975, 007.0, 4.029);
        test(0.999, 007.0, 4.785);
        test(0.9995, 007.0, 5.408);

        test(0.75, 008.0, 0.706);
        test(0.8, 008.0, 0.889);
        test(0.85, 008.0, 1.108);
        test(0.9, 008.0, 1.397);
        test(0.95, 008.0, 1.860);
        test(0.975, 008.0, 2.306);
        test(0.99, 008.0, 2.896);
        test(0.995, 008.0, 3.355);
        test(0.9975, 008.0, 3.833);
        test(0.999, 008.0, 4.501);
        test(0.9995, 008.0, 5.041);

        test(0.75, 009.0, 0.703);
        test(0.8, 009.0, 0.883);
        test(0.85, 009.0, 1.100);
        test(0.9, 009.0, 1.383);
        test(0.95, 009.0, 1.833);
        test(0.975, 009.0, 2.262);
        test(0.99, 009.0, 2.821);
        test(0.995, 009.0, 3.250);
        test(0.9975, 009.0, 3.690);
        test(0.999, 009.0, 4.297);
        test(0.9995, 009.0, 4.781);

        test(0.75, 010.0, 0.700);
        test(0.8, 010.0, 0.879);
        test(0.85, 010.0, 1.093);
        test(0.9, 010.0, 1.372);
        test(0.95, 010.0, 1.812);
        test(0.975, 010.0, 2.228);
        test(0.99, 010.0, 2.764);
        test(0.995, 010.0, 3.169);
        test(0.9975, 010.0, 3.581);
        test(0.999, 010.0, 4.144);
        test(0.9995, 010.0, 4.587);

        test(0.75, 011.0, 0.697);
        test(0.8, 011.0, 0.876);
        test(0.85, 011.0, 1.088);
        test(0.9, 011.0, 1.363);
        test(0.95, 011.0, 1.796);
        test(0.975, 011.0, 2.201);
        test(0.99, 011.0, 2.718);
        test(0.995, 011.0, 3.106);
        test(0.9975, 011.0, 3.497);
        test(0.999, 011.0, 4.025);
        test(0.9995, 011.0, 4.437);

        test(0.75, 012.0, 0.695);
        test(0.8, 012.0, 0.873);
        test(0.85, 012.0, 1.083);
        test(0.9, 012.0, 1.356);
        test(0.95, 012.0, 1.782);
        test(0.975, 012.0, 2.179);
        test(0.99, 012.0, 2.681);
        test(0.995, 012.0, 3.055);
        test(0.9975, 012.0, 3.428);
        test(0.999, 012.0, 3.930);
        test(0.9995, 012.0, 4.318);

        test(0.75, 013.0, 0.694);
        test(0.8, 013.0, 0.870);
        test(0.85, 013.0, 1.079);
        test(0.9, 013.0, 1.350);
        test(0.95, 013.0, 1.771);
        test(0.975, 013.0, 2.160);
        test(0.99, 013.0, 2.650);
        test(0.995, 013.0, 3.012);
        test(0.9975, 013.0, 3.372);
        test(0.999, 013.0, 3.852);
        test(0.9995, 013.0, 4.221);

        test(0.75, 014.0, 0.692);
        test(0.8, 014.0, 0.868);
        test(0.85, 014.0, 1.076);
        test(0.9, 014.0, 1.345);
        test(0.95, 014.0, 1.761);
        test(0.975, 014.0, 2.145);
        test(0.99, 014.0, 2.624);
        test(0.995, 014.0, 2.977);
        test(0.9975, 014.0, 3.326);
        test(0.999, 014.0, 3.787);
        test(0.9995, 014.0, 4.140);

        test(0.75, 015.0, 0.691);
        test(0.8, 015.0, 0.866);
        test(0.85, 015.0, 1.074);
        test(0.9, 015.0, 1.341);
        test(0.95, 015.0, 1.753);
        test(0.975, 015.0, 2.131);
        test(0.99, 015.0, 2.602);
        test(0.995, 015.0, 2.947);
        test(0.9975, 015.0, 3.286);
        test(0.999, 015.0, 3.733);
        test(0.9995, 015.0, 4.073);

        test(0.75, 016.0, 0.690);
        test(0.8, 016.0, 0.865);
        test(0.85, 016.0, 1.071);
        test(0.9, 016.0, 1.337);
        test(0.95, 016.0, 1.746);
        test(0.975, 016.0, 2.120);
        test(0.99, 016.0, 2.583);
        test(0.995, 016.0, 2.921);
        test(0.9975, 016.0, 3.252);
        test(0.999, 016.0, 3.686);
        test(0.9995, 016.0, 4.015);

        test(0.75, 017.0, 0.689);
        test(0.8, 017.0, 0.863);
        test(0.85, 017.0, 1.069);
        test(0.9, 017.0, 1.333);
        test(0.95, 017.0, 1.740);
        test(0.975, 017.0, 2.110);
        test(0.99, 017.0, 2.567);
        test(0.995, 017.0, 2.898);
        test(0.9975, 017.0, 3.222);
        test(0.999, 017.0, 3.646);
        test(0.9995, 017.0, 3.965);

        test(0.75, 018.0, 0.688);
        test(0.8, 018.0, 0.862);
        test(0.85, 018.0, 1.067);
        test(0.9, 018.0, 1.330);
        test(0.95, 018.0, 1.734);
        test(0.975, 018.0, 2.101);
        test(0.99, 018.0, 2.552);
        test(0.995, 018.0, 2.878);
        test(0.9975, 018.0, 3.197);
        test(0.999, 018.0, 3.610);
        test(0.9995, 018.0, 3.922);

        test(0.75, 019.0, 0.688);
        test(0.8, 019.0, 0.861);
        test(0.85, 019.0, 1.066);
        test(0.9, 019.0, 1.328);
        test(0.95, 019.0, 1.729);
        test(0.975, 019.0, 2.093);
        test(0.99, 019.0, 2.539);
        test(0.995, 019.0, 2.861);
        test(0.9975, 019.0, 3.174);
        test(0.999, 019.0, 3.579);
        test(0.9995, 019.0, 3.883);

        test(0.75, 020.0, 0.687);
        test(0.8, 020.0, 0.860);
        test(0.85, 020.0, 1.064);
        test(0.9, 020.0, 1.325);
        test(0.95, 020.0, 1.725);
        test(0.975, 020.0, 2.086);
        test(0.99, 020.0, 2.528);
        test(0.995, 020.0, 2.845);
        test(0.9975, 020.0, 3.153);
        test(0.999, 020.0, 3.552);
        test(0.9995, 020.0, 3.850);

        test(0.75, 021.0, 0.686);
        test(0.8, 021.0, 0.859);
        test(0.85, 021.0, 1.063);
        test(0.9, 021.0, 1.323);
        test(0.95, 021.0, 1.721);
        test(0.975, 021.0, 2.080);
        test(0.99, 021.0, 2.518);
        test(0.995, 021.0, 2.831);
        test(0.9975, 021.0, 3.135);
        test(0.999, 021.0, 3.527);
        test(0.9995, 021.0, 3.819);

        test(0.75, 022.0, 0.686);
        test(0.8, 022.0, 0.858);
        test(0.85, 022.0, 1.061);
        test(0.9, 022.0, 1.321);
        test(0.95, 022.0, 1.717);
        test(0.975, 022.0, 2.074);
        test(0.99, 022.0, 2.508);
        test(0.995, 022.0, 2.819);
        test(0.9975, 022.0, 3.119);
        test(0.999, 022.0, 3.505);
        test(0.9995, 022.0, 3.792);

        test(0.75, 023.0, 0.685);
        test(0.8, 023.0, 0.858);
        test(0.85, 023.0, 1.060);
        test(0.9, 023.0, 1.319);
        test(0.95, 023.0, 1.714);
        test(0.975, 023.0, 2.069);
        test(0.99, 023.0, 2.500);
        test(0.995, 023.0, 2.807);
        test(0.9975, 023.0, 3.104);
        test(0.999, 023.0, 3.485);
        test(0.9995, 023.0, 3.767);

        test(0.75, 024.0, 0.685);
        test(0.8, 024.0, 0.857);
        test(0.85, 024.0, 1.059);
        test(0.9, 024.0, 1.318);
        test(0.95, 024.0, 1.711);
        test(0.975, 024.0, 2.064);
        test(0.99, 024.0, 2.492);
        test(0.995, 024.0, 2.797);
        test(0.9975, 024.0, 3.091);
        test(0.999, 024.0, 3.467);
        test(0.9995, 024.0, 3.745);

        test(0.75, 025.0, 0.684);
        test(0.8, 025.0, 0.856);
        test(0.85, 025.0, 1.058);
        test(0.9, 025.0, 1.316);
        test(0.95, 025.0, 1.708);
        test(0.975, 025.0, 2.060);
        test(0.99, 025.0, 2.485);
        test(0.995, 025.0, 2.787);
        test(0.9975, 025.0, 3.078);
        test(0.999, 025.0, 3.450);
        test(0.9995, 025.0, 3.725);

        test(0.75, 026.0, 0.684);
        test(0.8, 026.0, 0.856);
        test(0.85, 026.0, 1.058);
        test(0.9, 026.0, 1.315);
        test(0.95, 026.0, 1.706);
        test(0.975, 026.0, 2.056);
        test(0.99, 026.0, 2.479);
        test(0.995, 026.0, 2.779);
        test(0.9975, 026.0, 3.067);
        test(0.999, 026.0, 3.435);
        test(0.9995, 026.0, 3.707);

        test(0.75, 027.0, 0.684);
        test(0.8, 027.0, 0.855);
        test(0.85, 027.0, 1.057);
        test(0.9, 027.0, 1.314);
        test(0.95, 027.0, 1.703);
        test(0.975, 027.0, 2.052);
        test(0.99, 027.0, 2.473);
        test(0.995, 027.0, 2.771);
        test(0.9975, 027.0, 3.057);
        test(0.999, 027.0, 3.421);
        test(0.9995, 027.0, 3.690);

        test(0.75, 028.0, 0.683);
        test(0.8, 028.0, 0.855);
        test(0.85, 028.0, 1.056);
        test(0.9, 028.0, 1.313);
        test(0.95, 028.0, 1.701);
        test(0.975, 028.0, 2.048);
        test(0.99, 028.0, 2.467);
        test(0.995, 028.0, 2.763);
        test(0.9975, 028.0, 3.047);
        test(0.999, 028.0, 3.408);
        test(0.9995, 028.0, 3.674);

        test(0.75, 029.0, 0.683);
        test(0.8, 029.0, 0.854);
        test(0.85, 029.0, 1.055);
        test(0.9, 029.0, 1.311);
        test(0.95, 029.0, 1.699);
        test(0.975, 029.0, 2.045);
        test(0.99, 029.0, 2.462);
        test(0.995, 029.0, 2.756);
        test(0.9975, 029.0, 3.038);
        test(0.999, 029.0, 3.396);
        test(0.9995, 029.0, 3.659);

        test(0.75, 030.0, 0.683);
        test(0.8, 030.0, 0.854);
        test(0.85, 030.0, 1.055);
        test(0.9, 030.0, 1.310);
        test(0.95, 030.0, 1.697);
        test(0.975, 030.0, 2.042);
        test(0.99, 030.0, 2.457);
        test(0.995, 030.0, 2.750);
        test(0.9975, 030.0, 3.030);
        test(0.999, 030.0, 3.385);
        test(0.9995, 030.0, 3.646);

        test(0.75, 040.0, 0.681);
        test(0.8, 040.0, 0.851);
        test(0.85, 040.0, 1.050);
        test(0.9, 040.0, 1.303);
        test(0.95, 040.0, 1.684);
        test(0.975, 040.0, 2.021);
        test(0.99, 040.0, 2.423);
        test(0.995, 040.0, 2.704);
        test(0.9975, 040.0, 2.971);
        test(0.999, 040.0, 3.307);
        test(0.9995, 040.0, 3.551);

        test(0.75, 050.0, 0.679);
        test(0.8, 050.0, 0.849);
        test(0.85, 050.0, 1.047);
        test(0.9, 050.0, 1.299);
        test(0.95, 050.0, 1.676);
        test(0.975, 050.0, 2.009);
        test(0.99, 050.0, 2.403);
        test(0.995, 050.0, 2.678);
        test(0.9975, 050.0, 2.937);
        test(0.999, 050.0, 3.261);
        test(0.9995, 050.0, 3.496);

        test(0.75, 060.0, 0.679);
        test(0.8, 060.0, 0.848);
        test(0.85, 060.0, 1.045);
        test(0.9, 060.0, 1.296);
        test(0.95, 060.0, 1.671);
        test(0.975, 060.0, 2.000);
        test(0.99, 060.0, 2.390);
        test(0.995, 060.0, 2.660);
        test(0.9975, 060.0, 2.915);
        test(0.999, 060.0, 3.232);
        test(0.9995, 060.0, 3.460);

        test(0.75, 080.0, 0.678);
        test(0.8, 080.0, 0.846);
        test(0.85, 080.0, 1.043);
        test(0.9, 080.0, 1.292);
        test(0.95, 080.0, 1.664);
        test(0.975, 080.0, 1.990);
        test(0.99, 080.0, 2.374);
        test(0.995, 080.0, 2.639);
        test(0.9975, 080.0, 2.887);
        test(0.999, 080.0, 3.195);
        test(0.9995, 080.0, 3.416);

        test(0.75, 100.0, 0.677);
        test(0.8, 100.0, 0.845);
        test(0.85, 100.0, 1.042);
        test(0.9, 100.0, 1.290);
        test(0.95, 100.0, 1.660);
        test(0.975, 100.0, 1.984);
        test(0.99, 100.0, 2.364);
        test(0.995, 100.0, 2.626);
        test(0.9975, 100.0, 2.871);
        test(0.999, 100.0, 3.174);
        test(0.9995, 100.0, 3.390);

        test(0.75, 120.0, 0.677);
        test(0.8, 120.0, 0.845);
        test(0.85, 120.0, 1.041);
        test(0.9, 120.0, 1.289);
        test(0.95, 120.0, 1.658);
        test(0.975, 120.0, 1.980);
        test(0.99, 120.0, 2.358);
        test(0.995, 120.0, 2.617);
        test(0.9975, 120.0, 2.860);
        test(0.999, 120.0, 3.160);
        test(0.9995, 120.0, 3.373);
    }

    #[test]
    fn test_inv_cdf_high_precision() {
        let test = |x: f64, freedom: f64, expected: f64| {
            use approx::assert_relative_eq;
            let d = StudentsT::new(0., 1., freedom).unwrap();
            assert_relative_eq!(d.inverse_cdf(x), expected, max_relative = ACC);
        };
        // The data in this table of expected values was generated in
        // Python, using the mpsci package (based on mpmath):
        //
        //   import mpmath
        //   from mpsci.distributions import t
        //
        //   # Set the number of digits of precision
        //   mpmath.mp.dps = 200
        //
        //   ps = [0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
        //   dfs = [1.0, 10.0, 100.0]
        //
        //   for df in dfs:
        //       for p in ps:
        //           q = t.invcdf(p, df)
        //           print(f"({p:5.3f}, {df:5.1f}, {float(q)}),")
        //
        let invcdf_data = [
            // p       df    inverse_cdf(p, df)
            (0.001,   1.0, -318.30883898555044),
            (0.010,   1.0, -31.820515953773956),
            (0.100,   1.0, -3.077683537175253),
            (0.150,   1.0, -1.9626105055051506),
            (0.200,   1.0, -1.3763819204711734),
            (0.250,   1.0, -1.0),
            (0.300,   1.0, -0.7265425280053609),
            (0.350,   1.0, -0.5095254494944289),
            (0.400,   1.0, -0.32491969623290623),
            (0.450,   1.0, -0.15838444032453625),
            (0.001,  10.0, -4.143700494046589),
            (0.010,  10.0, -2.763769458112696),
            (0.100,  10.0, -1.3721836411103356),
            (0.150,  10.0, -1.093058073590526),
            (0.200,  10.0, -0.8790578285505887),
            (0.250,  10.0, -0.6998120613124317),
            (0.300,  10.0, -0.5415280387550157),
            (0.350,  10.0, -0.3965914937556218),
            (0.400,  10.0, -0.26018482949208016),
            (0.450,  10.0, -0.12889018929327375),
            (0.001, 100.0, -3.173739493738783),
            (0.010, 100.0, -2.364217366238482),
            (0.100, 100.0, -1.290074761346516),
            (0.150, 100.0, -1.041835900908347),
            (0.200, 100.0, -0.845230424491016),
            (0.250, 100.0, -0.6769510430114715),
            (0.300, 100.0, -0.5260762706003463),
            (0.350, 100.0, -0.3864289804076715),
            (0.400, 100.0, -0.2540221824582278),
            (0.450, 100.0, -0.12598088204153965),
        ];
        for (p, df, expected) in invcdf_data.iter() {
            test(*p, *df, *expected);
            test(1.0 - *p, *df, -*expected);
        }
    }

    #[test]
    fn test_inv_cdf_midpoint() {
        for loc in [0.0, 1.0, -3.5] {
            let d = StudentsT::new(loc, 1.0, 12.0).unwrap();
            // inverse_cdf(p) is a floating point calculation, so using
            // assert_eq here is optimistic.  For the given location values,
            // the check passes, so let's use the optimistic check for now.
            assert_eq!(d.inverse_cdf(0.5), loc);
        }
    }

    #[test]
    fn test_inv_cdf_p0() {
        let d = StudentsT::new(0.0, 1.0, 12.0).unwrap();
        assert_eq!(d.inverse_cdf(0.0), std::f64::NEG_INFINITY);
    }

    #[test]
    fn test_inv_cdf_p1() {
        let d = StudentsT::new(0.0, 1.0, 12.0).unwrap();
        assert_eq!(d.inverse_cdf(1.0), std::f64::INFINITY);
    }
}
