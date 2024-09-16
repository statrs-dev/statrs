use crate::consts;
use crate::distribution::{Continuous, ContinuousCDF};
use crate::function::gamma;
use crate::statistics::*;
use std::f64;

/// Implements the [Weibull](https://en.wikipedia.org/wiki/Weibull_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Weibull, Continuous};
/// use statrs::statistics::Distribution;
/// use statrs::prec;
///
/// let n = Weibull::new(10.0, 1.0).unwrap();
/// assert!(prec::almost_eq(n.mean().unwrap(),
/// 0.95135076986687318362924871772654021925505786260884, 1e-15));
/// assert_eq!(n.pdf(1.0), 3.6787944117144232159552377016146086744581113103177);
/// ```
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Weibull {
    shape: f64,
    scale: f64,
    scale_pow_shape_inv: f64,
}

/// Represents the errors that can occur when creating a [`Weibull`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum WeibullError {
    /// The shape is NaN, zero or less than zero.
    ShapeInvalid,

    /// The scale is NaN, zero or less than zero.
    ScaleInvalid,
}

impl std::fmt::Display for WeibullError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            WeibullError::ShapeInvalid => write!(f, "Shape is NaN, zero or less than zero."),
            WeibullError::ScaleInvalid => write!(f, "Scale is NaN, zero or less than zero."),
        }
    }
}

impl std::error::Error for WeibullError {}

impl Weibull {
    /// Constructs a new weibull distribution with a shape (k) of `shape`
    /// and a scale (λ) of `scale`
    ///
    /// # Errors
    ///
    /// Returns an error if `shape` or `scale` are `NaN`.
    /// Returns an error if `shape <= 0.0` or `scale <= 0.0`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Weibull;
    ///
    /// let mut result = Weibull::new(10.0, 1.0);
    /// assert!(result.is_ok());
    ///
    /// result = Weibull::new(0.0, 0.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(shape: f64, scale: f64) -> Result<Weibull, WeibullError> {
        if shape.is_nan() || shape <= 0.0 {
            return Err(WeibullError::ShapeInvalid);
        }

        if scale.is_nan() || scale <= 0.0 {
            return Err(WeibullError::ScaleInvalid);
        }

        Ok(Weibull {
            shape,
            scale,
            scale_pow_shape_inv: scale.powf(-shape),
        })
    }

    /// Returns the shape of the weibull distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Weibull;
    ///
    /// let n = Weibull::new(10.0, 1.0).unwrap();
    /// assert_eq!(n.shape(), 10.0);
    /// ```
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// Returns the scale of the weibull distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Weibull;
    ///
    /// let n = Weibull::new(10.0, 1.0).unwrap();
    /// assert_eq!(n.scale(), 1.0);
    /// ```
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

impl std::fmt::Display for Weibull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Weibull({},{})", self.scale, self.shape)
    }
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl ::rand::distributions::Distribution<f64> for Weibull {
    fn sample<R: ::rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let x: f64 = rng.gen();
        self.scale * (-x.ln()).powf(1.0 / self.shape)
    }
}

impl ContinuousCDF<f64, f64> for Weibull {
    /// Calculates the cumulative distribution function for the weibull
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// 1 - e^-((x/λ)^k)
    /// ```
    ///
    /// where `k` is the shape and `λ` is the scale
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            -(-x.powf(self.shape) * self.scale_pow_shape_inv).exp_m1()
        }
    }

    /// Calculates the survival function for the weibull
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// e^-((x/λ)^k)
    /// ```
    ///
    /// where `k` is the shape and `λ` is the scale
    fn sf(&self, x: f64) -> f64 {
        if x < 0.0 {
            1.0
        } else {
            (-x.powf(self.shape) * self.scale_pow_shape_inv).exp()
        }
    }

    /// Calculates the inverse cumulative distribution function for the weibull
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// λ (-ln(1 - x))^(1 / k)
    /// ```
    ///
    /// where `k` is the shape and `λ` is the scale
    fn inverse_cdf(&self, p: f64) -> f64 {
        if !(0.0..=1.0).contains(&p) {
            panic!("x must be in [0, 1]");
        }

        (-((-p).ln_1p() / self.scale_pow_shape_inv)).powf(1.0 / self.shape)
    }
}

impl Min<f64> for Weibull {
    /// Returns the minimum value in the domain of the weibull
    /// distribution representable by a double precision float
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

impl Max<f64> for Weibull {
    /// Returns the maximum value in the domain of the weibull
    /// distribution representable by a double precision float
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

impl Distribution<f64> for Weibull {
    /// Returns the mean of the weibull distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// λΓ(1 + 1 / k)
    /// ```
    ///
    /// where `k` is the shape, `λ` is the scale, and `Γ` is
    /// the gamma function
    fn mean(&self) -> Option<f64> {
        Some(self.scale * gamma::gamma(1.0 + 1.0 / self.shape))
    }

    /// Returns the variance of the weibull distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// λ^2 * (Γ(1 + 2 / k) - Γ(1 + 1 / k)^2)
    /// ```
    ///
    /// where `k` is the shape, `λ` is the scale, and `Γ` is
    /// the gamma function
    fn variance(&self) -> Option<f64> {
        let mean = self.mean()?;
        Some(self.scale * self.scale * gamma::gamma(1.0 + 2.0 / self.shape) - mean * mean)
    }

    /// Returns the entropy of the weibull distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// γ(1 - 1 / k) + ln(λ / k) + 1
    /// ```
    ///
    /// where `k` is the shape, `λ` is the scale, and `γ` is
    /// the Euler-Mascheroni constant
    fn entropy(&self) -> Option<f64> {
        let entr = consts::EULER_MASCHERONI * (1.0 - 1.0 / self.shape)
            + (self.scale / self.shape).ln()
            + 1.0;
        Some(entr)
    }

    /// Returns the skewness of the weibull distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// (Γ(1 + 3 / k) * λ^3 - 3μσ^2 - μ^3) / σ^3
    /// ```
    ///
    /// where `k` is the shape, `λ` is the scale, and `Γ` is
    /// the gamma function, `μ` is the mean of the distribution.
    /// and `σ` the standard deviation of the distribution
    fn skewness(&self) -> Option<f64> {
        let mu = self.mean()?;
        let sigma = self.std_dev()?;
        let sigma2 = sigma * sigma;
        let sigma3 = sigma2 * sigma;
        let skew = (self.scale * self.scale * self.scale * gamma::gamma(1.0 + 3.0 / self.shape)
            - 3.0 * sigma2 * mu
            - (mu * mu * mu))
            / sigma3;
        Some(skew)
    }
}

impl Median<f64> for Weibull {
    /// Returns the median of the weibull distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// λ(ln(2))^(1 / k)
    /// ```
    ///
    /// where `k` is the shape and `λ` is the scale
    fn median(&self) -> f64 {
        self.scale * f64::consts::LN_2.powf(1.0 / self.shape)
    }
}

impl Mode<Option<f64>> for Weibull {
    /// Returns the median of the weibull distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// if k == 1 {
    ///     0
    /// } else {
    ///     λ((k - 1) / k)^(1 / k)
    /// }
    /// ```
    ///
    /// where `k` is the shape and `λ` is the scale
    fn mode(&self) -> Option<f64> {
        let mode = if ulps_eq!(self.shape, 1.0) {
            0.0
        } else {
            self.scale * ((self.shape - 1.0) / self.shape).powf(1.0 / self.shape)
        };
        Some(mode)
    }
}

impl Continuous<f64, f64> for Weibull {
    /// Calculates the probability density function for the weibull
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// (k / λ) * (x / λ)^(k - 1) * e^(-(x / λ)^k)
    /// ```
    ///
    /// where `k` is the shape and `λ` is the scale
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else if x == 0.0 && ulps_eq!(self.shape, 1.0) {
            1.0 / self.scale
        } else if x.is_infinite() {
            0.0
        } else {
            self.shape
                * (x / self.scale).powf(self.shape - 1.0)
                * (-(x.powf(self.shape)) * self.scale_pow_shape_inv).exp()
                / self.scale
        }
    }

    /// Calculates the log probability density function for the weibull
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// ln((k / λ) * (x / λ)^(k - 1) * e^(-(x / λ)^k))
    /// ```
    ///
    /// where `k` is the shape and `λ` is the scale
    fn ln_pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            f64::NEG_INFINITY
        } else if x == 0.0 && ulps_eq!(self.shape, 1.0) {
            0.0 - self.scale.ln()
        } else if x.is_infinite() {
            f64::NEG_INFINITY
        } else {
            self.shape.ln() + (self.shape - 1.0) * (x / self.scale).ln()
                - x.powf(self.shape) * self.scale_pow_shape_inv
                - self.scale.ln()
        }
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::internal::*;
    use crate::testing_boiler;

    testing_boiler!(shape: f64, scale: f64; Weibull; WeibullError);

    #[test]
    fn test_create() {
        create_ok(1.0, 0.1);
        create_ok(10.0, 1.0);
        create_ok(11.0, 10.0);
        create_ok(12.0, f64::INFINITY);
    }

    #[test]
    fn test_bad_create() {
        test_create_err(f64::NAN, 1.0, WeibullError::ShapeInvalid);
        test_create_err(1.0, f64::NAN, WeibullError::ScaleInvalid);
        create_err(f64::NAN, f64::NAN);
        create_err(1.0, -1.0);
        create_err(-1.0, 1.0);
        create_err(-1.0, -1.0);
        create_err(0.0, 0.0);
        create_err(0.0, 1.0);
        create_err(1.0, 0.0);
    }

    #[test]
    fn test_mean() {
        let mean = |x: Weibull| x.mean().unwrap();
        test_exact(1.0, 0.1, 0.1, mean);
        test_exact(1.0, 1.0, 1.0, mean);
        test_absolute(10.0, 10.0, 9.5135076986687318362924871772654021925505786260884, 1e-14, mean);
        test_absolute(10.0, 1.0, 0.95135076986687318362924871772654021925505786260884, 1e-15, mean);
    }

    #[test]
    fn test_variance() {
        let variance = |x: Weibull| x.variance().unwrap();
        test_absolute(1.0, 0.1, 0.01, 1e-16, variance);
        test_absolute(1.0, 1.0, 1.0, 1e-14, variance);
        test_absolute(10.0, 10.0, 1.3100455073468309147154581687505295026863354547057, 1e-12, variance);
        test_absolute(10.0, 1.0, 0.013100455073468309147154581687505295026863354547057, 1e-14, variance);
    }

    #[test]
    fn test_entropy() {
        let entropy = |x: Weibull| x.entropy().unwrap();
        test_absolute(1.0, 0.1, -1.302585092994045684018, 1e-15, entropy);
        test_exact(1.0, 1.0, 1.0, entropy);
        test_exact(10.0, 10.0, 1.519494098411379574546, entropy);
        test_absolute(10.0, 1.0, -0.783090994582666109472, 1e-15, entropy);
    }

    #[test]
    fn test_skewnewss() {
        let skewness = |x: Weibull| x.skewness().unwrap();
        test_absolute(1.0, 0.1, 2.0, 1e-13, skewness);
        test_absolute(1.0, 1.0, 2.0, 1e-13, skewness);
        test_absolute(10.0, 10.0, -0.63763713390314440916597757156663888653981696212127, 1e-11, skewness);
        test_absolute(10.0, 1.0, -0.63763713390314440916597757156663888653981696212127, 1e-11, skewness);
    }

    #[test]
    fn test_median() {
        let median = |x: Weibull| x.median();
        test_exact(1.0, 0.1, 0.069314718055994530941723212145817656807550013436026, median);
        test_exact(1.0, 1.0, 0.69314718055994530941723212145817656807550013436026, median);
        test_exact(10.0, 10.0, 9.6401223546778973665856033763604752124634905617583, median);
        test_exact(10.0, 1.0, 0.96401223546778973665856033763604752124634905617583, median);
    }

    #[test]
    fn test_mode() {
        let mode = |x: Weibull| x.mode().unwrap();
        test_exact(1.0, 0.1, 0.0, mode);
        test_exact(1.0, 1.0, 0.0, mode);
        test_exact(10.0, 10.0, 9.8951925820621439264623017041980483215553841533709, mode);
        test_exact(10.0, 1.0, 0.98951925820621439264623017041980483215553841533709, mode);
    }

    #[test]
    fn test_min_max() {
        let min = |x: Weibull| x.min();
        let max = |x: Weibull| x.max();
        test_exact(1.0, 1.0, 0.0, min);
        test_exact(1.0, 1.0, f64::INFINITY, max);
    }

    #[test]
    fn test_pdf() {
        let pdf = |arg: f64| move |x: Weibull| x.pdf(arg);
        test_exact(1.0, 0.1, 10.0, pdf(0.0));
        test_exact(1.0, 0.1, 0.00045399929762484851535591515560550610237918088866565, pdf(1.0));
        test_exact(1.0, 0.1, 3.7200759760208359629596958038631183373588922923768e-43, pdf(10.0));
        test_exact(1.0, 1.0, 1.0, pdf(0.0));
        test_exact(1.0, 1.0, 0.36787944117144232159552377016146086744581113103177, pdf(1.0));
        test_exact(1.0, 1.0, 0.000045399929762484851535591515560550610237918088866565, pdf(10.0));
        test_exact(10.0, 10.0, 0.0, pdf(0.0));
        test_absolute(10.0, 10.0, 9.9999999990000000000499999999983333333333750000000e-10, 1e-24, pdf(1.0));
        test_exact(10.0, 10.0, 0.36787944117144232159552377016146086744581113103177, pdf(10.0));
        test_exact(10.0, 1.0, 0.0, pdf(0.0));
        test_exact(10.0, 1.0, 3.6787944117144232159552377016146086744581113103177, pdf(1.0));
        test_exact(10.0, 1.0, 0.0, pdf(10.0));
    }

    #[test]
    fn test_ln_pdf() {
        let ln_pdf = |arg: f64| move |x: Weibull| x.ln_pdf(arg);
        test_absolute(1.0, 0.1, 2.3025850929940456840179914546843642076011014886288, 1e-15, ln_pdf(0.0));
        test_absolute(1.0, 0.1, -7.6974149070059543159820085453156357923988985113712, 1e-15, ln_pdf(1.0));
        test_exact(1.0, 0.1, -97.697414907005954315982008545315635792398898511371, ln_pdf(10.0));
        test_exact(1.0, 1.0, 0.0, ln_pdf(0.0));
        test_exact(1.0, 1.0, -1.0, ln_pdf(1.0));
        test_exact(1.0, 1.0, -10.0, ln_pdf(10.0));
        test_exact(10.0, 10.0, f64::NEG_INFINITY, ln_pdf(0.0));
        test_absolute(10.0, 10.0, -20.723265837046411156161923092159277868409913397659, 1e-14, ln_pdf(1.0));
        test_exact(10.0, 10.0, -1.0, ln_pdf(10.0));
        test_exact(10.0, 1.0, f64::NEG_INFINITY, ln_pdf(0.0));
        test_absolute(10.0, 1.0, 1.3025850929940456840179914546843642076011014886288, 1e-15, ln_pdf(1.0));
        test_exact(10.0, 1.0, -9.999999976974149070059543159820085453156357923988985113712e9, ln_pdf(10.0));
    }

    #[test]
    fn test_cdf() {
        let cdf = |arg: f64| move |x: Weibull| x.cdf(arg);
        test_exact(1.0, 0.1, 0.0, cdf(0.0));
        test_exact(1.0, 0.1, 0.99995460007023751514846440848443944938976208191113, cdf(1.0));
        test_exact(1.0, 0.1, 0.99999999999999999999999999999999999999999996279924, cdf(10.0));
        test_exact(1.0, 1.0, 0.0, cdf(0.0));
        test_exact(1.0, 1.0, 0.63212055882855767840447622983853913255418886896823, cdf(1.0));
        test_exact(1.0, 1.0, 0.99995460007023751514846440848443944938976208191113, cdf(10.0));
        test_exact(10.0, 10.0, 0.0, cdf(0.0));
        test_absolute(10.0, 10.0, 9.9999999995000000000166666666662500000000083333333e-11, 1e-25, cdf(1.0));
        test_exact(10.0, 10.0, 0.63212055882855767840447622983853913255418886896823, cdf(10.0));
        test_exact(10.0, 1.0, 0.0, cdf(0.0));
        test_exact(10.0, 1.0, 0.63212055882855767840447622983853913255418886896823, cdf(1.0));
        test_exact(10.0, 1.0, 1.0, cdf(10.0));
    }

    #[test]
    fn test_sf() {
        let sf = |arg: f64| move |x: Weibull| x.sf(arg);
        test_exact(1.0, 0.1, 1.0, sf(0.0));
        test_exact(1.0, 0.1, 4.5399929762484854e-5, sf(1.0));
        test_exact(1.0, 0.1, 3.720075976020836e-44, sf(10.0));
        test_exact(1.0, 1.0, 1.0, sf(0.0));
        test_exact(1.0, 1.0, 0.36787944117144233, sf(1.0));
        test_exact(1.0, 1.0, 4.5399929762484854e-5, sf(10.0));
        test_exact(10.0, 10.0, 1.0, sf(0.0));
        test_absolute(10.0, 10.0, 0.9999999999, 1e-25, sf(1.0));
        test_exact(10.0, 10.0, 0.36787944117144233, sf(10.0));
        test_exact(10.0, 1.0, 1.0, sf(0.0));
        test_exact(10.0, 1.0, 0.36787944117144233, sf(1.0));
        test_exact(10.0, 1.0, 0.0, sf(10.0));
    }

    #[test]
    fn test_inverse_cdf() {
        let func = |arg: f64| move |x: Weibull| x.inverse_cdf(x.cdf(arg));
        test_exact(1.0, 0.1, 0.0, func(0.0));
        test_absolute(1.0, 0.1, 1.0, 1e-13, func(1.0));
        test_exact(1.0, 1.0, 0.0, func(0.0));
        test_exact(1.0, 1.0, 1.0, func(1.0));
        test_absolute(1.0, 1.0, 10.0, 1e-10, func(10.0));
        test_exact(10.0, 10.0, 0.0, func(0.0));
        test_absolute(10.0, 10.0, 1.0, 1e-5, func(1.0));
        test_absolute(10.0, 10.0, 10.0, 1e-10, func(10.0));
        test_exact(10.0, 1.0, 0.0, func(0.0));
        test_exact(10.0, 1.0, 1.0, func(1.0));
    }

    #[test]
    fn test_continuous() {
        test::check_continuous_distribution(&create_ok(1.0, 0.2), 0.0, 10.0);
    }
}
