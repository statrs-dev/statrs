use crate::distribution::{Continuous, ContinuousCDF};
use crate::function::gamma;
use crate::is_zero;
use crate::statistics::*;
use crate::{consts, Result, StatsError};
use rand::Rng;
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
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Weibull {
    shape: f64,
    scale: f64,
    scale_pow_shape_inv: f64,
}

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
    pub fn new(shape: f64, scale: f64) -> Result<Weibull> {
        let is_nan = shape.is_nan() || scale.is_nan();
        match (shape, scale, is_nan) {
            (_, _, true) => Err(StatsError::BadParams),
            (_, _, false) if shape <= 0.0 || scale <= 0.0 => Err(StatsError::BadParams),
            (_, _, false) => Ok(Weibull {
                shape,
                scale,
                scale_pow_shape_inv: scale.powf(-shape),
            }),
        }
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

impl ::rand::distributions::Distribution<f64> for Weibull {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
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
    /// ```ignore
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
    /// ```ignore
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
}

impl Min<f64> for Weibull {
    /// Returns the minimum value in the domain of the weibull
    /// distribution representable by a double precision float
    ///
    /// # Formula
    ///
    /// ```ignore
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
    /// ```ignore
    /// INF
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
    /// ```ignore
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
    /// ```ignore
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
    /// ```ignore
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
    /// ```ignore
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
    /// ```ignore
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
    /// ```ignore
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
    /// ```ignore
    /// (k / λ) * (x / λ)^(k - 1) * e^(-(x / λ)^k)
    /// ```
    ///
    /// where `k` is the shape and `λ` is the scale
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else if is_zero(x) && ulps_eq!(self.shape, 1.0) {
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
    /// ```ignore
    /// ln((k / λ) * (x / λ)^(k - 1) * e^(-(x / λ)^k))
    /// ```
    ///
    /// where `k` is the shape and `λ` is the scale
    fn ln_pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            f64::NEG_INFINITY
        } else if is_zero(x) && ulps_eq!(self.shape, 1.0) {
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
#[cfg(all(test, feature = "nightly"))]
mod tests {
    use crate::statistics::*;
    use crate::distribution::{ContinuousCDF, Continuous, Weibull};
    use crate::distribution::internal::*;
    use crate::consts::ACC;

    fn try_create(shape: f64, scale: f64) -> Weibull {
        let n = Weibull::new(shape, scale);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(shape: f64, scale: f64) {
        let n = try_create(shape, scale);
        assert_eq!(shape, n.shape());
        assert_eq!(scale, n.scale());
    }

    fn bad_create_case(shape: f64, scale: f64) {
        let n = Weibull::new(shape, scale);
        assert!(n.is_err());
    }

    fn get_value<F>(shape: f64, scale: f64, eval: F) -> f64
        where F: Fn(Weibull) -> f64
    {
        let n = try_create(shape, scale);
        eval(n)
    }

    fn test_case<F>(shape: f64, scale: f64, expected: f64, eval: F)
        where F: Fn(Weibull) -> f64
    {
        let x = get_value(shape, scale, eval);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(shape: f64, scale: f64, expected: f64, acc: f64, eval: F)
        where F: Fn(Weibull) -> f64
    {
        let x = get_value(shape, scale, eval);
        assert_almost_eq!(expected, x, acc);
    }

    #[test]
    fn test_create() {
        create_case(1.0, 0.1);
        create_case(10.0, 1.0);
        create_case(11.0, 10.0);
        create_case(12.0, f64::INFINITY);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(f64::NAN, 1.0);
        bad_create_case(1.0, f64::NAN);
        bad_create_case(f64::NAN, f64::NAN);
        bad_create_case(1.0, -1.0);
        bad_create_case(-1.0, 1.0);
        bad_create_case(-1.0, -1.0);
        bad_create_case(0.0, 0.0);
        bad_create_case(0.0, 1.0);
        bad_create_case(1.0, 0.0);
    }

    #[test]
    fn test_mean() {
        let mean = |x: Weibull| x.mean().unwrap();
        test_case(1.0, 0.1, 0.1, mean);
        test_case(1.0, 1.0, 1.0, mean);
        test_almost(10.0, 10.0, 9.5135076986687318362924871772654021925505786260884, 1e-14, mean);
        test_almost(10.0, 1.0, 0.95135076986687318362924871772654021925505786260884, 1e-15, mean);
    }

    #[test]
    fn test_variance() {
        let variance = |x: Weibull| x.variance().unwrap();
        test_almost(1.0, 0.1, 0.01, 1e-16, variance);
        test_almost(1.0, 1.0, 1.0, 1e-14, variance);
        test_almost(10.0, 10.0, 1.3100455073468309147154581687505295026863354547057, 1e-12, variance);
        test_almost(10.0, 1.0, 0.013100455073468309147154581687505295026863354547057, 1e-14, variance);
    }

    #[test]
    fn test_entropy() {
        let entropy = |x: Weibull| x.entropy().unwrap();
        test_almost(1.0, 0.1, -1.302585092994045684018, 1e-15, entropy);
        test_case(1.0, 1.0, 1.0, entropy);
        test_case(10.0, 10.0, 1.519494098411379574546, entropy);
        test_almost(10.0, 1.0, -0.783090994582666109472, 1e-15, entropy);
    }

    #[test]
    fn test_skewnewss() {
        let skewness = |x: Weibull| x.skewness().unwrap();
        test_almost(1.0, 0.1, 2.0, 1e-13, skewness);
        test_almost(1.0, 1.0, 2.0, 1e-13, skewness);
        test_almost(10.0, 10.0, -0.63763713390314440916597757156663888653981696212127, 1e-11, skewness);
        test_almost(10.0, 1.0, -0.63763713390314440916597757156663888653981696212127, 1e-11, skewness);
    }

    #[test]
    fn test_median() {
        let median = |x: Weibull| x.median();
        test_case(1.0, 0.1, 0.069314718055994530941723212145817656807550013436026, median);
        test_case(1.0, 1.0, 0.69314718055994530941723212145817656807550013436026, median);
        test_case(10.0, 10.0, 9.6401223546778973665856033763604752124634905617583, median);
        test_case(10.0, 1.0, 0.96401223546778973665856033763604752124634905617583, median);
    }

    #[test]
    fn test_mode() {
        let mode = |x: Weibull| x.mode().unwrap();
        test_case(1.0, 0.1, 0.0, mode);
        test_case(1.0, 1.0, 0.0, mode);
        test_case(10.0, 10.0, 9.8951925820621439264623017041980483215553841533709, mode);
        test_case(10.0, 1.0, 0.98951925820621439264623017041980483215553841533709, mode);
    }

    #[test]
    fn test_min_max() {
        let min = |x: Weibull| x.min();
        let max = |x: Weibull| x.max();
        test_case(1.0, 1.0, 0.0, min);
        test_case(1.0, 1.0, f64::INFINITY, max);
    }

    #[test]
    fn test_pdf() {
        let pdf = |arg: f64| move |x: Weibull| x.pdf(arg);
        test_case(1.0, 0.1, 10.0, pdf(0.0));
        test_case(1.0, 0.1, 0.00045399929762484851535591515560550610237918088866565, pdf(1.0));
        test_case(1.0, 0.1, 3.7200759760208359629596958038631183373588922923768e-43, pdf(10.0));
        test_case(1.0, 1.0, 1.0, pdf(0.0));
        test_case(1.0, 1.0, 0.36787944117144232159552377016146086744581113103177, pdf(1.0));
        test_case(1.0, 1.0, 0.000045399929762484851535591515560550610237918088866565, pdf(10.0));
        test_case(10.0, 10.0, 0.0, pdf(0.0));
        test_almost(10.0, 10.0, 9.9999999990000000000499999999983333333333750000000e-10, 1e-24, pdf(1.0));
        test_case(10.0, 10.0, 0.36787944117144232159552377016146086744581113103177, pdf(10.0));
        test_case(10.0, 1.0, 0.0, pdf(0.0));
        test_case(10.0, 1.0, 3.6787944117144232159552377016146086744581113103177, pdf(1.0));
        test_case(10.0, 1.0, 0.0, pdf(10.0));
    }

    #[test]
    fn test_ln_pdf() {
        let ln_pdf = |arg: f64| move |x: Weibull| x.ln_pdf(arg);
        test_almost(1.0, 0.1, 2.3025850929940456840179914546843642076011014886288, 1e-15, ln_pdf(0.0));
        test_almost(1.0, 0.1, -7.6974149070059543159820085453156357923988985113712, 1e-15, ln_pdf(1.0));
        test_case(1.0, 0.1, -97.697414907005954315982008545315635792398898511371, ln_pdf(10.0));
        test_case(1.0, 1.0, 0.0, ln_pdf(0.0));
        test_case(1.0, 1.0, -1.0, ln_pdf(1.0));
        test_case(1.0, 1.0, -10.0, ln_pdf(10.0));
        test_case(10.0, 10.0, f64::NEG_INFINITY, ln_pdf(0.0));
        test_almost(10.0, 10.0, -20.723265837046411156161923092159277868409913397659, 1e-14, ln_pdf(1.0));
        test_case(10.0, 10.0, -1.0, ln_pdf(10.0));
        test_case(10.0, 1.0, f64::NEG_INFINITY, ln_pdf(0.0));
        test_almost(10.0, 1.0, 1.3025850929940456840179914546843642076011014886288, 1e-15, ln_pdf(1.0));
        test_case(10.0, 1.0, -9.999999976974149070059543159820085453156357923988985113712e9, ln_pdf(10.0));
    }

    #[test]
    fn test_cdf() {
        let cdf = |arg: f64| move |x: Weibull| x.cdf(arg);
        test_case(1.0, 0.1, 0.0, cdf(0.0));
        test_case(1.0, 0.1, 0.99995460007023751514846440848443944938976208191113, cdf(1.0));
        test_case(1.0, 0.1, 0.99999999999999999999999999999999999999999996279924, cdf(10.0));
        test_case(1.0, 1.0, 0.0, cdf(0.0));
        test_case(1.0, 1.0, 0.63212055882855767840447622983853913255418886896823, cdf(1.0));
        test_case(1.0, 1.0, 0.99995460007023751514846440848443944938976208191113, cdf(10.0));
        test_case(10.0, 10.0, 0.0, cdf(0.0));
        test_almost(10.0, 10.0, 9.9999999995000000000166666666662500000000083333333e-11, 1e-25, cdf(1.0));
        test_case(10.0, 10.0, 0.63212055882855767840447622983853913255418886896823, cdf(10.0));
        test_case(10.0, 1.0, 0.0, cdf(0.0));
        test_case(10.0, 1.0, 0.63212055882855767840447622983853913255418886896823, cdf(1.0));
        test_case(10.0, 1.0, 1.0, cdf(10.0));
    }

    #[test]
    fn test_sf() {
        let sf = |arg: f64| move |x: Weibull| x.sf(arg);
        test_case(1.0, 0.1, 1.0, sf(0.0));
        test_case(1.0, 0.1, 4.5399929762484854e-5, sf(1.0));
        test_case(1.0, 0.1, 3.720075976020836e-44, sf(10.0));
        test_case(1.0, 1.0, 1.0, sf(0.0));
        test_case(1.0, 1.0, 0.36787944117144233, sf(1.0));
        test_case(1.0, 1.0, 4.5399929762484854e-5, sf(10.0));
        test_case(10.0, 10.0, 1.0, sf(0.0));
        test_almost(10.0, 10.0, 0.9999999999, 1e-25, sf(1.0));
        test_case(10.0, 10.0, 0.36787944117144233, sf(10.0));
        test_case(10.0, 1.0, 1.0, sf(0.0));
        test_case(10.0, 1.0, 0.36787944117144233, sf(1.0));
        test_case(10.0, 1.0, 0.0, sf(10.0));
    }

    #[test]
    fn test_continuous() {
        test::check_continuous_distribution(&try_create(1.0, 0.2), 0.0, 10.0);
    }
}
