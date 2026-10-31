use crate::distribution::{Continuous, ContinuousCDF};
use crate::statistics::*;
use core::f64::consts as f64_consts;
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

/// Implements the [Cauchy](https://en.wikipedia.org/wiki/Cauchy_distribution)
/// distribution, also known as the Lorentz distribution.
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Cauchy, Continuous};
/// use statrs::statistics::Mode;
///
/// let n = Cauchy::new(0.0, 1.0).unwrap();
/// assert_eq!(n.mode().unwrap(), 0.0);
/// assert_eq!(n.pdf(1.0), 0.1591549430918953357689);
/// ```
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Cauchy {
    location: f64,
    scale: f64,
}

/// Represents the errors that can occur when creating a [`Cauchy`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum CauchyError {
    /// The location is NaN.
    LocationInvalid,

    /// The scale is NaN, zero or less than zero.
    ScaleInvalid,
}

impl core::fmt::Display for CauchyError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match self {
            CauchyError::LocationInvalid => write!(f, "Location is NaN"),
            CauchyError::ScaleInvalid => write!(f, "Scale is NaN, zero or less than zero"),
        }
    }
}

impl core::error::Error for CauchyError {}

impl Cauchy {
    /// Constructs a new cauchy distribution with the given
    /// location and scale.
    ///
    /// # Errors
    ///
    /// Returns an error if location or scale are `NaN` or `scale <= 0.0`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Cauchy;
    ///
    /// let mut result = Cauchy::new(0.0, 1.0);
    /// assert!(result.is_ok());
    ///
    /// result = Cauchy::new(0.0, -1.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(location: f64, scale: f64) -> Result<Cauchy, CauchyError> {
        if location.is_nan() {
            return Err(CauchyError::LocationInvalid);
        }

        if scale.is_nan() || scale <= 0.0 {
            return Err(CauchyError::ScaleInvalid);
        }

        Ok(Cauchy { location, scale })
    }

    /// Returns the location of the cauchy distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Cauchy;
    ///
    /// let n = Cauchy::new(0.0, 1.0).unwrap();
    /// assert_eq!(n.location(), 0.0);
    /// ```
    pub fn location(&self) -> f64 {
        self.location
    }

    /// Returns the scale of the cauchy distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Cauchy;
    ///
    /// let n = Cauchy::new(0.0, 1.0).unwrap();
    /// assert_eq!(n.scale(), 1.0);
    /// ```
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

impl Cauchy {
    /// for `d = x - x_0`, returns `(γ / d, γ / d^2, ln |d|)` without letting
    /// `d` itself overflow when `x` and `x_0` are finite
    fn tail_parts(&self, x: f64) -> (f64, f64, f64) {
        let d = x - self.location;
        if d.is_finite() || !x.is_finite() {
            let r = self.scale / d;
            (r, r / d, d.abs().ln())
        } else {
            // |d| > f64::MAX, so work with h = d / 2
            let h = 0.5 * x - 0.5 * self.location;
            let r = 0.5 * self.scale / h;
            (r, 0.5 * (r / h), h.abs().ln() + f64_consts::LN_2)
        }
    }
}

impl core::fmt::Display for Cauchy {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Cauchy({}, {})", self.location, self.scale)
    }
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl ::rand::distr::Distribution<f64> for Cauchy {
    fn sample<R: ::rand::Rng + ?Sized>(&self, r: &mut R) -> f64 {
        let x = ::rand::RngExt::random::<f64>(r);
        self.location + self.scale * (f64_consts::PI * (x - 0.5)).tan()
    }
}

impl ContinuousCDF<f64, f64> for Cauchy {
    /// Calculates the cumulative distribution function for the
    /// cauchy distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// (1 / π) * arctan((x - x_0) / γ) + 0.5
    /// ```
    ///
    /// where `x_0` is the location and `γ` is the scale
    fn cdf(&self, x: f64) -> f64 {
        (1.0 / f64_consts::PI) * ((x - self.location) / self.scale).atan() + 0.5
    }

    /// Calculates the survival function for the
    /// cauchy distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// (1 / π) * arctan(-(x - x_0) / γ) + 0.5
    /// ```
    ///
    /// where `x_0` is the location and `γ` is the scale.
    /// note that this is identical to the cdf except for
    /// the negative argument to the arctan function
    fn sf(&self, x: f64) -> f64 {
        (1.0 / f64_consts::PI) * ((self.location - x) / self.scale).atan() + 0.5
    }

    /// Calculates the inverse cumulative distribution function for the
    /// cauchy distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// x_0 + γ tan((x - 0.5) π)
    /// ```
    ///
    /// where `x_0` is the location and `γ` is the scale
    fn inverse_cdf(&self, x: f64) -> f64 {
        if !(0.0..=1.0).contains(&x) {
            panic!("x must be in [0, 1]");
        } else {
            self.location + self.scale * (f64_consts::PI * (x - 0.5)).tan()
        }
    }
}

impl Min<f64> for Cauchy {
    /// Returns the minimum value in the domain of the cauchy
    /// distribution representable by a double precision float
    ///
    /// # Formula
    ///
    /// ```text
    /// NEG_INF
    /// ```
    fn min(&self) -> f64 {
        f64::NEG_INFINITY
    }
}

impl Max<f64> for Cauchy {
    /// Returns the maximum value in the domain of the cauchy
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

impl Distribution<f64> for Cauchy {
    /// Returns the entropy of the cauchy distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// ln(γ) + ln(4π)
    /// ```
    ///
    /// where `γ` is the scale
    fn entropy(&self) -> Option<f64> {
        Some((4.0 * f64_consts::PI * self.scale).ln())
    }
}

impl Median<f64> for Cauchy {
    /// Returns the median of the cauchy distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// x_0
    /// ```
    ///
    /// where `x_0` is the location
    fn median(&self) -> f64 {
        self.location
    }
}

impl Mode<Option<f64>> for Cauchy {
    /// Returns the mode of the cauchy distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// x_0
    /// ```
    ///
    /// where `x_0` is the location
    fn mode(&self) -> Option<f64> {
        Some(self.location)
    }
}

impl Continuous<f64, f64> for Cauchy {
    /// Calculates the probability density function for the cauchy
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// 1 / (πγ * (1 + ((x - x_0) / γ)^2))
    /// ```
    ///
    /// where `x_0` is the location and `γ` is the scale
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.location) / self.scale;
        let d = f64_consts::PI * self.scale * (1.0 + z * z);
        if d.is_finite() {
            return 1.0 / d;
        }
        // the denominator overflowed while the density itself may still be
        // representable (gh #472)
        let (r, r_over_d, _) = self.tail_parts(x);
        if r.abs() >= 1.0 {
            // |x - x_0| <= γ: z is small, so only 1 / (πγ) can overflow, and
            // then the density really is that large
            let z = 1.0 / r;
            f64_consts::FRAC_1_PI / self.scale / (1.0 + z * z)
        } else {
            // γ / (π (x - x_0)^2) / (1 + r^2); this never forms z, z^2 or
            // 1 / γ, which can overflow when the density does not
            f64_consts::FRAC_1_PI * r_over_d / (1.0 + r * r)
        }
    }

    /// Calculates the log probability density function for the cauchy
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// ln(1 / (πγ * (1 + ((x - x_0) / γ)^2)))
    /// ```
    ///
    /// where `x_0` is the location and `γ` is the scale
    fn ln_pdf(&self, x: f64) -> f64 {
        let z = (x - self.location) / self.scale;
        let d = f64_consts::PI * self.scale * (1.0 + z * z);
        if d.is_finite() {
            return -d.ln();
        }
        // the product overflowed (gh #472); sum the logs instead
        let (r, _, ln_abs_d) = self.tail_parts(x);
        if r.abs() >= 1.0 {
            let z = 1.0 / r;
            -f64_consts::PI.ln() - self.scale.ln() - (z * z).ln_1p()
        } else {
            // ln(γ / (π d^2) / (1 + r^2)), without forming d^2 or z
            -f64_consts::PI.ln() + self.scale.ln() - 2.0 * ln_abs_d - (r * r).ln_1p()
        }
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::internal::density_util;
    crate::distribution::internal::testing_boiler!(location: f64, scale: f64; Cauchy; CauchyError);

    #[test]
    fn test_create() {
        create_ok(0.0, 0.1);
        create_ok(0.0, 1.0);
        create_ok(0.0, 10.0);
        create_ok(10.0, 11.0);
        create_ok(-5.0, 100.0);
        create_ok(0.0, f64::INFINITY);
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_sample() {
        use rand::{distr::Distribution as _, rngs::StdRng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(437);
        assert!(create_ok(0.0, 1.0).sample(&mut rng).is_finite());
    }

    #[test]
    fn test_bad_create() {
        let invalid = [
            (f64::NAN, 1.0, CauchyError::LocationInvalid),
            (1.0, f64::NAN, CauchyError::ScaleInvalid),
            (f64::NAN, f64::NAN, CauchyError::LocationInvalid),
            (1.0, 0.0, CauchyError::ScaleInvalid),
        ];

        for (location, scale, err) in invalid {
            test_create_err(location, scale, err);
        }
    }

    #[test]
    fn test_entropy() {
        let entropy = |x: Cauchy| x.entropy().unwrap();
        test_exact(0.0, 2.0, 3.224171427529236102395, entropy);
        test_exact(0.1, 4.0, 3.917318608089181411812, entropy);
        test_exact(1.0, 10.0, 4.833609339963336476996, entropy);
        test_exact(10.0, 11.0, 4.92891951976766133704, entropy);
    }

    #[test]
    fn test_mode() {
        let mode = |x: Cauchy| x.mode().unwrap();
        test_exact(0.0, 2.0, 0.0, mode);
        test_exact(0.1, 4.0, 0.1, mode);
        test_exact(1.0, 10.0, 1.0, mode);
        test_exact(10.0, 11.0, 10.0, mode);
        test_exact(0.0, f64::INFINITY, 0.0, mode);
    }

    #[test]
    fn test_median() {
        let median = |x: Cauchy| x.median();
        test_exact(0.0, 2.0, 0.0, median);
        test_exact(0.1, 4.0, 0.1, median);
        test_exact(1.0, 10.0, 1.0, median);
        test_exact(10.0, 11.0, 10.0, median);
        test_exact(0.0, f64::INFINITY, 0.0, median);
    }

    #[test]
    fn test_min_max() {
        let min = |x: Cauchy| x.min();
        let max = |x: Cauchy| x.max();
        test_exact(0.0, 1.0, f64::NEG_INFINITY, min);
        test_exact(0.0, 1.0, f64::INFINITY, max);
    }

    #[test]
    fn test_pdf() {
        let pdf = |arg: f64| move |x: Cauchy| x.pdf(arg);
        test_exact(0.0, 0.1, 0.001272730452554141029739, pdf(-5.0));
        test_exact(0.0, 0.1, 0.03151583031522679916216, pdf(-1.0));
        test_absolute(0.0, 0.1, 3.183098861837906715378, 1e-14, pdf(0.0));
        test_exact(0.0, 0.1, 0.03151583031522679916216, pdf(1.0));
        test_exact(0.0, 0.1, 0.001272730452554141029739, pdf(5.0));
        test_absolute(0.0, 1.0, 0.01224268793014579505914, 1e-17, pdf(-5.0));
        test_exact(0.0, 1.0, 0.1591549430918953357689, pdf(-1.0));
        test_exact(0.0, 1.0, f64_consts::FRAC_1_PI, pdf(0.0));
        test_exact(0.0, 1.0, 0.1591549430918953357689, pdf(1.0));
        test_absolute(0.0, 1.0, 0.01224268793014579505914, 1e-17, pdf(5.0));
        test_exact(0.0, 10.0, 0.02546479089470325372302, pdf(-5.0));
        test_exact(0.0, 10.0, 0.03151583031522679916216, pdf(-1.0));
        test_exact(0.0, 10.0, 0.03183098861837906715378, pdf(0.0));
        test_exact(0.0, 10.0, 0.03151583031522679916216, pdf(1.0));
        test_exact(0.0, 10.0, 0.02546479089470325372302, pdf(5.0));
        test_exact(-5.0, 100.0, 0.003183098861837906715378, pdf(-5.0));
        test_absolute(-5.0, 100.0, 0.003178014039374906864395, 1e-17, pdf(-1.0));
        test_exact(-5.0, 100.0, 0.003175160959439308444267, pdf(0.0));
        test_exact(-5.0, 100.0, 0.003171680810918599756255, pdf(1.0));
        test_absolute(-5.0, 100.0, 0.003151583031522679916216, 1e-17, pdf(5.0));
        test_exact(0.0, f64::INFINITY, 0.0, pdf(-5.0));
        test_exact(0.0, f64::INFINITY, 0.0, pdf(-1.0));
        test_exact(0.0, f64::INFINITY, 0.0, pdf(0.0));
        test_exact(0.0, f64::INFINITY, 0.0, pdf(1.0));
        test_exact(0.0, f64::INFINITY, 0.0, pdf(5.0));
        test_exact(f64::INFINITY, 1.0, 0.0, pdf(-5.0));
        test_exact(f64::INFINITY, 1.0, 0.0, pdf(-1.0));
        test_exact(f64::INFINITY, 1.0, 0.0, pdf(0.0));
        test_exact(f64::INFINITY, 1.0, 0.0, pdf(1.0));
        test_exact(f64::INFINITY, 1.0, 0.0, pdf(5.0));
    }

    #[test]
    fn test_ln_pdf() {
        let ln_pdf = |arg: f64| move |x: Cauchy| x.ln_pdf(arg);
        test_exact(0.0, 0.1, -6.666590723732973542744, ln_pdf(-5.0));
        test_absolute(0.0, 0.1, -3.457265309696613941009, 1e-14, ln_pdf(-1.0));
        test_exact(0.0, 0.1, 1.157855207144645509875, ln_pdf(0.0));
        test_absolute(0.0, 0.1, -3.457265309696613941009, 1e-14, ln_pdf(1.0));
        test_exact(0.0, 0.1, -6.666590723732973542744, ln_pdf(5.0));
        test_exact(0.0, 1.0, -4.402826423870882219615, ln_pdf(-5.0));
        test_absolute(0.0, 1.0, -1.837877066409345483561, 1e-15, ln_pdf(-1.0));
        test_exact(0.0, 1.0, -1.144729885849400174143, ln_pdf(0.0));
        test_absolute(0.0, 1.0, -1.837877066409345483561, 1e-15, ln_pdf(1.0));
        test_exact(0.0, 1.0, -4.402826423870882219615, ln_pdf(5.0));
        test_exact(0.0, 10.0, -3.670458530157655613928, ln_pdf(-5.0));
        test_absolute(0.0, 10.0, -3.457265309696613941009, 1e-14, ln_pdf(-1.0));
        test_exact(0.0, 10.0, -3.447314978843445858161, ln_pdf(0.0));
        test_absolute(0.0, 10.0, -3.457265309696613941009, 1e-14, ln_pdf(1.0));
        test_exact(0.0, 10.0, -3.670458530157655613928, ln_pdf(5.0));
        test_exact(-5.0, 100.0, -5.749900071837491542179, ln_pdf(-5.0));
        test_exact(-5.0, 100.0, -5.751498793201188569872, ln_pdf(-1.0));
        test_exact(-5.0, 100.0, -5.75239695203607874116, ln_pdf(0.0));
        test_exact(-5.0, 100.0, -5.75349360734762171285, ln_pdf(1.0));
        test_exact(-5.0, 100.0, -5.759850402690659625027, ln_pdf(5.0));
        test_exact(0.0, f64::INFINITY, f64::NEG_INFINITY, ln_pdf(-5.0));
        test_exact(0.0, f64::INFINITY, f64::NEG_INFINITY, ln_pdf(-1.0));
        test_exact(0.0, f64::INFINITY, f64::NEG_INFINITY, ln_pdf(0.0));
        test_exact(0.0, f64::INFINITY, f64::NEG_INFINITY, ln_pdf(1.0));
        test_exact(0.0, f64::INFINITY, f64::NEG_INFINITY, ln_pdf(5.0));
        test_exact(f64::INFINITY, 1.0, f64::NEG_INFINITY, ln_pdf(-5.0));
        test_exact(f64::INFINITY, 1.0, f64::NEG_INFINITY, ln_pdf(-1.0));
        test_exact(f64::INFINITY, 1.0, f64::NEG_INFINITY, ln_pdf(0.0));
        test_exact(f64::INFINITY, 1.0, f64::NEG_INFINITY, ln_pdf(1.0));
        test_exact(f64::INFINITY, 1.0, f64::NEG_INFINITY, ln_pdf(5.0));
    }

    #[test]
    fn test_cdf() {
        let cdf = |arg: f64| move |x: Cauchy| x.cdf(arg);
        test_absolute(0.0, 0.1, 0.006365349100972796679298, 1e-16, cdf(-5.0));
        test_absolute(0.0, 0.1, 0.03172551743055356951498, 1e-16, cdf(-1.0));
        test_exact(0.0, 0.1, 0.5, cdf(0.0));
        test_exact(0.0, 0.1, 0.968274482569446430485, cdf(1.0));
        test_exact(0.0, 0.1, 0.9936346508990272033207, cdf(5.0));
        test_absolute(0.0, 1.0, 0.06283295818900118381375, 1e-16, cdf(-5.0));
        test_exact(0.0, 1.0, 0.25, cdf(-1.0));
        test_exact(0.0, 1.0, 0.5, cdf(0.0));
        test_exact(0.0, 1.0, 0.75, cdf(1.0));
        test_exact(0.0, 1.0, 0.9371670418109988161863, cdf(5.0));
        test_exact(0.0, 10.0, 0.3524163823495667258246, cdf(-5.0));
        test_exact(0.0, 10.0, 0.468274482569446430485, cdf(-1.0));
        test_exact(0.0, 10.0, 0.5, cdf(0.0));
        test_exact(0.0, 10.0, 0.531725517430553569515, cdf(1.0));
        test_exact(0.0, 10.0, 0.6475836176504332741754, cdf(5.0));
        test_exact(-5.0, 100.0, 0.5, cdf(-5.0));
        test_exact(-5.0, 100.0, 0.5127256113479918307809, cdf(-1.0));
        test_exact(-5.0, 100.0, 0.5159022512561763751816, cdf(0.0));
        test_exact(-5.0, 100.0, 0.5190757242358362337495, cdf(1.0));
        test_exact(-5.0, 100.0, 0.531725517430553569515, cdf(5.0));
        test_exact(0.0, f64::INFINITY, 0.5, cdf(-5.0));
        test_exact(0.0, f64::INFINITY, 0.5, cdf(-1.0));
        test_exact(0.0, f64::INFINITY, 0.5, cdf(0.0));
        test_exact(0.0, f64::INFINITY, 0.5, cdf(1.0));
        test_exact(0.0, f64::INFINITY, 0.5, cdf(5.0));
        test_exact(f64::INFINITY, 1.0, 0.0, cdf(-5.0));
        test_exact(f64::INFINITY, 1.0, 0.0, cdf(-1.0));
        test_exact(f64::INFINITY, 1.0, 0.0, cdf(0.0));
        test_exact(f64::INFINITY, 1.0, 0.0, cdf(1.0));
        test_exact(f64::INFINITY, 1.0, 0.0, cdf(5.0));
    }

    #[test]
    fn test_sf() {
        let sf = |arg: f64| move |x: Cauchy| x.sf(arg);
        test_absolute(0.0, 0.1, 0.9936346508990272, 1e-16, sf(-5.0));
        test_absolute(0.0, 0.1, 0.9682744825694465, 1e-16, sf(-1.0));
        test_exact(0.0, 0.1, 0.5, sf(0.0));
        test_absolute(0.0, 0.1, 0.03172551743055352, 1e-16, sf(1.0));
        test_exact(0.0, 0.1, 0.006365349100972806, sf(5.0));
        test_absolute(0.0, 1.0, 0.9371670418109989, 1e-16, sf(-5.0));
        test_exact(0.0, 1.0, 0.75, sf(-1.0));
        test_exact(0.0, 1.0, 0.5, sf(0.0));
        test_exact(0.0, 1.0, 0.25, sf(1.0));
        test_exact(0.0, 1.0, 0.06283295818900114, sf(5.0));
        test_exact(0.0, 10.0, 0.6475836176504333, sf(-5.0));
        test_exact(0.0, 10.0, 0.5317255174305535, sf(-1.0));
        test_exact(0.0, 10.0, 0.5, sf(0.0));
        test_exact(0.0, 10.0, 0.4682744825694464, sf(1.0));
        test_exact(0.0, 10.0, 0.35241638234956674, sf(5.0));
        test_exact(-5.0, 100.0, 0.5, sf(-5.0));
        test_exact(-5.0, 100.0, 0.4872743886520082, sf(-1.0));
        test_exact(-5.0, 100.0, 0.4840977487438236, sf(0.0));
        test_exact(-5.0, 100.0, 0.48092427576416374, sf(1.0));
        test_exact(-5.0, 100.0, 0.4682744825694464, sf(5.0));
        test_exact(0.0, f64::INFINITY, 0.5, sf(-5.0));
        test_exact(0.0, f64::INFINITY, 0.5, sf(-1.0));
        test_exact(0.0, f64::INFINITY, 0.5, sf(0.0));
        test_exact(0.0, f64::INFINITY, 0.5, sf(1.0));
        test_exact(0.0, f64::INFINITY, 0.5, sf(5.0));
        test_exact(f64::INFINITY, 1.0, 1.0, sf(-5.0));
        test_exact(f64::INFINITY, 1.0, 1.0, sf(-1.0));
        test_exact(f64::INFINITY, 1.0, 1.0, sf(0.0));
        test_exact(f64::INFINITY, 1.0, 1.0, sf(1.0));
        test_exact(f64::INFINITY, 1.0, 1.0, sf(5.0));
    }

    #[test]
    fn test_inverse_cdf() {
        let func = |arg: f64| move |x: Cauchy| x.inverse_cdf(x.cdf(arg));
        test_absolute(0.0, 0.1, -5.0, 1e-10, func(-5.0));
        test_absolute(0.0, 0.1, -1.0, 1e-14, func(-1.0));
        test_exact(0.0, 0.1, 0.0, func(0.0));
        test_absolute(0.0, 0.1, 1.0, 1e-14, func(1.0));
        test_absolute(0.0, 0.1, 5.0, 1e-10, func(5.0));
        test_absolute(0.0, 1.0, -5.0, 1e-14, func(-5.0));
        test_absolute(0.0, 1.0, -1.0, 1e-15, func(-1.0));
        test_exact(0.0, 1.0, 0.0, func(0.0));
        test_absolute(0.0, 1.0, 1.0, 1e-15, func(1.0));
        test_absolute(0.0, 1.0, 5.0, 1e-14, func(5.0));
        test_absolute(0.0, 10.0, -5.0, 1e-14, func(-5.0));
        test_absolute(0.0, 10.0, -1.0, 1e-14, func(-1.0));
        test_exact(0.0, 10.0, 0.0, func(0.0));
        test_absolute(0.0, 10.0, 1.0, 1e-14, func(1.0));
        test_absolute(0.0, 10.0, 5.0, 1e-14, func(5.0));
        test_exact(-5.0, 100.0, -5.0, func(-5.0));
        test_absolute(-5.0, 100.0, -1.0, 1e-10, func(-1.0));
        test_absolute(-5.0, 100.0, 0.0, 1e-14, func(0.0));
        test_absolute(-5.0, 100.0, 1.0, 1e-14, func(1.0));
        test_absolute(-5.0, 100.0, 5.0, 1e-10, func(5.0));
    }

    #[test]
    fn test_continuous() {
        density_util::check_continuous_distribution(&create_ok(-1.2, 3.4), -1500.0, 1500.0);
        density_util::check_continuous_distribution(&create_ok(-4.5, 6.7), -5000.0, 5000.0);
    }

    #[test]
    fn test_pdf_far_tail_and_huge_scale() {
        // gh #472: the density is representable here but used to come back as 0
        let d = create_ok(0.0, 1.0);
        let got = d.pdf(7.6e153);
        assert!((got - 5.5109052317138284e-309).abs() <= 1e-15 * 5.5109052317138284e-309, "{got:e}");
        assert!(d.pdf(1e160) > 0.0);
        approx::assert_relative_eq!(d.ln_pdf(1e160), -(core::f64::consts::PI.ln()) - 2.0 * 1e160f64.ln(), max_relative = 1e-15);
        approx::assert_relative_eq!(d.ln_pdf(1e300), -(core::f64::consts::PI.ln()) - 2.0 * 1e300f64.ln(), max_relative = 1e-15);
        let wide = create_ok(0.0, 1e308);
        approx::assert_relative_eq!(wide.pdf(0.0), core::f64::consts::FRAC_1_PI / 1e308, max_relative = 1e-15);
        approx::assert_relative_eq!(wide.ln_pdf(0.0), -(core::f64::consts::PI.ln()) - 1e308f64.ln(), max_relative = 1e-15);
    }

    #[test]
    fn test_pdf_tail_without_overflowing_intermediates() {
        use core::f64::consts::{FRAC_1_PI, PI};
        // z = (x - x_0) / γ overflows, but γ / (π x^2) is a (subnormal) number
        let tiny = create_ok(0.0, 1e-300);
        let want = FRAC_1_PI * 1e-300 / 1e18;
        approx::assert_relative_eq!(tiny.pdf(1e9), want, max_relative = 1e-6);
        approx::assert_relative_eq!(tiny.ln_pdf(1e9), -PI.ln() + 1e-300f64.ln() - 2.0 * 1e9f64.ln(), max_relative = 1e-15);
        // x - x_0 overflows, though z = 2e308 / 1e308 = 2
        let wide = create_ok(-1e308, 1e308);
        approx::assert_relative_eq!(wide.pdf(1e308), FRAC_1_PI / 1e308 / 5.0, max_relative = 1e-14);
        approx::assert_relative_eq!(wide.ln_pdf(1e308), -PI.ln() - 1e308f64.ln() - 5f64.ln(), max_relative = 1e-14);
        // 1 / γ overflows for a subnormal γ, but the density is about 1 / π
        let sub = create_ok(0.0, 1e-310);
        approx::assert_relative_eq!(sub.pdf(1e-155), FRAC_1_PI, max_relative = 1e-9);
        approx::assert_relative_eq!(sub.ln_pdf(1e-155), -PI.ln(), max_relative = 1e-9);
        assert_eq!(tiny.pdf(f64::INFINITY), 0.0);
        assert_eq!(tiny.ln_pdf(f64::NEG_INFINITY), f64::NEG_INFINITY);
    }
}
