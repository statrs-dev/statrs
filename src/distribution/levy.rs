use crate::distribution::{Continuous, ContinuousCDF};
use crate::function::erf::{erf, erfc, erfc_inv};

use crate::statistics::*;
use core::f64;

/// Implements the [Levy](https://en.wikipedia.org/wiki/L%C3%A9vy_distribution) distribution.
///
/// # Example
///
/// ```
/// use statrs::distribution::{Levy, Continuous};
/// use statrs::statistics::Distribution;
///
/// let n = Levy::new(1.0, 1.0).unwrap();
/// assert_eq!(n.pdf(0.0), 0.0);
/// ```
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Levy {
    mu: f64,
    c: f64,
}

/// Represents the errors that can occur when creating a [`Levy`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum LevyError {
    /// Location is NaN or infinite
    LocationInvalid,
    /// Scale is NaN, infinite or nonpositive
    ScaleInvalid,
}

impl std::fmt::Display for LevyError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            LevyError::LocationInvalid => write!(f, "location is NaN or infinite"),
            LevyError::ScaleInvalid => write!(f, "scale is NaN, infinite or nonpositive"),
        }
    }
}

impl std::error::Error for LevyError {}

impl Levy {
    /// Constructs a new Levy distribution with a location (μ) and dispersion (c)
    ///
    /// # Errors
    ///
    /// Returns and error if `mu` is NaN or infinite or if `c` is NaN, infinite or nonpositive
    ///
    /// # Example
    ///
    /// ```
    /// use statrs::distribution::Levy;
    ///
    /// let mut result = Levy::new(0.0, 1.0);
    /// assert!(result.is_ok());
    ///
    /// result = Levy::new(0.0, 0.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(mu: f64, c: f64) -> Result<Levy, LevyError> {
        if mu.is_nan() || mu.is_infinite() {
            return Err(LevyError::LocationInvalid);
        }
        if c.is_nan() || c.is_infinite() || c <= 0.0 {
            return Err(LevyError::ScaleInvalid);
        }
        Ok(Levy { mu, c })
    }

    /// Returns the location (μ) of the Levy distribution
    ///
    /// # Example
    ///
    /// ```
    /// use statrs::distribution::Levy;
    ///
    /// let n = Levy::new(1.0, 1.0).unwrap();
    /// assert_eq!(n.mu(), 1.0);
    /// ```
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Returns the dispersion (c) of the Levy distribution
    ///
    /// # Example
    ///
    /// ```
    /// use statrs::distribution::Levy;
    ///
    /// let n = Levy::new(1.0, 1.0).unwrap();
    /// assert_eq!(n.c(), 1.0);
    /// ```
    pub fn c(&self) -> f64 {
        self.c
    }
}

impl std::fmt::Display for Levy {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Levy(mu = {}, c = {})", self.mu, self.c)
    }
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl ::rand::distr::Distribution<f64> for Levy {
    fn sample<R: ::rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand::distr::OpenClosed01;

        // Inverse transform sampling
        let u: f64 = rng.sample(OpenClosed01);
        self.mu + (0.5 * self.c) / erfc_inv(u).powf(2.0)
    }
}

impl ContinuousCDF<f64, f64> for Levy {
    /// Calculates the cumulative distribution function for the Levy distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// 0 if x <= μ
    /// erfc(sqrt(c / (2 * (x - μ)))) if x > μ
    /// ```
    ///
    /// where `μ` is the location, `c` is the dispersion, and `erfc` is the
    /// complementary error function.
    fn cdf(&self, x: f64) -> f64 {
        if x <= self.mu {
            0.0
        } else if x > 0.0 && x.is_infinite() {
            1.0
        } else {
            erfc(((0.5 * self.c) / (x - self.mu)).sqrt())
        }
    }

    /// Calculates the survival function for the Levy distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// 1 if x <= μ
    /// erf(sqrt(c / (2 * (x - μ)))) if x > μ
    /// ```
    ///
    /// where `μ` is the location, `c` is the dispersion, and `erf` is the error
    /// function.
    fn sf(&self, x: f64) -> f64 {
        if x <= self.mu {
            1.0
        } else if x > 0.0 && x.is_infinite() {
            0.0
        } else {
            erf(((0.5 * self.c) / (x - self.mu)).sqrt())
        }
    }

    /// Calculates the inverse cumulative distribution function for the
    /// normal distribution at `x`.
    ///
    /// # Panics
    ///
    /// If `x < 0.0` or `x > 1.0`
    ///
    /// # Formula
    ///
    /// ```text
    /// μ + c * (erfc_inv(x)^2)/2
    /// ```
    ///
    /// where `μ` is the mean, `σ` is the standard deviation and `erfc_inv` is
    /// the inverse of the complementary error function
    fn inverse_cdf(&self, x: f64) -> f64 {
        if !(0.0..=1.0).contains(&x) {
            panic!("x must be in [0, 1]");
        } else {
            self.mu + 0.5 * self.c / (erfc_inv(x).powf(2.0))
        }
    }
}

impl Min<f64> for Levy {
    /// Returns the minimum value in the domain of the
    /// Levy distribution representable by a double precision float
    ///
    /// # Formula
    ///
    /// ```text
    /// μ
    /// ```
    fn min(&self) -> f64 {
        self.mu
    }
}

impl Max<f64> for Levy {
    /// Returns the maximum value in the domain of the
    /// Levy distribution representable by a double precision float
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

impl Distribution<f64> for Levy {
    /// Returns the mean of the Levy distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// f64::INFINITY
    /// ```
    fn mean(&self) -> Option<f64> {
        Some(f64::INFINITY)
    }

    /// Returns the variance of the Levy distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// f64::INFINITY
    /// ```
    fn variance(&self) -> Option<f64> {
        Some(f64::INFINITY)
    }

    /// Returns the standard deviation of the Levy distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// f64::INFINITY
    /// ```
    fn std_dev(&self) -> Option<f64> {
        Some(f64::INFINITY)
    }

    /// Returns the entropy of the Levy distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// (1 + 3γ + ln(16πc^2))/2
    /// ```
    fn entropy(&self) -> Option<f64> {
        /// CONSTANT_PART = 1.5 * EULER_MASCHERONI + 0.5 * (1.0 + LN_PI + 16.0_f64.ln())
        const CONSTANT_PART: f64 = 3.32448280139688989720525569282472133636474609375;
        Some(CONSTANT_PART + self.c.ln())
    }
}

impl Median<f64> for Levy {
    /// Returns the median of the Levy distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// μ + c/(2 * erfc_inv(0.5)^2)
    /// ```
    ///
    /// where `μ` is the mean, `c` is the dispersion and `erfc_inv` is
    /// the inverse of the complementary error function.
    fn median(&self) -> f64 {
        self.mu + self.c * 0.5 * erfc_inv(0.5).powf(-2.0)
    }
}

impl Mode<Option<f64>> for Levy {
    /// Returns the mode of the Levy distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// μ + c/3
    /// ```
    ///
    /// where `μ` is the mean and `c` is the dispersion.
    fn mode(&self) -> Option<f64> {
        Some(self.mu + self.c / 3.0)
    }
}

impl Continuous<f64, f64> for Levy {
    /// Calculates the probability density function for the Levy distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// (sqrt(c / 2 * π)) * e^(-1/2 * (c / (x - μ))) * (1 / (x - μ)^(3/2))
    /// ```
    ///
    /// where `μ` is the mean and `c` is the dispersion.
    fn pdf(&self, x: f64) -> f64 {
        if x <= self.mu {
            0.0
        } else {
            let diff = x - self.mu;
            (self.c / f64::consts::TAU).sqrt() * (-((0.5 * self.c) / diff)).exp() / diff.powf(1.5)
        }
    }

    /// Calculates the log probability density function for the Levy distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// 1/2 * (ln(c) - ln(2 * π) - (c / (x - μ))) - 3/2 * ln(x - μ)
    /// ```
    ///
    /// where `μ` is the mean and `σ` is the standard deviation
    fn ln_pdf(&self, x: f64) -> f64 {
        use crate::consts::LN_SQRT_2PI;

        if x <= self.mu {
            f64::NEG_INFINITY
        } else {
            let diff = x - self.mu;
            0.5 * (self.c.ln() - self.c / diff) - (1.5 * diff.ln() + LN_SQRT_2PI)
        }
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use core::f64;

    use super::*;
    use crate::distribution::internal::*;
    use crate::testing_boiler;

    testing_boiler!(mu: f64, c: f64; Levy; LevyError);

    #[test]
    fn test_create() {
        create_ok(10.0, 0.1);
        create_ok(5.0, 1.0);
        create_ok(0.1, 10.0);
        create_ok(10.0, 100.0);
    }

    #[test]
    fn test_bad_create() {
        test_create_err(1.0, -1.0, LevyError::ScaleInvalid);
        test_create_err(f64::NAN, 1.0, LevyError::LocationInvalid);
        let invalid = [
            (0.0, 0.0),
            (-1.0, -1.0),
            (f64::NAN, 1.0),
            (1.0, f64::NAN),
            (f64::NAN, f64::NAN),
            (f64::INFINITY, 1.0),
            (1.0, f64::INFINITY),
            (f64::INFINITY, f64::INFINITY),
            (f64::NEG_INFINITY, 1.0),
            (1.0, f64::NEG_INFINITY),
            (f64::NEG_INFINITY, f64::NEG_INFINITY),
        ];
        for (a, b) in invalid {
            create_err(a, b);
        }
    }

    #[test]
    fn test_mean() {
        let mean = |x: Levy| x.mean().unwrap();
        test_exact(1.0, 3.0, f64::INFINITY, mean);
    }

    #[test]
    fn test_variance() {
        let variance = |x: Levy| x.variance().unwrap();
        test_exact(1.0, 3.0, f64::INFINITY, variance);
    }

    #[test]
    fn test_entropy() {
        let entropy = |x: Levy| x.entropy().unwrap();
        test_exact(
            0.1,
            0.1,
            1.0218977084028444402008517499780282378196716308593750000000000000,
            entropy,
        );
        test_exact(
            1.0,
            1.0,
            3.3244828013968898972052556928247213363647460937500000000000000000,
            entropy,
        );
        test_exact(
            10.0,
            10.0,
            5.6270678943909357982988694857340306043624877929687500000000000000,
            entropy,
        );
        test_exact(
            3.0,
            1.0,
            3.32448280139688989720525569282472133636474609375,
            entropy,
        );
        test_exact(
            1.0,
            3.0,
            4.42309509006499990135807820479385554790496826171875,
            entropy,
        );
    }

    #[test]
    fn test_median() {
        let median = |x: Levy| x.median();
        test_exact(
            1.0,
            1.0,
            3.198109338317732142087379543227143585681915283203125,
            median,
        );
        test_exact(
            1.0,
            3.0,
            7.5943280149531968703513484797440469264984130859375,
            median,
        );
        test_exact(
            3.0,
            1.0,
            5.198109338317731697998169693164527416229248046875,
            median,
        );
        test_exact(
            3.0,
            3.0,
            9.5943280149531968703513484797440469264984130859375,
            median,
        );
    }

    #[test]
    fn test_mode() {
        let mode = |x: Levy| x.mode().unwrap();
        test_exact(1.0, 1.0, 4.0 / 3.0, mode);
        test_exact(1.0, 3.0, 2.0, mode);
        test_exact(3.0, 1.0, 10.0 / 3.0, mode);
        test_exact(3.0, 3.0, 4.0, mode);
    }

    #[test]
    fn test_min() {
        let min = |x: Levy| x.min();
        test_exact(1.0, 1.0, 1.0, min);
        test_exact(1.0, 3.0, 1.0, min);
        test_exact(3.0, 1.0, 3.0, min);
        test_exact(3.0, 3.0, 3.0, min);
    }

    #[test]
    fn test_max() {
        let max = |x: Levy| x.max();
        test_relative(1.0, 1.0, f64::INFINITY, max);
    }

    #[test]
    fn test_pdf_input_outside_support() {
        let pdf = |arg: f64| move |x: Levy| x.pdf(arg);
        test_relative(1.0, 1.0, 0.0, pdf(1.0));
        test_relative(1.0, 1.0, 0.0, pdf(-1.0));
    }

    #[test]
    fn test_ln_pdf_input_outside_support() {
        let ln_pdf = |arg: f64| move |x: Levy| x.ln_pdf(arg);
        test_relative(1.0, 1.0, f64::NEG_INFINITY, ln_pdf(1.0));
        test_relative(1.0, 1.0, f64::NEG_INFINITY, ln_pdf(-1.0));
    }

    #[test]
    fn test_cdf_input_outside_support() {
        let cdf = |arg: f64| move |x: Levy| x.cdf(arg);
        test_relative(1.0, 1.0, 0.0, cdf(1.0));
        test_relative(1.0, 1.0, 0.0, cdf(-1.0));
    }

    #[test]
    fn test_sf_input_outside_support() {
        let sf = |arg: f64| move |x: Levy| x.sf(arg);
        test_relative(1.0, 1.0, 1.0, sf(1.0));
        test_relative(1.0, 1.0, 1.0, sf(-1.0));
    }

    #[test]
    fn test_pdf() {
        let pdf = |arg: f64| move |x: Levy| x.pdf(arg);
        test_absolute(
            1.0,
            1.0,
            0.000278563104187508890109692405445684926235117018222808837890625,
            1e-14,
            pdf(127.721),
        );
        test_absolute(
            1.0,
            3.0,
            0.00047869297068690841966132065721239996491931378841400146484375,
            1e-14,
            pdf(127.721),
        );
        test_absolute(
            3.0,
            1.0,
            0.0002852723142770544561240553260716978911659680306911468505859375,
            1e-14,
            pdf(127.721),
        );
        test_absolute(
            3.0,
            3.0,
            0.000490160290539595430193975378330151215777732431888580322265625,
            1e-14,
            pdf(127.721),
        );
        test_absolute(
            1.0,
            10.0,
            0.00085016128857238830972276044661839478067122399806976318359375,
            1e-14,
            pdf(127.721),
        );
        test_absolute(
            10.0,
            1.0,
            0.000311017309166119158836405489410026348195970058441162109375,
            1e-14,
            pdf(127.721),
        );
        test_absolute(
            10.0,
            10.0,
            0.000946636464835523094678293443138272778014652431011199951171875,
            1e-14,
            pdf(127.721),
        );
    }

    #[test]
    fn test_cdf() {
        let cdf = |arg: f64| move |x: Levy| x.cdf(arg);
        test_absolute(
            1.0,
            1.0,
            0.92921440758307749518962737056426703929901123046875,
            1e-14,
            cdf(127.721),
        );
        test_absolute(
            1.0,
            3.0,
            0.87771716176899639005881681441678665578365325927734375,
            1e-14,
            cdf(127.721),
        );
        test_absolute(
            3.0,
            1.0,
            0.92865061651845781653946687583811581134796142578125,
            1e-14,
            cdf(127.721),
        );
        test_absolute(
            3.0,
            3.0,
            0.87674838391215370592135514016263186931610107421875,
            1e-14,
            cdf(127.721),
        );
        test_absolute(
            1.0,
            10.0,
            0.77877521115243608651468321113497950136661529541015625,
            1e-14,
            cdf(127.721),
        );
        test_absolute(
            10.0,
            1.0,
            0.92656576512976263071408311589038930833339691162109375,
            1e-14,
            cdf(127.721),
        );
        test_absolute(
            10.0,
            10.0,
            0.7707025761090431359434660407714545726776123046875,
            1e-14,
            cdf(127.721),
        );
    }

    #[test]
    fn test_inverse_cdf() {
        let inverse_cdf = |arg: f64| move |x: Levy| x.inverse_cdf(arg);
        test_exact(
            1.0,
            1.0,
            6366.8643851062215617275796830654144287109375,
            inverse_cdf(0.99),
        );
        test_exact(
            1.0,
            3.0,
            19098.59315531866377568803727626800537109375,
            inverse_cdf(0.99),
        );
        test_exact(
            3.0,
            1.0,
            6368.8643851062215617275796830654144287109375,
            inverse_cdf(0.99),
        );
        test_exact(
            3.0,
            3.0,
            19100.59315531866377568803727626800537109375,
            inverse_cdf(0.99),
        );
        test_exact(
            1.0,
            10.0,
            63659.6438510622174362652003765106201171875,
            inverse_cdf(0.99),
        );
        test_exact(
            10.0,
            1.0,
            6375.8643851062215617275796830654144287109375,
            inverse_cdf(0.99),
        );
        test_exact(
            10.0,
            10.0,
            63668.6438510622174362652003765106201171875,
            inverse_cdf(0.99),
        );
    }

    #[test]
    fn test_sf() {
        let sf = |arg: f64| move |x: Levy| x.sf(arg);
        test_absolute(
            1.0,
            1.0,
            0.07078559241692249093258482162127620540559291839599609375,
            1e-14,
            sf(127.721),
        );
        test_absolute(
            1.0,
            3.0,
            0.12228283823100359606339537776875658892095088958740234375,
            1e-14,
            sf(127.721),
        );
        test_absolute(
            3.0,
            1.0,
            0.07134938348154218346053312416188418865203857421875,
            1e-14,
            sf(127.721),
        );
        test_absolute(
            3.0,
            3.0,
            0.1232516160878462663230692442084546200931072235107421875,
            1e-14,
            sf(127.721),
        );
        test_absolute(
            1.0,
            10.0,
            0.22122478884756391348531678886502049863338470458984375,
            1e-14,
            sf(127.721),
        );
        test_absolute(
            10.0,
            1.0,
            0.07343423487023741091928030755298095755279064178466796875,
            1e-14,
            sf(127.721),
        );
        test_absolute(
            10.0,
            10.0,
            0.2292974238909568363009583435996319167315959930419921875,
            1e-14,
            sf(127.721),
        );
    }

    #[test]
    fn test_continuous() {
        test::check_continuous_distribution(
            &create_ok(1.0, 0.05),
            1.0,
            319.2932192553111008237465284764766693115234375,
        );
        test::check_continuous_distribution(
            &create_ok(3.0, 0.05),
            3.0,
            321.2932192553111008237465284764766693115234375,
        );
        test::check_continuous_distribution(
            &create_ok(3.0, 0.05),
            3.0,
            321.2932192553111008237465284764766693115234375,
        );
        test::check_continuous_distribution(
            &create_ok(1.0, 0.05),
            1.0,
            319.2932192553111008237465284764766693115234375,
        );
        test::check_continuous_distribution(
            &create_ok(10.0, 0.05),
            10.0,
            328.2932192553111008237465284764766693115234375,
        );
        test::check_continuous_distribution(
            &create_ok(10.0, 0.05),
            10.0,
            328.2932192553111008237465284764766693115234375,
        );
    }
}
