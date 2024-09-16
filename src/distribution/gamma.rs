use crate::distribution::{Continuous, ContinuousCDF};
use crate::function::gamma;
use crate::prec;
use crate::statistics::*;

/// Implements the [Gamma](https://en.wikipedia.org/wiki/Gamma_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Gamma, Continuous};
/// use statrs::statistics::Distribution;
/// use statrs::prec;
///
/// let n = Gamma::new(3.0, 1.0).unwrap();
/// assert_eq!(n.mean().unwrap(), 3.0);
/// assert!(prec::almost_eq(n.pdf(2.0), 0.270670566473225383788, 1e-15));
/// ```
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Gamma {
    shape: f64,
    rate: f64,
}

/// Represents the errors that can occur when creating a [`Gamma`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum GammaError {
    /// The shape is NaN, zero or less than zero.
    ShapeInvalid,

    /// The rate is NaN, zero or less than zero.
    RateInvalid,

    /// The shape and rate are both infinite.
    ShapeAndRateInfinite,
}

impl std::fmt::Display for GammaError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            GammaError::ShapeInvalid => write!(f, "Shape is NaN zero, or less than zero."),
            GammaError::RateInvalid => write!(f, "Rate is NaN zero, or less than zero."),
            GammaError::ShapeAndRateInfinite => write!(f, "Shape and rate are infinite"),
        }
    }
}

impl std::error::Error for GammaError {}

impl Gamma {
    /// Constructs a new gamma distribution with a shape (α)
    /// of `shape` and a rate (β) of `rate`
    ///
    /// # Errors
    ///
    /// Returns an error if `shape` is 'NaN' or inf or `rate` is `NaN` or inf.
    /// Also returns an error if `shape <= 0.0` or `rate <= 0.0`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Gamma;
    ///
    /// let mut result = Gamma::new(3.0, 1.0);
    /// assert!(result.is_ok());
    ///
    /// result = Gamma::new(0.0, 0.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(shape: f64, rate: f64) -> Result<Gamma, GammaError> {
        if shape.is_nan() || shape <= 0.0 {
            return Err(GammaError::ShapeInvalid);
        }

        if rate.is_nan() || rate <= 0.0 {
            return Err(GammaError::RateInvalid);
        }

        if shape.is_infinite() && rate.is_infinite() {
            return Err(GammaError::ShapeAndRateInfinite);
        }

        Ok(Gamma { shape, rate })
    }

    /// Returns the shape (α) of the gamma distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Gamma;
    ///
    /// let n = Gamma::new(3.0, 1.0).unwrap();
    /// assert_eq!(n.shape(), 3.0);
    /// ```
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// Returns the rate (β) of the gamma distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Gamma;
    ///
    /// let n = Gamma::new(3.0, 1.0).unwrap();
    /// assert_eq!(n.rate(), 1.0);
    /// ```
    pub fn rate(&self) -> f64 {
        self.rate
    }
}

impl std::fmt::Display for Gamma {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Γ({}, {})", self.shape, self.rate)
    }
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl ::rand::distributions::Distribution<f64> for Gamma {
    fn sample<R: ::rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        sample_unchecked(rng, self.shape, self.rate)
    }
}

impl ContinuousCDF<f64, f64> for Gamma {
    /// Calculates the cumulative distribution function for the gamma
    /// distribution
    /// at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// (1 / Γ(α)) * γ(α, β * x)
    /// ```
    ///
    /// where `α` is the shape, `β` is the rate, `Γ` is the gamma function,
    /// and `γ` is the lower incomplete gamma function
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else if ulps_eq!(x, self.shape) && self.rate.is_infinite() {
            1.0
        } else if self.rate.is_infinite() {
            0.0
        } else if x.is_infinite() {
            1.0
        } else {
            gamma::gamma_lr(self.shape, x * self.rate)
        }
    }

    /// Calculates the survival function for the gamma
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// (1 / Γ(α)) * γ(α, β * x)
    /// ```
    ///
    /// where `α` is the shape, `β` is the rate, `Γ` is the gamma function,
    /// and `γ` is the upper incomplete gamma function
    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            1.0
        } else if ulps_eq!(x, self.shape) && self.rate.is_infinite() {
            0.0
        } else if self.rate.is_infinite() {
            1.0
        } else if x.is_infinite() {
            0.0
        } else {
            gamma::gamma_ur(self.shape, x * self.rate)
        }
    }

    fn inverse_cdf(&self, p: f64) -> f64 {
        if !(0.0..=1.0).contains(&p) {
            panic!("default inverse_cdf implementation should be provided probability on [0,1]")
        }
        if p == 0.0 {
            return self.min();
        };
        if p == 1.0 {
            return self.max();
        };

        // Bisection search for MAX_ITERS.0 iterations
        let mut high = 2.0;
        let mut low = 1.0;
        while self.cdf(low) > p {
            low /= 2.0;
        }
        while self.cdf(high) < p {
            high *= 2.0;
        }
        let mut x_0 = (high + low) / 2.0;

        for _ in 0..8 {
            if self.cdf(x_0) >= p {
                high = x_0;
            } else {
                low = x_0;
            }
            if prec::convergence(&mut x_0, (high + low) / 2.0) {
                break;
            }
        }

        // Newton Raphson, for at least one step
        for _ in 0..4 {
            let x_next = x_0 - (self.cdf(x_0) - p) / self.pdf(x_0);
            if prec::convergence(&mut x_0, x_next) {
                break;
            }
        }

        x_0
    }
}

impl Min<f64> for Gamma {
    /// Returns the minimum value in the domain of the
    /// gamma distribution representable by a double precision
    /// float
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

impl Max<f64> for Gamma {
    /// Returns the maximum value in the domain of the
    /// gamma distribution representable by a double precision
    /// float
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

impl Distribution<f64> for Gamma {
    /// Returns the mean of the gamma distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// α / β
    /// ```
    ///
    /// where `α` is the shape and `β` is the rate
    fn mean(&self) -> Option<f64> {
        Some(self.shape / self.rate)
    }

    /// Returns the variance of the gamma distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// α / β^2
    /// ```
    ///
    /// where `α` is the shape and `β` is the rate
    fn variance(&self) -> Option<f64> {
        Some(self.shape / (self.rate * self.rate))
    }

    /// Returns the entropy of the gamma distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// α - ln(β) + ln(Γ(α)) + (1 - α) * ψ(α)
    /// ```
    ///
    /// where `α` is the shape, `β` is the rate, `Γ` is the gamma function,
    /// and `ψ` is the digamma function
    fn entropy(&self) -> Option<f64> {
        let entr = self.shape - self.rate.ln()
            + gamma::ln_gamma(self.shape)
            + (1.0 - self.shape) * gamma::digamma(self.shape);
        Some(entr)
    }

    /// Returns the skewness of the gamma distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// 2 / sqrt(α)
    /// ```
    ///
    /// where `α` is the shape
    fn skewness(&self) -> Option<f64> {
        Some(2.0 / self.shape.sqrt())
    }
}

impl Mode<Option<f64>> for Gamma {
    /// Returns the mode for the gamma distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// (α - 1) / β, where α≥1
    /// ```
    ///
    /// where `α` is the shape and `β` is the rate
    fn mode(&self) -> Option<f64> {
        if self.shape < 1.0 {
            None
        } else {
            Some((self.shape - 1.0) / self.rate)
        }
    }
}

impl Continuous<f64, f64> for Gamma {
    /// Calculates the probability density function for the gamma distribution
    /// at `x`
    ///
    /// # Remarks
    ///
    /// Returns `NAN` if any of `shape` or `rate` are `f64::INFINITY`
    /// or if `x` is `f64::INFINITY`
    ///
    /// # Formula
    ///
    /// ```text
    /// (β^α / Γ(α)) * x^(α - 1) * e^(-β * x)
    /// ```
    ///
    /// where `α` is the shape, `β` is the rate, and `Γ` is the gamma function
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else if ulps_eq!(self.shape, 1.0) {
            self.rate * (-self.rate * x).exp()
        } else if self.shape > 160.0 {
            self.ln_pdf(x).exp()
        } else if x.is_infinite() {
            0.0
        } else {
            self.rate.powf(self.shape) * x.powf(self.shape - 1.0) * (-self.rate * x).exp()
                / gamma::gamma(self.shape)
        }
    }

    /// Calculates the log probability density function for the gamma
    /// distribution
    /// at `x`
    ///
    /// # Remarks
    ///
    /// Returns `NAN` if any of `shape` or `rate` are `f64::INFINITY`
    /// or if `x` is `f64::INFINITY`
    ///
    /// # Formula
    ///
    /// ```text
    /// ln((β^α / Γ(α)) * x^(α - 1) * e ^(-β * x))
    /// ```
    ///
    /// where `α` is the shape, `β` is the rate, and `Γ` is the gamma function
    fn ln_pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            f64::NEG_INFINITY
        } else if ulps_eq!(self.shape, 1.0) {
            self.rate.ln() - self.rate * x
        } else if x.is_infinite() {
            f64::NEG_INFINITY
        } else {
            self.shape * self.rate.ln() + (self.shape - 1.0) * x.ln()
                - self.rate * x
                - gamma::ln_gamma(self.shape)
        }
    }
}
/// Samples from a gamma distribution with a shape of `shape` and a
/// rate of `rate` using `rng` as the source of randomness. Implementation from:
///
/// _"A Simple Method for Generating Gamma Variables"_ - Marsaglia & Tsang
///
/// ACM Transactions on Mathematical Software, Vol. 26, No. 3, September 2000,
/// Pages 363-372
#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
pub fn sample_unchecked<R: ::rand::Rng + ?Sized>(rng: &mut R, shape: f64, rate: f64) -> f64 {
    let mut a = shape;
    let mut afix = 1.0;
    if shape < 1.0 {
        a = shape + 1.0;
        afix = rng.gen::<f64>().powf(1.0 / shape);
    }

    let d = a - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let mut x;
        let mut v;
        loop {
            x = super::normal::sample_unchecked(rng, 0.0, 1.0);
            v = 1.0 + c * x;
            if v > 0.0 {
                break;
            };
        }

        v = v * v * v;
        x = x * x;
        let u: f64 = rng.gen();
        if u < 1.0 - 0.0331 * x * x || u.ln() < 0.5 * x + d * (1.0 - v + v.ln()) {
            return afix * d * v / rate;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::internal::*;
    use crate::testing_boiler;

    testing_boiler!(shape: f64, rate: f64; Gamma; GammaError);

    #[test]
    fn test_create() {
        let valid = [
            (1.0, 0.1),
            (1.0, 1.0),
            (10.0, 10.0),
            (10.0, 1.0),
            (10.0, f64::INFINITY),
        ];

        for (s, r) in valid {
            create_ok(s, r);
        }
    }

    #[test]
    fn test_bad_create() {
        let invalid = [
            (0.0, 0.0, GammaError::ShapeInvalid),
            (1.0, f64::NAN, GammaError::RateInvalid),
            (1.0, -1.0, GammaError::RateInvalid),
            (-1.0, 1.0, GammaError::ShapeInvalid),
            (-1.0, -1.0, GammaError::ShapeInvalid),
            (-1.0, f64::NAN, GammaError::ShapeInvalid),
            (
                f64::INFINITY,
                f64::INFINITY,
                GammaError::ShapeAndRateInfinite,
            ),
        ];
        for (s, r, err) in invalid {
            test_create_err(s, r, err);
        }
    }

    #[test]
    fn test_mean() {
        let f = |x: Gamma| x.mean().unwrap();
        let test = [
            ((1.0, 0.1), 10.0),
            ((1.0, 1.0), 1.0),
            ((10.0, 10.0), 1.0),
            ((10.0, 1.0), 10.0),
            ((10.0, f64::INFINITY), 0.0),
        ];
        for ((s, r), res) in test {
            test_relative(s, r, res, f);
        }
    }

    #[test]
    fn test_variance() {
        let f = |x: Gamma| x.variance().unwrap();
        let test = [
            ((1.0, 0.1), 100.0),
            ((1.0, 1.0), 1.0),
            ((10.0, 10.0), 0.1),
            ((10.0, 1.0), 10.0),
            ((10.0, f64::INFINITY), 0.0),
        ];
        for ((s, r), res) in test {
            test_relative(s, r, res, f);
        }
    }

    #[test]
    fn test_entropy() {
        let f = |x: Gamma| x.entropy().unwrap();
        let test = [
            ((1.0, 0.1), 3.302585092994045628506840223),
            ((1.0, 1.0), 1.0),
            ((10.0, 10.0), 0.2334690854869339583626209),
            ((10.0, 1.0), 2.53605417848097964238061239),
            ((10.0, f64::INFINITY), f64::NEG_INFINITY),
        ];
        for ((s, r), res) in test {
            test_relative(s, r, res, f);
        }
    }

    #[test]
    fn test_skewness() {
        let f = |x: Gamma| x.skewness().unwrap();
        let test = [
            ((1.0, 0.1), 2.0),
            ((1.0, 1.0), 2.0),
            ((10.0, 10.0), 0.6324555320336758663997787),
            ((10.0, 1.0), 0.63245553203367586639977870),
            ((10.0, f64::INFINITY), 0.6324555320336758),
        ];
        for ((s, r), res) in test {
            test_relative(s, r, res, f);
        }
    }

    #[test]
    fn test_mode() {
        let f = |x: Gamma| x.mode().unwrap();
        let test = [((1.0, 0.1), 0.0), ((1.0, 1.0), 0.0)];
        for &((s, r), res) in test.iter() {
            test_absolute(s, r, res, 10e-6, f);
        }
        let test = [
            ((10.0, 10.0), 0.9),
            ((10.0, 1.0), 9.0),
            ((10.0, f64::INFINITY), 0.0),
        ];
        for ((s, r), res) in test {
            test_relative(s, r, res, f);
        }
    }

    #[test]
    fn test_min_max() {
        let f = |x: Gamma| x.min();
        let test = [
            ((1.0, 0.1), 0.0),
            ((1.0, 1.0), 0.0),
            ((10.0, 10.0), 0.0),
            ((10.0, 1.0), 0.0),
            ((10.0, f64::INFINITY), 0.0),
        ];
        for ((s, r), res) in test {
            test_relative(s, r, res, f);
        }
        let f = |x: Gamma| x.max();
        let test = [
            ((1.0, 0.1), f64::INFINITY),
            ((1.0, 1.0), f64::INFINITY),
            ((10.0, 10.0), f64::INFINITY),
            ((10.0, 1.0), f64::INFINITY),
            ((10.0, f64::INFINITY), f64::INFINITY),
        ];
        for ((s, r), res) in test {
            test_relative(s, r, res, f);
        }
    }

    #[test]
    fn test_pdf() {
        let f = |arg: f64| move |x: Gamma| x.pdf(arg);
        let test = [
            ((1.0, 0.1), 1.0, 0.090483741803595961836995),
            ((1.0, 0.1), 10.0, 0.036787944117144234201693),
            ((1.0, 1.0), 1.0, 0.367879441171442321595523),
            ((1.0, 1.0), 10.0, 0.000045399929762484851535),
            ((10.0, 10.0), 1.0, 1.251100357211332989847649),
            ((10.0, 10.0), 10.0, 1.025153212086870580621609e-30),
            ((10.0, 1.0), 1.0, 0.000001013777119630297402),
            ((10.0, 1.0), 10.0, 0.125110035721133298984764),
        ];
        for ((s, r), x, res) in test {
            test_relative(s, r, res, f(x));
        }
        // TODO: test special
        // test_is_nan((10.0, f64::INFINITY), pdf(1.0)); // is this really the behavior we want?
        // TODO: test special
        // (10.0, f64::INFINITY, f64::INFINITY, 0.0, pdf(f64::INFINITY)),];
    }

    #[test]
    fn test_pdf_at_zero() {
        test_relative(1.0, 0.1, 0.1, |x| x.pdf(0.0));
        test_relative(1.0, 0.1, 0.1f64.ln(), |x| x.ln_pdf(0.0));
    }

    #[test]
    fn test_ln_pdf() {
        let f = |arg: f64| move |x: Gamma| x.ln_pdf(arg);
        let test = [
            ((1.0, 0.1), 1.0, -2.40258509299404563405795),
            ((1.0, 0.1), 10.0, -3.30258509299404562850684),
            ((1.0, 1.0), 1.0, -1.0),
            ((1.0, 1.0), 10.0, -10.0),
            ((10.0, 10.0), 1.0, 0.224023449858987228972196),
            ((10.0, 10.0), 10.0, -69.0527107131946016148658),
            ((10.0, 1.0), 1.0, -13.8018274800814696112077),
            ((10.0, 1.0), 10.0, -2.07856164313505845504579),
            ((10.0, f64::INFINITY), f64::INFINITY, f64::NEG_INFINITY),
        ];
        for ((s, r), x, res) in test {
            test_relative(s, r, res, f(x));
        }
        // TODO: test special
        // test_is_nan((10.0, f64::INFINITY), f(1.0)); // is this really the behavior we want?
    }

    #[test]
    fn test_cdf() {
        let f = |arg: f64| move |x: Gamma| x.cdf(arg);
        let test = [
            ((1.0, 0.1), 1.0, 0.095162581964040431858607),
            ((1.0, 0.1), 10.0, 0.632120558828557678404476),
            ((1.0, 1.0), 1.0, 0.632120558828557678404476),
            ((1.0, 1.0), 10.0, 0.999954600070237515148464),
            ((10.0, 10.0), 1.0, 0.542070285528147791685835),
            ((10.0, 10.0), 10.0, 0.999999999999999999999999),
            ((10.0, 1.0), 1.0, 0.000000111425478338720677),
            ((10.0, 1.0), 10.0, 0.542070285528147791685835),
            ((10.0, f64::INFINITY), 1.0, 0.0),
            ((10.0, f64::INFINITY), 10.0, 1.0),
        ];
        for ((s, r), x, res) in test {
            test_relative(s, r, res, f(x));
        }
    }

    #[test]
    fn test_cdf_at_zero() {
        test_relative(1.0, 0.1, 0.0, |x| x.cdf(0.0));
    }

    #[test]
    fn test_cdf_inverse_identity() {
        let f = |p: f64| move |g: Gamma| g.cdf(g.inverse_cdf(p));
        let params = [
            (1.0, 0.1),
            (1.0, 1.0),
            (10.0, 10.0),
            (10.0, 1.0),
            (100.0, 200.0),
        ];

        for (s, r) in params {
            for n in -5..0 {
                let p = 10.0f64.powi(n);
                test_relative(s, r, p, f(p));
            }
        }

        // test case from issue #200
        {
            let x = 20.5567;
            let f = |x: f64| move |g: Gamma| g.inverse_cdf(g.cdf(x));
            test_relative(3.0, 0.5, x, f(x))
        }
    }

    #[test]
    fn test_sf() {
        let f = |arg: f64| move |x: Gamma| x.sf(arg);
        let test = [
            ((1.0, 0.1), 1.0, 0.9048374180359595),
            ((1.0, 0.1), 10.0, 0.3678794411714419),
            ((1.0, 1.0), 1.0, 0.3678794411714419),
            ((1.0, 1.0), 10.0, 4.539992976249074e-5),
            ((10.0, 10.0), 1.0, 0.4579297144718528),
            ((10.0, 10.0), 10.0, 1.1253473960842808e-31),
            ((10.0, 1.0), 1.0, 0.9999998885745217),
            ((10.0, 1.0), 10.0, 0.4579297144718528),
            ((10.0, f64::INFINITY), 1.0, 1.0),
            ((10.0, f64::INFINITY), 10.0, 0.0),
        ];
        for ((s, r), x, res) in test {
            test_relative(s, r, res, f(x));
        }
    }

    #[test]
    fn test_sf_at_zero() {
        test_relative(1.0, 0.1, 1.0, |x| x.sf(0.0));
    }

    #[test]
    fn test_continuous() {
        test::check_continuous_distribution(&create_ok(1.0, 0.5), 0.0, 20.0);
        test::check_continuous_distribution(&create_ok(9.0, 2.0), 0.0, 20.0);
    }
}
