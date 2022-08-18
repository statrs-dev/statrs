use crate::distribution::{Continuous, ContinuousCDF};
use crate::function::gamma;
use crate::statistics::*;
use crate::{Result, StatsError};
use core::f64::INFINITY as INF;
use rand::Rng;

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
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Gamma {
    shape: f64,
    rate: f64,
}

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
    pub fn new(shape: f64, rate: f64) -> Result<Gamma> {
        if shape.is_nan()
            || rate.is_nan()
            || shape.is_infinite() && rate.is_infinite()
            || shape <= 0.0
            || rate <= 0.0
        {
            return Err(StatsError::BadParams);
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

impl ::rand::distributions::Distribution<f64> for Gamma {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
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
    /// ```ignore
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
    /// ```ignore
    /// (1 / Γ(α)) * γ(α, β * x)
    /// ```
    ///
    /// where `α` is the shape, `β` is the rate, `Γ` is the gamma function,
    /// and `γ` is the upper incomplete gamma function
    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            1.0
        }
        else if ulps_eq!(x, self.shape) && self.rate.is_infinite() {
            0.0
        }
        else if self.rate.is_infinite() {
            1.0
        }
        else if x.is_infinite() {
            0.0
        }
        else {
            gamma::gamma_ur(self.shape, x * self.rate)
        }
    }
}

impl Min<f64> for Gamma {
    /// Returns the minimum value in the domain of the
    /// gamma distribution representable by a double precision
    /// float
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

impl Max<f64> for Gamma {
    /// Returns the maximum value in the domain of the
    /// gamma distribution representable by a double precision
    /// float
    ///
    /// # Formula
    ///
    /// ```ignore
    /// INF
    /// ```
    fn max(&self) -> f64 {
        INF
    }
}

impl Distribution<f64> for Gamma {
    /// Returns the mean of the gamma distribution
    ///
    /// # Formula
    ///
    /// ```ignore
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
    /// ```ignore
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
    /// ```ignore
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
    /// ```ignore
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
    /// ```ignore
    /// (α - 1) / β
    /// ```
    ///
    /// where `α` is the shape and `β` is the rate
    fn mode(&self) -> Option<f64> {
        Some((self.shape - 1.0) / self.rate)
    }
}

impl Continuous<f64, f64> for Gamma {
    /// Calculates the probability density function for the gamma distribution
    /// at `x`
    ///
    /// # Remarks
    ///
    /// Returns `NAN` if any of `shape` or `rate` are `INF`
    /// or if `x` is `INF`
    ///
    /// # Formula
    ///
    /// ```ignore
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
    /// Returns `NAN` if any of `shape` or `rate` are `INF`
    /// or if `x` is `INF`
    ///
    /// # Formula
    ///
    /// ```ignore
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
/// <br />
/// <div>
/// <i>"A Simple Method for Generating Gamma Variables"</i> - Marsaglia & Tsang
/// </div>
/// <div>
/// ACM Transactions on Mathematical Software, Vol. 26, No. 3, September 2000,
/// Pages 363-372
/// </div>
/// <br />
pub fn sample_unchecked<R: Rng + ?Sized>(rng: &mut R, shape: f64, rate: f64) -> f64 {
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

        v *= v * v;
        x *= x;
        let u: f64 = rng.gen();
        if u < 1.0 - 0.0331 * x * x || u.ln() < 0.5 * x + d * (1.0 - v + v.ln()) {
            return afix * d * v / rate;
        }
    }
}

#[cfg(all(test, feature = "nightly"))]
mod tests {
    use super::*;
    use crate::consts::ACC;
    use crate::distribution::internal::*;
    use crate::testing_boiler;

    testing_boiler!((f64, f64), Gamma);

    #[test]
    fn test_create() {
        let valid = [
            (1.0, 0.1),
            (1.0, 1.0),
            (10.0, 10.0),
            (10.0, 1.0),
            (10.0, INF),
        ];

        for &arg in valid.iter() {
            try_create(arg);
        }
    }

    #[test]
    fn test_bad_create() {
        let invalid = [
            (0.0, 0.0),
            (1.0, f64::NAN),
            (1.0, -1.0),
            (-1.0, 1.0),
            (-1.0, -1.0),
            (-1.0, f64::NAN),
        ];
        for &arg in invalid.iter() {
            bad_create_case(arg);
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
            ((10.0, INF), 0.0),
        ];
        for &(arg, res) in test.iter() {
            test_case(arg, res, f);
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
            ((10.0, INF), 0.0),
        ];
        for &(arg, res) in test.iter() {
            test_case(arg, res, f);
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
            ((10.0, INF), f64::NEG_INFINITY),
        ];
        for &(arg, res) in test.iter() {
            test_case(arg, res, f);
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
            ((10.0, INF), 0.6324555320336758),
        ];
        for &(arg, res) in test.iter() {
            test_case(arg, res, f);
        }
    }

    #[test]
    fn test_mode() {
        let f = |x: Gamma| x.mode().unwrap();
        let test = [((1.0, 0.1), 0.0), ((1.0, 1.0), 0.0)];
        for &(arg, res) in test.iter() {
            test_case_special(arg, res, 10e-6, f);
        }
        let test = [((10.0, 10.0), 0.9), ((10.0, 1.0), 9.0), ((10.0, INF), 0.0)];
        for &(arg, res) in test.iter() {
            test_case(arg, res, f);
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
            ((10.0, INF), 0.0),
        ];
        for &(arg, res) in test.iter() {
            test_case(arg, res, f);
        }
        let f = |x: Gamma| x.max();
        let test = [
            ((1.0, 0.1), INF),
            ((1.0, 1.0), INF),
            ((10.0, 10.0), INF),
            ((10.0, 1.0), INF),
            ((10.0, INF), INF),
        ];
        for &(arg, res) in test.iter() {
            test_case(arg, res, f);
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
        for &(arg, x, res) in test.iter() {
            test_case(arg, res, f(x));
        }
        //TODO: test special
        // test_is_nan((10.0, INF), pdf(1.0)); // is this really the behavior we want?
        //TODO: test special
        // (10.0, INF, INF, 0.0, pdf(INF)),];
    }

    #[test]
    fn test_pdf_at_zero() {
        test_case((1.0, 0.1), 0.1, |x| x.pdf(0.0));
        test_case((1.0, 0.1), 0.1f64.ln(), |x| x.ln_pdf(0.0));
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
            ((10.0, INF), INF, f64::NEG_INFINITY),
        ];
        for &(arg, x, res) in test.iter() {
            test_case(arg, res, f(x));
        }
        // TODO: test special
        // test_is_nan((10.0, INF), f(1.0)); // is this really the behavior we want?
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
            ((10.0, INF), 1.0, 0.0),
            ((10.0, INF), 10.0, 1.0),
        ];
        for &(arg, x, res) in test.iter() {
            test_case(arg, res, f(x));
        }
    }

    #[test]
    fn test_cdf_at_zero() {
        test_case((1.0, 0.1), 0.0, |x| x.cdf(0.0));
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
            ((10.0, INF), 1.0, 1.0),
            ((10.0, INF), 10.0, 0.0),
        ];
        for &(arg, x, res) in test.iter() {
            test_case(arg, res, f(x));
        }
    }

    #[test]
    fn test_sf_at_zero() {
        test_case((1.0, 0.1), 1.0, |x| x.sf(0.0));
    }

    #[test]
    fn test_continuous() {
        test::check_continuous_distribution(&try_create((1.0, 0.5)), 0.0, 20.0);
        test::check_continuous_distribution(&try_create((9.0, 2.0)), 0.0, 20.0);
    }
}
