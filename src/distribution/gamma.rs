use crate::distribution::{Continuous, Univariate};
use crate::function::gamma;
use crate::statistics::*;
use crate::{Result, StatsError};
use rand::distributions::Distribution;
use rand::Rng;
use std::f64;

/// Implements the [Gamma](https://en.wikipedia.org/wiki/Gamma_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Gamma, Continuous};
/// use statrs::statistics::Mean;
/// use statrs::prec;
///
/// let n = Gamma::new(3.0, 1.0).unwrap();
/// assert_eq!(n.mean(), 3.0);
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
    /// Returns an error if `shape` or `rate` are `NaN`.
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
        let is_nan = shape.is_nan() || rate.is_nan();
        match (shape, rate, is_nan) {
            (_, _, true) => Err(StatsError::BadParams),
            (_, _, false) if shape <= 0.0 || rate <= 0.0 => Err(StatsError::BadParams),
            (_, _, false) => Ok(Gamma { shape, rate }),
        }
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

impl Distribution<f64> for Gamma {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> f64 {
        sample_unchecked(r, self.shape, self.rate)
    }
}

impl Univariate<f64, f64> for Gamma {
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
        f64::INFINITY
    }
}

impl Mean<f64> for Gamma {
    /// Returns the mean of the gamma distribution
    ///
    /// # Remarks
    ///
    /// Returns `shape` if `rate == f64::INFINITY`. This behavior
    /// is borrowed from the Math.NET implementation
    ///
    /// # Formula
    ///
    /// ```ignore
    /// α / β
    /// ```
    ///
    /// where `α` is the shape and `β` is the rate
    fn mean(&self) -> f64 {
        self.shape / self.rate
    }
}

impl Variance<f64> for Gamma {
    /// Returns the variance of the gamma distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// α / β^2
    /// ```
    ///
    /// where `α` is the shape and `β` is the rate
    fn variance(&self) -> f64 {
        self.shape / (self.rate * self.rate)
    }

    /// Returns the standard deviation of the gamma distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// sqrt(α) / β
    /// ```
    ///
    /// where `α` is the shape and `β` is the rate
    fn std_dev(&self) -> f64 {
        self.shape.sqrt() / self.rate
    }
}

impl Entropy<f64> for Gamma {
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
    fn entropy(&self) -> f64 {
        self.shape - self.rate.ln()
            + gamma::ln_gamma(self.shape)
            + (1.0 - self.shape) * gamma::digamma(self.shape)
    }
}

impl Skewness<f64> for Gamma {
    /// Returns the skewness of the gamma distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 2 / sqrt(α)
    /// ```
    ///
    /// where `α` is the shape
    fn skewness(&self) -> f64 {
        2.0 / self.shape.sqrt()
    }
}

impl Mode<f64> for Gamma {
    /// Returns the mode for the gamma distribution
    ///
    /// # Remarks
    ///
    /// Returns `shape` if `rate == f64::INFINITY`. This behavior
    /// is borrowed from the Math.NET implementation
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (α - 1) / β
    /// ```
    ///
    /// where `α` is the shape and `β` is the rate
    fn mode(&self) -> f64 {
        (self.shape - 1.0) / self.rate
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
/// rate of `rate` using `r` as the source of randomness. Implementation from:
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
        if u < 1.0 - 0.0331 * x * x || u.ln() < 0.5 * x + d * (1.0 - v - v.ln()) {
            return afix * d * v / rate;
        }
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod test {
    use std::f64;
    use crate::statistics::*;
    use crate::distribution::{Univariate, Continuous, Gamma};
    use crate::distribution::internal::*;

    fn try_create(shape: f64, rate: f64) -> Gamma {
        let n = Gamma::new(shape, rate);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(shape: f64, rate: f64) {
        let n = try_create(shape, rate);
        assert_eq!(shape, n.shape());
        assert_eq!(rate, n.rate());
    }

    fn bad_create_case(shape: f64, rate: f64) {
        let n = Gamma::new(shape, rate);
        assert!(n.is_err());
    }

    fn get_value<F>(shape: f64, rate: f64, eval: F) -> f64
        where F: Fn(Gamma) -> f64
    {
        let n = try_create(shape, rate);
        eval(n)
    }

    fn test_case<F>(shape: f64, rate: f64, expected: f64, eval: F)
        where F: Fn(Gamma) -> f64
    {
        let x = get_value(shape, rate, eval);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(shape: f64, rate: f64, expected: f64, acc: f64, eval: F)
        where F: Fn(Gamma) -> f64
    {
        let x = get_value(shape, rate, eval);
        assert_almost_eq!(expected, x, acc);
    }

    fn test_is_nan<F>(shape: f64, rate: f64, eval: F)
        where F: Fn(Gamma) -> f64
    {
        let x = get_value(shape, rate, eval);
        assert!(x.is_nan());
    }

    #[test]
    fn test_create() {
        create_case(1.0, 0.1);
        create_case(1.0, 1.0);
        create_case(10.0, 10.0);
        create_case(10.0, 1.0);
        create_case(10.0, f64::INFINITY);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(0.0, 0.0);
        bad_create_case(1.0, f64::NAN);
        bad_create_case(1.0, -1.0);
        bad_create_case(-1.0, 1.0);
        bad_create_case(-1.0, -1.0);
        bad_create_case(-1.0, f64::NAN);
    }

    #[test]
    fn test_mean() {
        test_case(1.0, 0.1, 10.0, |x| x.mean());
        test_case(1.0, 1.0, 1.0, |x| x.mean());
        test_case(10.0, 10.0, 1.0, |x| x.mean());
        test_case(10.0, 1.0, 10.0, |x| x.mean());
        test_case(10.0, f64::INFINITY, 0.0, |x| x.mean());
    }

    #[test]
    fn test_variance() {
        test_almost(1.0, 0.1, 100.0, 1e-13, |x| x.variance());
        test_case(1.0, 1.0, 1.0, |x| x.variance());
        test_case(10.0, 10.0, 0.1, |x| x.variance());
        test_case(10.0, 1.0, 10.0, |x| x.variance());
        test_case(10.0, f64::INFINITY, 0.0, |x| x.variance());
    }

    #[test]
    fn test_std_dev() {
        test_case(1.0, 0.1, 10.0, |x| x.std_dev());
        test_case(1.0, 1.0, 1.0, |x| x.std_dev());
        test_case(10.0, 10.0, 0.31622776601683794197697302588502426416723164097476643, |x| x.std_dev());
        test_case(10.0, 1.0, 3.1622776601683793319988935444327185337195551393252168, |x| x.std_dev());
        test_case(10.0, f64::INFINITY, 0.0, |x| x.std_dev());
    }

    #[test]
    fn test_entropy() {
        test_almost(1.0, 0.1, 3.3025850929940456285068402234265387271634735938763824, 1e-15, |x| x.entropy());
        test_almost(1.0, 1.0, 1.0, 1e-15, |x| x.entropy());
        test_almost(10.0, 10.0, 0.23346908548693395836262094490967812177376750477943892, 1e-13, |x| x.entropy());
        test_almost(10.0, 1.0, 2.5360541784809796423806123995940423293748689934081866, 1e-13, |x| x.entropy());
        test_case(10.0, f64::INFINITY, f64::NEG_INFINITY, |x| x.entropy());
    }

    #[test]
    fn test_skewness() {
        test_case(1.0, 0.1, 2.0, |x| x.skewness());
        test_case(1.0, 1.0, 2.0, |x| x.skewness());
        test_case(10.0, 10.0, 0.63245553203367586639977870888654370674391102786504337, |x| x.skewness());
        test_case(10.0, 1.0, 0.63245553203367586639977870888654370674391102786504337, |x| x.skewness());
        test_case(10.0, f64::INFINITY, 0.63245553203367586639977870888654370674391102786504337, |x| x.skewness());
    }

    #[test]
    fn test_mode() {
        test_case(1.0, 0.1, 0.0, |x| x.mode());
        test_case(1.0, 1.0, 0.0, |x| x.mode());
        test_case(10.0, 10.0, 0.9, |x| x.mode());
        test_case(10.0, 1.0, 9.0, |x| x.mode());
        test_case(10.0, f64::INFINITY, 0.0, |x| x.mode());
    }

    #[test]
    fn test_min_max() {
        test_case(1.0, 0.1, 0.0, |x| x.min());
        test_case(1.0, 1.0, 0.0, |x| x.min());
        test_case(10.0, 10.0, 0.0, |x| x.min());
        test_case(10.0, 1.0, 0.0, |x| x.min());
        test_case(10.0, f64::INFINITY, 0.0, |x| x.min());
        test_case(1.0, 0.1, f64::INFINITY, |x| x.max());
        test_case(1.0, 1.0, f64::INFINITY, |x| x.max());
        test_case(10.0, 10.0, f64::INFINITY, |x| x.max());
        test_case(10.0, 1.0, f64::INFINITY, |x| x.max());
        test_case(10.0, f64::INFINITY, f64::INFINITY, |x| x.max());
    }

    #[test]
    fn test_pdf() {
        test_case(1.0, 0.1, 0.090483741803595961836995913651194571475319347018875963, |x| x.pdf(1.0));
        test_case(1.0, 0.1, 0.036787944117144234201693506390001264039984687455876246, |x| x.pdf(10.0));
        test_case(1.0, 1.0, 0.36787944117144232159552377016146086744581113103176804, |x| x.pdf(1.0));
        test_case(1.0, 1.0, 0.000045399929762484851535591515560550610237918088866564953, |x| x.pdf(10.0));
        test_almost(10.0, 10.0, 1.2511003572113329898476497894772544708420990097708588, 1e-14, |x| x.pdf(1.0));
        test_almost(10.0, 10.0, 1.0251532120868705806216092933926141802686541811003037e-30, 1e-44, |x| x.pdf(10.0));
        test_almost(10.0, 1.0, 0.0000010137771196302974029859010421116095333052555418644397, 1e-20, |x| x.pdf(1.0));
        test_almost(10.0, 1.0, 0.12511003572113329898476497894772544708420990097708601, 1e-15, |x| x.pdf(10.0));
        test_is_nan(10.0, f64::INFINITY, |x| x.pdf(1.0)); // is this really the behavior we want?
        test_case(10.0, f64::INFINITY, 0.0, |x| x.pdf(f64::INFINITY));
    }

    #[test]
    fn test_pdf_at_zero() {
        test_almost(1.0, 0.1, 0.1, 1e-10, |x| x.pdf(0.0));
        test_almost(1.0, 0.1, 0.1f64.ln(), 1e-10, |x| x.ln_pdf(0.0));
    }

    #[test]
    fn test_ln_pdf() {
        test_case(1.0, 0.1, -2.402585092994045634057955346552321429281631934330484, |x| x.ln_pdf(1.0));
        test_case(1.0, 0.1, -3.3025850929940456285068402234265387271634735938763824, |x| x.ln_pdf(10.0));
        test_case(1.0, 1.0, -1.0, |x| x.ln_pdf(1.0));
        test_case(1.0, 1.0, -10.0, |x| x.ln_pdf(10.0));
        test_almost(10.0, 10.0, 0.22402344985898722897219667227693591172986563062456522, 1e-15, |x| x.ln_pdf(1.0));
        test_case(10.0, 10.0, -69.052710713194601614865880235563786219860220971716511, |x| x.ln_pdf(10.0));
        test_almost(10.0, 1.0, -13.801827480081469611207717874566706164281149255663166, 1e-14, |x| x.ln_pdf(1.0));
        test_almost(10.0, 1.0,  -2.0785616431350584550457947824074282958712358580042068, 1e-14, |x| x.ln_pdf(10.0));
        test_is_nan(10.0, f64::INFINITY, |x| x.ln_pdf(1.0)); // is this really the behavior we want?
        test_case(10.0, f64::INFINITY, f64::NEG_INFINITY, |x| x.ln_pdf(f64::INFINITY));
    }

    #[test]
    fn test_cdf() {
        test_almost(1.0, 0.1, 0.095162581964040431858607615783064404690935346242622848, 1e-16, |x| x.cdf(1.0));
        test_almost(1.0, 0.1, 0.63212055882855767840447622983853913255418886896823196, 1e-15, |x| x.cdf(10.0));
        test_almost(1.0, 1.0, 0.63212055882855767840447622983853913255418886896823196, 1e-15, |x| x.cdf(1.0));
        test_case(1.0, 1.0, 0.99995460007023751514846440848443944938976208191113396,|x| x.cdf(10.0));
        test_almost(10.0, 10.0, 0.54207028552814779168583514294066541824736464003242184, 1e-15, |x| x.cdf(1.0));
        test_case(10.0, 10.0, 0.99999999999999999999999999999988746526039157266114706, |x| x.cdf(10.0));
        test_almost(10.0, 1.0, 0.00000011142547833872067735305068724025236288094949815466035, 1e-21, |x| x.cdf(1.0));
        test_almost(10.0, 1.0, 0.54207028552814779168583514294066541824736464003242184, 1e-15, |x| x.cdf(10.0));
        test_case(10.0, f64::INFINITY, 0.0, |x| x.cdf(1.0));
        test_case(10.0, f64::INFINITY, 1.0, |x| x.cdf(10.0));
    }

    #[test]
    fn test_cdf_at_zero() {
        test_case(1.0, 0.1, 0.0, |x| x.cdf(0.0));
    }

    #[test]
    fn test_continuous() {
        test::check_continuous_distribution(&try_create(1.0, 0.5), 0.0, 20.0);
        test::check_continuous_distribution(&try_create(9.0, 2.0), 0.0, 20.0);
    }
}
