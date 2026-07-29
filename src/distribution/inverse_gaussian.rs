use crate::distribution::{Continuous, ContinuousCDF};
use crate::function::erf;
use crate::statistics::*;
use core::f64;
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

/// Implements the
/// [Inverse Gaussian](https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution)
/// (Wald) distribution, parameterized by its mean `mu` and shape `lambda`.
///
/// # Examples
///
/// ```
/// use statrs::distribution::{InverseGaussian, Continuous};
/// use statrs::statistics::Distribution;
///
/// let n = InverseGaussian::new(1.0, 1.0).unwrap();
/// assert_eq!(n.mean().unwrap(), 1.0);
/// assert!((n.pdf(1.0) - 0.3989422804014327).abs() < 1e-15);
/// ```
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct InverseGaussian {
    mu: f64,
    lambda: f64,
}

/// Represents the errors that can occur when creating an [`InverseGaussian`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum InverseGaussianError {
    /// The mean is NaN, zero, negative, or infinite.
    MuInvalid,

    /// The shape is NaN, zero, negative, or infinite.
    LambdaInvalid,
}

impl core::fmt::Display for InverseGaussianError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match self {
            InverseGaussianError::MuInvalid => {
                write!(f, "Mean is NaN, zero, negative, or infinite")
            }
            InverseGaussianError::LambdaInvalid => {
                write!(f, "Shape is NaN, zero, negative, or infinite")
            }
        }
    }
}

impl core::error::Error for InverseGaussianError {}

impl InverseGaussian {
    /// Constructs a new inverse Gaussian distribution with mean `mu` and
    /// shape `lambda`.
    ///
    /// Both parameters must be finite and positive; infinite parameters are
    /// rejected rather than admitted as degenerate limits.
    ///
    /// # Errors
    ///
    /// Returns an error if `mu` or `lambda` is NaN, zero, negative, or
    /// infinite.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::InverseGaussian;
    ///
    /// let mut result = InverseGaussian::new(1.0, 1.0);
    /// assert!(result.is_ok());
    ///
    /// result = InverseGaussian::new(0.0, 1.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(mu: f64, lambda: f64) -> Result<InverseGaussian, InverseGaussianError> {
        if !(mu.is_finite() && mu > 0.0) {
            return Err(InverseGaussianError::MuInvalid);
        }
        if !(lambda.is_finite() && lambda > 0.0) {
            return Err(InverseGaussianError::LambdaInvalid);
        }
        Ok(InverseGaussian { mu, lambda })
    }

    /// Returns the mean `mu` of the inverse Gaussian distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::InverseGaussian;
    ///
    /// let n = InverseGaussian::new(1.0, 2.0).unwrap();
    /// assert_eq!(n.mu(), 1.0);
    /// ```
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Returns the shape `lambda` of the inverse Gaussian distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::InverseGaussian;
    ///
    /// let n = InverseGaussian::new(1.0, 2.0).unwrap();
    /// assert_eq!(n.lambda(), 2.0);
    /// ```
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// The same two arguments as `erfc` takes rather than `Phi`:
    /// `u_minus = sqrt(lambda / 2x) (x/mu - 1)` and
    /// `u_plus = sqrt(lambda / 2x) (x/mu + 1)`, so that
    /// `Phi(a) = erfc(-u_minus) / 2` and `Phi(b) = erfc(u_plus) / 2`.
    ///
    /// These are the natural variables for the tails, because
    /// `2 lambda / mu - u_plus^2 == -u_minus^2` identically - the exponential
    /// prefactor of the second cdf term is exactly the difference of the two
    /// squares.
    fn erfc_args(&self, x: f64) -> (f64, f64) {
        let s = (self.lambda / (2.0 * x)).sqrt();
        let t = x / self.mu;
        (s * (t - 1.0), s * (t + 1.0))
    }
}

/// `exp(-u^2)` with the rounding error of the square folded back in, so the
/// squaring contributes nothing on top of the exponential's own inherent
/// amplification.
fn exp_neg_square(u: f64) -> f64 {
    let sq = u * u;
    let err = crate::prec::dekker_product_err(u, u, sq);
    (-sq).exp() * (1.0 - err)
}

/// `-u^2` as an unevaluated sum `(hi, lo)`, exact to twice working precision.
///
/// Returned as a pair rather than summed. `lo` is at most half an ulp of `hi`,
/// so `hi - lo` rounds straight back to `hi` and discards it; it survives only
/// if folded into the final total, which [`add_log_term`] does.
fn neg_square_parts(u: f64) -> (f64, f64) {
    let sq = u * u;
    (-sq, -crate::prec::dekker_product_err(u, u, sq))
}

/// `hi + lo + log_term`, keeping `lo` in play across the addition so it can
/// still tip the final rounding.
fn add_log_term((hi, lo): (f64, f64), log_term: f64) -> f64 {
    let (s, e) = crate::prec::two_sum(hi, log_term);
    s + (e + lo)
}

impl core::fmt::Display for InverseGaussian {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "IG({},{})", self.mu, self.lambda)
    }
}

/// The smaller root of the Michael-Schucany-Haas quadratic for a standard
/// normal draw `z`.
///
/// Kept free of the RNG so the boundary cases are testable: `z` really can be
/// exactly zero, since the ziggurat draws `u = 2f - 1` from a 53-bit `f` and
/// `f == 0.5` is one of the values it can take.
///
/// The textbook root `mu + (mu / 2 lambda) (mnu - sqrt(mnu (4 lambda + mnu)))`
/// cancels catastrophically for `mnu >> lambda`. Rationalizing removes the
/// subtraction, but the obvious rationalized form divides by
/// `mnu + sqrt(mnu (4 lambda + mnu))`, which is `0/0` at `z == 0`. Pulling
/// `sqrt(mnu)` out of the radical cancels it against the numerator and leaves a
/// denominator bounded below by `2 sqrt(lambda)`, so the expression is finite
/// throughout and tends to `mu` as it should.
#[cfg(feature = "rand")]
fn msh_smaller_root(mu: f64, lambda: f64, z: f64) -> f64 {
    let mnu = mu * z * z;
    let s = mnu.sqrt();
    mu - 2.0 * mu * s / (s + (4.0 * lambda + mnu).sqrt())
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl ::rand::distr::Distribution<f64> for InverseGaussian {
    /// Samples by the transformation method of Michael, Schucany & Haas
    /// (1976): a chi-square(1) variate is mapped to the smaller root of the
    /// quadratic it satisfies, then that root or its conjugate `mu^2 / y` is
    /// selected with the appropriate probability.
    fn sample<R: ::rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let z = crate::distribution::ziggurat::sample_std_normal(rng);
        let y = msh_smaller_root(self.mu, self.lambda, z);
        let u: f64 = ::rand::RngExt::random(rng);
        if u <= self.mu / (self.mu + y) {
            y
        } else {
            self.mu * self.mu / y
        }
    }
}

impl ContinuousCDF<f64, f64> for InverseGaussian {
    /// Calculates the cumulative distribution function for the inverse
    /// Gaussian distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// Phi(sqrt(λ/x) (x/μ - 1)) + e^(2λ/μ) Phi(-sqrt(λ/x) (x/μ + 1))
    /// ```
    ///
    /// where `Phi` is the standard normal cdf. Evaluated in the equivalent
    /// scaled form below, which never forms `e^(2λ/μ)` - that overflows for
    /// `λ/μ > ~355` even though the product it appears in is a probability.
    ///
    /// # Remarks
    ///
    /// Measured against mpmath at 60 digits over 1800 points (six parameter
    /// sets, ten decades each): 0.11 ulp median, 1179 ulp worst case. The worst
    /// case is in the far left tail, where relative accuracy degrades like
    /// `|ln cdf| * eps` - inherent to returning a linear-domain probability
    /// that small, since the exponent itself is not representable more finely
    /// than that. Use [`ContinuousCDF::ln_cdf`] for tail work.
    fn cdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x <= 0.0 {
            return 0.0;
        }
        if x.is_infinite() {
            return 1.0;
        }
        let (um, up) = self.erfc_args(x);
        if um <= 0.0 {
            self.cdf_scaled(um, up).min(1.0)
        } else {
            (1.0 - self.sf_scaled(um, up, x)).clamp(0.0, 1.0)
        }
    }

    /// Calculates the survival function for the inverse Gaussian distribution
    /// at `x`
    ///
    /// # Remarks
    ///
    /// In the far right tail the two terms of the cdf formula approach each
    /// other - their ratio is `(x/μ - 1)/(x/μ + 1)` - so relative accuracy
    /// degrades by roughly a factor `x / 2μ` on top of the `|ln sf| * eps`
    /// amplification inherent to a linear-domain result. Measured over the same
    /// 1800 points: 0.12 ulp median, 78686 ulp (about `9e-12` relative) worst
    /// case. Use [`ContinuousCDF::ln_sf`] for tail work.
    fn sf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x <= 0.0 {
            return 1.0;
        }
        if x.is_infinite() {
            return 0.0;
        }
        let (um, up) = self.erfc_args(x);
        if um > 0.0 {
            self.sf_scaled(um, up, x).max(0.0)
        } else {
            (1.0 - self.cdf_scaled(um, up)).clamp(0.0, 1.0)
        }
    }

    /// Tail-accurate log of the cdf, finite far past the point where `cdf`
    /// itself underflows to zero.
    ///
    /// # Remarks
    ///
    /// The absolute error of a log is the relative error of the probability it
    /// represents, so that is the metric quoted here. Wherever the cdf is a
    /// representable nonzero `f64`, the measured absolute error is `1.1e-23`
    /// median and `1.2e-13` worst case - i.e. the implied probability is good
    /// to at least 12 significant digits, against the 1179 ulp that [`Self::cdf`]
    /// can lose there.
    ///
    /// Past that point the log itself is the limit: at `ln cdf = -3.4e8` one
    /// ulp *is* `6e-8`, and the measured error stays within 1-2 ulp of the
    /// returned value.
    fn ln_cdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if x.is_infinite() {
            return 0.0;
        }
        let (um, up) = self.erfc_args(x);
        if um <= 0.0 {
            // both scaled terms are positive here, so this sum never cancels
            add_log_term(
                neg_square_parts(um),
                (0.5 * (erf::erfcx(-um) + erf::erfcx(up))).ln(),
            )
        } else {
            (-self.sf_scaled(um, up, x)).ln_1p()
        }
    }

    /// Tail-accurate log of the survival function; see [`Self::ln_cdf`] for the
    /// error metric. Wherever `sf` is a representable nonzero `f64` the measured
    /// absolute error is `9.4e-20` median and `1.0e-11` worst case, against the
    /// 78686 ulp [`Self::sf`] can lose in the same region; beyond it the error
    /// stays within 1-2 ulp of the returned log.
    fn ln_sf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x <= 0.0 {
            return 0.0;
        }
        if x.is_infinite() {
            return f64::NEG_INFINITY;
        }
        let (um, up) = self.erfc_args(x);
        if um > 0.0 {
            add_log_term(neg_square_parts(um), (0.5 * self.erfcx_gap(um, up, x)).ln())
        } else {
            (-self.cdf_scaled(um, up)).ln_1p()
        }
    }
}

impl InverseGaussian {
    /// `cdf(x)` for `u_minus <= 0` (i.e. `x <= μ`), where the cdf is the
    /// smaller of the two tails.
    ///
    /// Writing `Phi(a) = erfc(-u_minus)/2` and
    /// `e^(2λ/μ) Phi(b) = e^(-u_minus^2) erfcx(u_plus) / 2` - the latter using
    /// `2λ/μ - u_plus^2 == -u_minus^2` - gives
    /// `cdf = e^(-u_minus^2) [erfcx(-u_minus) + erfcx(u_plus)] / 2`. Both
    /// `erfcx` terms are `O(1/u)` and positive, so nothing large is ever
    /// formed and nothing cancels.
    fn cdf_scaled(&self, um: f64, up: f64) -> f64 {
        let _ = self;
        0.5 * exp_neg_square(um) * (erf::erfcx(-um) + erf::erfcx(up))
    }

    /// `sf(x)` for `u_minus > 0` (i.e. `x > μ`), the mirror of
    /// [`Self::cdf_scaled`]:
    /// `sf = e^(-u_minus^2) [erfcx(u_minus) - erfcx(u_plus)] / 2`.
    fn sf_scaled(&self, um: f64, up: f64, x: f64) -> f64 {
        0.5 * exp_neg_square(um) * self.erfcx_gap(um, up, x)
    }

    /// `erfcx(u_minus) - erfcx(u_plus)`, which is positive because `erfcx` is
    /// decreasing and `u_plus > u_minus`.
    ///
    /// The two values converge as `x / μ` grows, so for large enough `x` the
    /// difference underflows to zero or below. There the limiting ratio is
    /// `u_minus / u_plus = (x/μ - 1)/(x/μ + 1)`, so the gap tends to
    /// `erfcx(u_minus) * 2 / (x/μ + 1)`.
    fn erfcx_gap(&self, um: f64, up: f64, x: f64) -> f64 {
        let lo = erf::erfcx(um);
        let gap = lo - erf::erfcx(up);
        if gap > 0.0 {
            gap
        } else {
            lo * 2.0 / (x / self.mu + 1.0)
        }
    }
}

impl Min<f64> for InverseGaussian {
    /// Returns the minimum value in the domain of the inverse Gaussian
    /// distribution
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

impl Max<f64> for InverseGaussian {
    /// Returns the maximum value in the domain of the inverse Gaussian
    /// distribution
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

impl Distribution<f64> for InverseGaussian {
    /// Returns the mean of the inverse Gaussian distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// μ
    /// ```
    fn mean(&self) -> Option<f64> {
        Some(self.mu)
    }

    /// Returns the variance of the inverse Gaussian distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// μ^3 / λ
    /// ```
    fn variance(&self) -> Option<f64> {
        Some(self.mu * self.mu * self.mu / self.lambda)
    }

    /// Returns the skewness of the inverse Gaussian distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// 3 sqrt(μ / λ)
    /// ```
    fn skewness(&self) -> Option<f64> {
        Some(3.0 * (self.mu / self.lambda).sqrt())
    }
}

impl Mode<Option<f64>> for InverseGaussian {
    /// Returns the mode of the inverse Gaussian distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// μ [sqrt(1 + 9μ^2 / (4λ^2)) - 3μ / (2λ)]
    /// ```
    ///
    /// evaluated in the equivalent form `λ / (sqrt((λ/μ)^2 + 9/4) + 3/2)`.
    ///
    /// # Remarks
    ///
    /// The bracket above is a difference of two terms that converge as `μ/λ`
    /// grows, so it cancels: at `μ = 1e8, λ = 1` both are `1.5e8` to working
    /// precision and it evaluates to `0` rather than `1/3`. Multiplying through
    /// by the conjugate removes the subtraction entirely, and the reciprocal
    /// form also keeps the squared term from overflowing.
    fn mode(&self) -> Option<f64> {
        let r = self.lambda / self.mu;
        Some(self.lambda / ((r * r + 2.25).sqrt() + 1.5))
    }
}

impl Continuous<f64, f64> for InverseGaussian {
    /// Calculates the probability density function for the inverse Gaussian
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// sqrt(λ / (2π x^3)) e^(-λ (x - μ)^2 / (2 μ^2 x))
    /// ```
    ///
    /// # Remarks
    ///
    /// Measured at 2.54 ulp median, 583 ulp worst case. The worst case is in the
    /// tails, and is a property of `exp`, not of this expression: a relative
    /// error `eps` in the exponent becomes a relative error `|exponent| * eps`
    /// in the result, and once the exponent reaches `-10^3` no `f64` evaluation
    /// can do better. [`Self::ln_pdf`] returns that exponent directly and is
    /// accurate to a few ulp throughout.
    fn pdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x <= 0.0 || x.is_infinite() {
            return 0.0;
        }
        let d = x - self.mu;
        (self.lambda / (2.0 * f64::consts::PI * x * x * x)).sqrt()
            * (-self.lambda * d * d / (2.0 * self.mu * self.mu * x)).exp()
    }

    /// Calculates the log probability density function for the inverse
    /// Gaussian distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// (1/2) [ln λ - ln(2π) - 3 ln x] - λ (x - μ)^2 / (2 μ^2 x)
    /// ```
    fn ln_pdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x <= 0.0 || x.is_infinite() {
            return f64::NEG_INFINITY;
        }
        let d = x - self.mu;
        0.5 * (self.lambda.ln() - (2.0 * f64::consts::PI).ln() - 3.0 * x.ln())
            - self.lambda * d * d / (2.0 * self.mu * self.mu * x)
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::internal::density_util;
    use crate::prec;

    crate::distribution::internal::testing_boiler!(mu: f64, lambda: f64; InverseGaussian; InverseGaussianError);

    #[test]
    fn test_create() {
        create_ok(1.0, 1.0);
        create_ok(0.1, 10.0);
        create_ok(1e5, 1e-5);
    }

    #[test]
    fn test_bad_create() {
        test_create_err(0.0, 1.0, InverseGaussianError::MuInvalid);
        test_create_err(1.0, 0.0, InverseGaussianError::LambdaInvalid);
        create_err(-1.0, 1.0);
        create_err(1.0, -1.0);
        create_err(f64::NAN, 1.0);
        create_err(1.0, f64::NAN);
        create_err(f64::INFINITY, 1.0);
        create_err(1.0, f64::INFINITY);
    }

    #[test]
    fn test_moments() {
        let mean = |x: InverseGaussian| x.mean().unwrap();
        let variance = |x: InverseGaussian| x.variance().unwrap();
        let skewness = |x: InverseGaussian| x.skewness().unwrap();
        test_exact(2.0, 3.0, 2.0, mean);
        test_exact(2.0, 3.0, 8.0 / 3.0, variance);
        test_absolute(2.0, 3.0, 2.449489742783178098197, 1e-15, skewness);
        // entropy has no elementary closed form and stays None
        assert!(create_ok(1.0, 1.0).entropy().is_none());
    }

    #[test]
    fn test_mode() {
        // mpmath at 50 significant digits
        let mode = |x: InverseGaussian| x.mode().unwrap();
        test_absolute(1.0, 1.0, 0.3027756377319946465596, 1e-15, mode);
        test_absolute(2.0, 3.0, 0.8284271247461900976034, 1e-15, mode);
    }

    /// The textbook bracket `sqrt(1 + 9u^2/4L^2) - 3u/2L` differences two terms
    /// that converge as `mu/lambda` grows (statrs-dev/statrs#423). References
    /// are mpmath, but computed from the conjugate form: evaluating the
    /// textbook one at `mu = 1e150` needs about 400 digits to get an answer at
    /// all, and returns zero at 50.
    #[test]
    fn test_mode_does_not_cancel() {
        let mode = |x: InverseGaussian| x.mode().unwrap();
        test_relative(1e8, 1.0, 0.33333333333333332963, mode);
        test_relative(1e6, 1.0, 0.3333333333332962963, mode);
        test_relative(1.0, 1e8, 0.9999999850000001125, mode);
        test_relative(1e-8, 1.0, 9.9999998500000013342e-9, mode);
        // mu^2 overflows f64 here, so the squared term has to stay out of the
        // expression entirely, not merely be evaluated carefully
        test_relative(1e150, 1.0, 1.0 / 3.0, mode);
        test_relative(1e300, 1.0, 1.0 / 3.0, mode);

        // the mode is in the support, and below the mean, for every case above
        for &(mu, lambda) in &[
            (1e8, 1.0),
            (1e6, 1.0),
            (1.0, 1e8),
            (1e-8, 1.0),
            (1e150, 1.0),
            (1.0, 1.0),
        ] {
            let d = create_ok(mu, lambda);
            let m = d.mode().unwrap();
            assert!(m > 0.0 && m.is_finite(), "mu={mu:e} lambda={lambda:e}: mode {m}");
            assert!(m <= mu, "mu={mu:e} lambda={lambda:e}: mode {m} exceeds the mean");
        }
    }

    /// `z` can be exactly zero -- the ziggurat forms `u = 2f - 1` from a 53-bit
    /// `f`, and `f == 0.5` is one of its values (statrs-dev/statrs#423). The
    /// rationalized root divided by `mnu + sqrt(mnu (4 lambda + mnu))`, which
    /// is `0/0` there, so every draw from such a `z` was NaN.
    #[cfg(feature = "rand")]
    #[test]
    fn test_msh_root_at_zero() {
        use super::msh_smaller_root;

        for &(mu, lambda) in &[(1.0, 1.0), (2.0, 3.0), (0.1, 1000.0), (10.0, 0.1)] {
            let at_zero = msh_smaller_root(mu, lambda, 0.0);
            assert!(at_zero.is_finite(), "mu={mu} lambda={lambda}: {at_zero}");
            // nu = 0 puts the root at the mean
            assert_eq!(at_zero, mu, "mu={mu} lambda={lambda}");

            // and it is continuous into that corner rather than merely defined
            for &z in &[1e-300f64, 1e-160, 1e-8, 1e-4] {
                let y = msh_smaller_root(mu, lambda, z);
                assert!(y.is_finite() && y > 0.0, "mu={mu} lambda={lambda} z={z:e}: {y}");
                assert!(y <= mu, "mu={mu} lambda={lambda} z={z:e}: {y} exceeds mu");
            }

            // ordinary draws stay in the support
            for i in 1..200 {
                let z = -6.0 + i as f64 * 0.06;
                let y = msh_smaller_root(mu, lambda, z);
                assert!(y > 0.0 && y.is_finite(), "mu={mu} lambda={lambda} z={z}: {y}");
            }
        }
    }

    /// A sample is never NaN, over the full path including the conjugate branch.
    #[cfg(feature = "rand")]
    #[test]
    fn test_samples_are_finite_and_positive() {
        use ::rand::SeedableRng;
        use ::rand::distr::Distribution as _;
        use ::rand::rngs::StdRng;

        for &(mu, lambda) in &[(1.0, 1.0), (2.0, 3.0), (0.1, 1000.0), (10.0, 0.1)] {
            let d = create_ok(mu, lambda);
            let mut rng = StdRng::seed_from_u64(0x16 + (mu as u64));
            for _ in 0..20_000 {
                let x: f64 = d.sample(&mut rng);
                assert!(x.is_finite() && x > 0.0, "mu={mu} lambda={lambda}: sampled {x}");
            }
        }
    }

    #[test]
    fn test_pdf() {
        // mpmath at 50 significant digits
        let pdf = |arg: f64| move |x: InverseGaussian| x.pdf(arg);
        test_absolute(1.0, 1.0, 0.3989422804014326779399, 1e-15, pdf(1.0));
        test_absolute(1.0, 1.0, 0.8787825789354447940937, 1e-14, pdf(0.5));
        test_absolute(1.0, 1.0, 0.03941835796981973098901, 1e-15, pdf(3.0));
        test_absolute(2.0, 3.0, 0.3533380431253714144991, 1e-15, pdf(1.5));
        test_absolute(0.5, 0.25, 0.7786842707660154116726, 1e-14, pdf(0.4));
        test_absolute(1.0, 800.0, 11.28379167095512573896, 1e-13, pdf(1.0));
        test_absolute(3.0, 1.0, 0.009609160709649644201603, 1e-16, pdf(10.0));
        test_exact(1.0, 1.0, 0.0, pdf(0.0));
        test_exact(1.0, 1.0, 0.0, pdf(-1.0));
    }

    #[test]
    fn test_ln_pdf_consistent_with_pdf() {
        for (mu, lambda) in [(1.0, 1.0), (2.0, 3.0), (0.5, 0.25)] {
            let n = create_ok(mu, lambda);
            for x in [0.1, 0.5, 1.0, 2.0, 10.0] {
                prec::assert_relative_eq!(n.pdf(x).ln(), n.ln_pdf(x), max_relative = 1e-13);
            }
        }
    }

    #[test]
    fn test_cdf() {
        // mpmath at 50 significant digits
        let cdf = |arg: f64| move |x: InverseGaussian| x.cdf(arg);
        test_absolute(1.0, 1.0, 0.6681020012231706064271, 1e-15, cdf(1.0));
        test_absolute(1.0, 1.0, 0.3649755481729598905864, 1e-15, cdf(0.5));
        test_absolute(1.0, 1.0, 0.9531879207427883590297, 1e-15, cdf(3.0));
        test_absolute(2.0, 3.0, 0.4956901248416294820537, 1e-15, cdf(1.5));
        test_exact(1.0, 1.0, 0.0, cdf(0.0));
        test_exact(1.0, 1.0, 1.0, cdf(f64::INFINITY));
    }

    /// `e^(2 lambda / mu)` alone is `e^1600 = inf` here; a naive evaluation of
    /// the cdf formula returns NaN or inf. The log-domain second term keeps it
    /// a probability (Giner & Smyth, 2016).
    #[test]
    fn test_cdf_shape_overflow_regime() {
        let cdf = |arg: f64| move |x: InverseGaussian| x.cdf(arg);
        assert_eq!((2.0f64 * 800.0 / 1.0).exp(), f64::INFINITY, "premise");
        test_absolute(1.0, 800.0, 0.001517236267531762119023, 1e-16, cdf(0.9));
        test_absolute(1.0, 800.0, 0.5070501679916889068124, 1e-13, cdf(1.0));
        test_absolute(1.0, 800.0, 0.9966850760983806309312, 1e-13, cdf(1.1));
        test_absolute(0.1, 1000.0, 0.501994661537961729716, 1e-13, cdf(0.1));
    }

    #[test]
    fn test_sf() {
        // mpmath at 50 significant digits
        let sf = |arg: f64| move |x: InverseGaussian| x.sf(arg);
        test_absolute(1.0, 1.0, 0.04681207925721164097026, 1e-15, sf(3.0));
        test_absolute(1.0, 1.0, 3.631365872581210534235e-9, 1e-19, sf(30.0));
        test_absolute(2.0, 3.0, 0.00004039669278096080163591, 1e-16, sf(20.0));
        test_exact(1.0, 1.0, 1.0, sf(0.0));
        test_exact(1.0, 1.0, 0.0, sf(f64::INFINITY));
    }

    #[test]
    fn test_cdf_sf_sum_to_one() {
        for (mu, lambda) in [(1.0, 1.0), (2.0, 3.0), (1.0, 800.0)] {
            let n = create_ok(mu, lambda);
            for x in [0.2, 0.5, 1.0, 2.0, 5.0] {
                prec::assert_abs_diff_eq!(n.cdf(x) + n.sf(x), 1.0, epsilon = 1e-14);
            }
        }
    }

    /// References are mpmath at 60 significant digits; every probability here
    /// is far below `f64::MIN_POSITIVE`, so the linear-domain cdf/sf are hard
    /// zeros and their naive logs are `-inf`.
    #[test]
    fn test_ln_cdf_ln_sf_deep_tails() {
        let n = create_ok(1.0, 1.0);
        prec::assert_relative_eq!(n.ln_cdf(0.001), -502.6811655093445338242, max_relative = 1e-14);
        prec::assert_relative_eq!(n.ln_cdf(0.01), -51.54304262742703284796, max_relative = 1e-14);
        // and past the point where even the log-domain-assembled cdf is a
        // hard zero in linear space
        assert_eq!(n.cdf(0.0001), 0.0, "premise: cdf underflows here");
        prec::assert_relative_eq!(n.ln_cdf(0.0001), -5003.831111503650139571, max_relative = 1e-14);
        prec::assert_relative_eq!(n.ln_sf(1000.0), -509.5909128464241764622, max_relative = 1e-13);
        assert_eq!(n.sf(100000.0), 0.0, "premise: sf underflows here");
        prec::assert_relative_eq!(n.ln_sf(100000.0), -50016.49521454895014606, max_relative = 1e-13);
        let m = create_ok(2.0, 3.0);
        prec::assert_relative_eq!(m.ln_sf(5000.0), -1885.665692017919709583, max_relative = 1e-13);
        let s = create_ok(1.0, 800.0);
        prec::assert_relative_eq!(s.ln_cdf(0.5), -203.6289210970444983027, max_relative = 1e-13);
    }

    #[test]
    fn test_ln_cdf_ln_sf_consistent_with_linear() {
        for (mu, lambda) in [(1.0, 1.0), (2.0, 3.0), (1.0, 800.0)] {
            let n = create_ok(mu, lambda);
            for x in [0.3, 0.8, 1.0, 1.5, 4.0, 20.0] {
                let (c, s) = (n.cdf(x), n.sf(x));
                if c > 1e-290 && c < 1.0 {
                    prec::assert_relative_eq!(c.ln(), n.ln_cdf(x), epsilon = 1e-12, max_relative = 1e-12);
                }
                if s > 1e-290 && s < 1.0 {
                    prec::assert_relative_eq!(s.ln(), n.ln_sf(x), epsilon = 1e-12, max_relative = 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_continuous() {
        density_util::check_continuous_distribution(&create_ok(1.0, 1.0), 0.0, 40.0);
        density_util::check_continuous_distribution(&create_ok(2.0, 3.0), 0.0, 50.0);
    }

    #[cfg(feature = "rand")]
    #[test]
    fn test_sample_moments() {
        use ::rand::SeedableRng;
        use ::rand::distr::Distribution as _;
        use ::rand::rngs::StdRng;

        for (mu, lambda) in [(1.0, 1.0), (2.0, 3.0), (0.5, 8.0)] {
            let n = create_ok(mu, lambda);
            let mut rng = StdRng::seed_from_u64(0x1C + lambda as u64);
            const SAMPLES: usize = 200_000;
            let mut sum = 0.0;
            let mut min = f64::INFINITY;
            for _ in 0..SAMPLES {
                let x: f64 = n.sample(&mut rng);
                assert!(x > 0.0 && x.is_finite(), "sample {x} outside support");
                sum += x;
                min = min.min(x);
            }
            let sample_mean = sum / SAMPLES as f64;
            // sample mean is within ~6 sd / sqrt(n) of mu on a fixed seed
            let tol = 6.0 * (n.variance().unwrap() / SAMPLES as f64).sqrt();
            prec::assert_abs_diff_eq!(sample_mean, mu, epsilon = tol);
        }
    }
}
