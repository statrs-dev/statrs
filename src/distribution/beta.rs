use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use function::{beta, gamma};
use statistics::*;
use distribution::{Univariate, Continuous, Distribution};
use result::Result;
use error::StatsError;
use Float;

/// Implements the [Beta](https://en.wikipedia.org/wiki/Beta_distribution) distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Beta, Continuous};
/// use statrs::statistics::Mean;
/// use statrs::prec;
///
/// let n = Beta::new(2.0, 2.0).unwrap();
/// assert_eq!(n.mean(), 0.5);
/// assert!(prec::almost_eq(n.pdf(0.5), 1.5, 1e-14));
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Beta<T>
    where T: Float
{
    shape_a: T,
    shape_b: T,
}

impl<T> Beta<T>
    where T: Float
{
    /// Constructs a new beta distribution with shapeA (α) of `shape_a`
    /// and shapeB (β) of `shape_b`
    ///
    /// # Errors
    ///
    /// Returns an error if `shape_a` or `shape_b` are NaN.
    /// Also returns an error if `shape_a <= 0.0` or `shape_b <= 0.0`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Beta;
    ///
    /// let mut result = Beta::new(2f64, 2f64);
    /// assert!(result.is_ok());
    ///
    /// result = Beta::new(0f64, 0f64);
    /// assert!(result.is_err());
    /// ```
    pub fn new(shape_a: T, shape_b: T) -> Result<Beta<T>> {
        if !valid_beta_parameters(shape_a, shape_b) {
            Err(StatsError::BadParams)
        } else {
            Ok(Beta {
                shape_a: shape_a,
                shape_b: shape_b,
            })
        }
    }

    /// Returns the shapeA (α) of the beta distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Beta;
    ///
    /// let n = Beta::new(2.0, 2.0).unwrap();
    /// assert_eq!(n.shape_a(), 2.0);
    /// ```
    pub fn shape_a(&self) -> T {
        self.shape_a
    }

    /// Returns the shapeB (β) of the beta distributionβ
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Beta;
    ///
    /// let n = Beta::new(2.0, 2.0).unwrap();
    /// assert_eq!(n.shape_b(), 2.0);
    /// ```
    pub fn shape_b(&self) -> T {
        self.shape_b
    }
}

impl<T> Sample<T> for Beta<T>
    where T: Float
{
    /// Generate a random sample from a beta distribution
    /// using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details.
    fn sample<R: Rng>(&mut self, r: &mut R) -> T {
        super::Distribution::sample(self, r)
    }
}

impl<T> IndependentSample<T> for Beta<T>
    where T: Float
{
    /// Generate a random independent sample from a beta distribution
    /// using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details.
    fn ind_sample<R: Rng>(&self, r: &mut R) -> T {
        super::Distribution::sample(self, r)
    }
}

impl<T> Distribution<T> for Beta<T>
    where T: Float
{
    /// Generate a random sample from a beta distribution using
    /// `r` as the source of randomness. Generated by sampling
    /// two gamma distributions and normalizing.
    ///
    /// # Examples
    ///
    /// ```
    /// # extern crate rand;
    /// # extern crate statrs;
    /// use rand::StdRng;
    /// use statrs::distribution::{Beta, Distribution};
    ///
    /// # fn main() {
    /// let mut r = rand::StdRng::new().unwrap();
    /// let n = Beta::new(2f64, 2f64).unwrap();
    /// print!("{}", n.sample::<StdRng>(&mut r));
    /// # }
    /// ```
    fn sample<R: Rng>(&self, r: &mut R) -> T {
        let x = super::gamma::sample_unchecked(r, self.shape_a, T::one());
        let y = super::gamma::sample_unchecked(r, self.shape_b, T::one());
        x / (x + y)
    }
}

impl<T> Univariate<T, T> for Beta<T>
    where T: Float
{
    /// Calculates the cumulative distribution function for the beta distribution
    /// at `x`
    ///
    /// # Panics
    ///
    /// If `x < 0.0` or `x > 1.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// I_x(α, β)
    /// ```
    ///
    /// where `α` is shapeA, `β` is shapeB, and `I_x` is the regularized
    /// lower incomplete beta function
    fn cdf(&self, x: T) -> T {
        assert!(x >= T::zero() && x <= T::one(),
                format!("{}", StatsError::ArgIntervalIncl("x", 0.0, 1.0)));
        if x == T::one() {
            x
        } else if self.shape_a == T::infinity() && self.shape_b == T::infinity() {
            if x < T::from(0.5).unwrap() {
                T::zero()
            } else {
                T::one()
            }
        } else if self.shape_a == T::infinity() {
            if x < T::one() { T::zero() } else { T::one() }
        } else if self.shape_b == T::infinity() {
            T::one()
        } else if self.shape_a == T::one() && self.shape_b == T::one() {
            x
        } else {
            beta::beta_reg(self.shape_a, self.shape_b, x)
        }
    }
}

impl<T> Min<T> for Beta<T>
    where T: Float
{
    /// Returns the minimum value in the domain of the
    /// beta distribution representable by a double precision
    /// float
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 0
    /// ```
    fn min(&self) -> T {
        T::zero()
    }
}

impl<T> Max<T> for Beta<T>
    where T: Float
{
    /// Returns the maximum value in the domain of the
    /// beta distribution representable by a double precision
    /// float
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 1
    /// ```
    fn max(&self) -> T {
        T::one()
    }
}

impl<T> Mean<T> for Beta<T>
    where T: Float
{
    /// Returns the mean of the beta distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// α / (α + β)
    /// ```
    ///
    /// where `α` is shapeA and `β` is shapeB
    fn mean(&self) -> T {
        if self.shape_a == T::infinity() && self.shape_b == T::infinity() {
            T::from(0.5).unwrap()
        } else if self.shape_a == T::infinity() {
            T::one()
        } else if self.shape_b == T::infinity() {
            T::zero()
        } else {
            self.shape_a / (self.shape_a + self.shape_b)
        }
    }
}

impl<T> Variance<T> for Beta<T>
    where T: Float
{
    /// Returns the variance of the beta distribution
    ///
    /// # Remarks
    ///
    /// Returns NaN if either `shape_a` or `shape_b` are
    /// positive infinity
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (α * β) / ((α + β)^2 * (α + β + 1))
    /// ```
    ///
    /// where `α` is shapeA and `β` is shapeB
    fn variance(&self) -> T {
        self.shape_a * self.shape_b /
        ((self.shape_a + self.shape_b) * (self.shape_a + self.shape_b) *
         (self.shape_a + self.shape_b + T::one()))
    }

    /// Returns the standard deviation of the beta distribution
    ///
    /// # Remarks
    ///
    /// Returns NaN if either `shape_a` or `shape_b` are
    /// positive infinity
    ///
    /// # Formula
    ///
    /// ```ignore
    /// sqrt((α * β) / ((α + β)^2 * (α + β + 1)))
    /// ```
    ///
    /// where `α` is shapeA and `β` is shapeB
    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }
}

impl<T> Entropy<T> for Beta<T>
    where T: Float
{
    /// Returns the entropy of the beta distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln(B(α, β)) - (α - 1)ψ(α) - (β - 1)ψ(β) + (α + β - 2)ψ(α + β)
    /// ```
    ///
    /// where `α` is shapeA, `β` is shapeB and `ψ` is the digamma function
    fn entropy(&self) -> T {
        if self.shape_a == T::infinity() || self.shape_b == T::infinity() {
            T::zero()
        } else {
            beta::ln_beta(self.shape_a, self.shape_b) -
            (self.shape_a - T::one()) * gamma::digamma(self.shape_a) -
            (self.shape_b - T::one()) * gamma::digamma(self.shape_b) +
            (self.shape_a + self.shape_b - T::from(2.0).unwrap()) *
            gamma::digamma(self.shape_a + self.shape_b)
        }
    }
}

impl<T> Skewness<T> for Beta<T>
    where T: Float
{
    /// Returns the skewness of the Beta distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 2(β - α) * sqrt(α + β + 1) / ((α + β + 2) * sqrt(αβ))
    /// ```
    ///
    /// where `α` is shapeA and `β` is shapeB
    fn skewness(&self) -> T {
        if self.shape_a == T::infinity() && self.shape_b == T::infinity() {
            T::zero()
        } else if self.shape_a == T::infinity() {
            T::from(-2.0).unwrap()
        } else if self.shape_b == T::infinity() {
            T::from(2.0).unwrap()
        } else {
            T::from(2.0).unwrap() * (self.shape_b - self.shape_a) *
            (self.shape_a + self.shape_b + T::one()).sqrt() /
            ((self.shape_a + self.shape_b + T::from(2.0).unwrap()) *
             (self.shape_a * self.shape_b).sqrt())
        }
    }
}

impl<T> Mode<T> for Beta<T>
    where T: Float
{
    /// Returns the mode of the Beta distribution.
    ///
    /// # Remarks
    ///
    /// Since the mode is technically only calculate for `α > 1, β > 1`, those
    /// are the only values we allow. We may consider relaxing this constraint in
    /// the future.
    ///
    /// # Panics
    ///
    /// If `α <= 1` or `β <= 1`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (α - 1) / (α + β - 2)
    /// ```
    ///
    /// where `α` is shapeA and `β` is shapeB
    fn mode(&self) -> T {
        // TODO: perhaps relax constraint in order to allow calculation
        // of 'anti-mode;
        assert!(self.shape_a > T::one(),
                format!("{}", StatsError::ArgGt("shape_a", 1.0)));
        assert!(self.shape_b > T::one(),
                format!("{}", StatsError::ArgGt("shape_b", 1.0)));
        if self.shape_a == T::infinity() && self.shape_b == T::infinity() {
            T::from(0.5).unwrap()
        } else if self.shape_a == T::infinity() {
            T::one()
        } else if self.shape_b == T::infinity() {
            T::zero()
        } else {
            (self.shape_a - T::one()) / (self.shape_a + self.shape_b - T::from(2.0).unwrap())
        }
    }
}

impl<T> Continuous<T, T> for Beta<T>
    where T: Float
{
    /// Calculates the probability density function for the beta distribution at `x`.
    ///
    /// # Panics
    ///
    /// If `x < 0.0` or `x > 1.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// let B(α, β) = Γ(α)Γ(β)/Γ(α + β)
    ///
    /// x^(α - 1) * (1 - x)^(β - 1) / B(α, β)
    /// ```
    ///
    /// where `α` is shapeA, `β` is shapeB, and `Γ` is the gamma function
    fn pdf(&self, x: T) -> T {
        assert!(x >= T::zero() && x <= T::one(),
                format!("{}", StatsError::ArgIntervalIncl("x", 0.0, 1.0)));
        if self.shape_a == T::infinity() && self.shape_b == T::infinity() {
            if x == T::from(0.5).unwrap() {
                T::infinity()
            } else {
                T::zero()
            }
        } else if self.shape_a == T::infinity() {
            if x == T::one() {
                T::infinity()
            } else {
                T::zero()
            }
        } else if self.shape_b == T::infinity() {
            if x == T::one() {
                T::infinity()
            } else {
                T::zero()
            }
        } else if self.shape_a == T::one() && self.shape_b == T::one() {
            T::one()
        } else if self.shape_a > T::from(80.0).unwrap() || self.shape_b > T::from(80.0).unwrap() {
            self.ln_pdf(x).exp()
        } else {
            let bb = gamma::gamma(self.shape_a + self.shape_b) /
                     (gamma::gamma(self.shape_a) * gamma::gamma(self.shape_b));
            bb * x.powf(self.shape_a - T::one()) * (T::one() - x).powf(self.shape_b - T::one())
        }
    }

    /// Calculates the log probability density function for the beta distribution at `x`.
    ///
    /// # Panics
    ///
    /// If `x < 0.0` or `x > 1.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// let B(α, β) = Γ(α)Γ(β)/Γ(α + β)
    ///
    /// ln(x^(α - 1) * (1 - x)^(β - 1) / B(α, β))
    /// ```
    ///
    /// where `α` is shapeA, `β` is shapeB, and `Γ` is the gamma function
    fn ln_pdf(&self, x: T) -> T {
        assert!(x >= T::zero() && x <= T::one(),
                format!("{}", StatsError::ArgIntervalIncl("x", 0.0, 1.0)));
        if self.shape_a == T::infinity() && self.shape_b == T::infinity() {
            if x == T::from(0.5).unwrap() {
                T::infinity()
            } else {
                T::neg_infinity()
            }
        } else if self.shape_a == T::infinity() {
            if x == T::one() {
                T::infinity()
            } else {
                T::neg_infinity()
            }
        } else if self.shape_b == T::infinity() {
            if x == T::zero() {
                T::infinity()
            } else {
                T::neg_infinity()
            }
        } else if self.shape_a == T::one() && self.shape_b == T::one() {
            T::zero()
        } else {
            let aa = gamma::ln_gamma(self.shape_a + self.shape_b) - gamma::ln_gamma(self.shape_a) -
                     gamma::ln_gamma(self.shape_b);
            let bb = if self.shape_a == T::one() && x == T::zero() {
                T::zero()
            } else if x == T::zero() {
                T::neg_infinity()
            } else {
                (self.shape_a - T::one()) * x.ln()
            };
            let cc = if self.shape_b == T::one() && x == T::one() {
                T::zero()
            } else if x == T::one() {
                T::neg_infinity()
            } else {
                (self.shape_b - T::one()) * (T::one() - x).ln()
            };
            aa + bb + cc
        }
    }
}

// Returns if `shape_a` and `shape_b` are valid parameters
// for a beta distribution
fn valid_beta_parameters<T>(shape_a: T, shape_b: T) -> bool
    where T: Float
{
    if shape_a.is_nan() || shape_b.is_nan() {
        false
    } else if shape_a <= T::zero() || shape_b <= T::zero() {
        false
    } else {
        true
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use std::f64;
    use statistics::*;
    use distribution::{Univariate, Continuous, Beta};

    fn try_create(shape_a: f64, shape_b: f64) -> Beta {
        let n = Beta::new(shape_a, shape_b);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(shape_a: f64, shape_b: f64) {
        let n = try_create(shape_a, shape_b);
        assert_eq!(n.shape_a(), shape_a);
        assert_eq!(n.shape_b(), shape_b);
    }

    fn bad_create_case(shape_a: f64, shape_b: f64) {
        let n = Beta::new(shape_a, shape_b);
        assert!(n.is_err());
    }

    fn get_value<F>(shape_a: f64, shape_b: f64, eval: F) -> f64
        where F: Fn(Beta) -> f64
    {
        let n = try_create(shape_a, shape_b);
        eval(n)
    }

    fn test_case<F>(shape_a: f64, shape_b: f64, expected: f64, eval: F)
        where F: Fn(Beta) -> f64
    {
        let x = get_value(shape_a, shape_b, eval);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(shape_a: f64, shape_b: f64, expected: f64, acc: f64, eval: F)
        where F: Fn(Beta) -> f64
    {
        let x = get_value(shape_a, shape_b, eval);
        assert_almost_eq!(expected, x, acc);
    }

    fn test_is_nan<F>(shape_a: f64, shape_b: f64, eval: F)
        where F: Fn(Beta) -> f64
    {
        assert!(get_value(shape_a, shape_b, eval).is_nan())
    }

    #[test]
    fn test_create() {
        create_case(1.0, 1.0);
        create_case(9.0, 1.0);
        create_case(5.0, 100.0);
        create_case(1.0, f64::INFINITY);
        create_case(f64::INFINITY, 1.0);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(0.0, 0.0);
        bad_create_case(0.0, 0.1);
        bad_create_case(1.0, 0.0);
        bad_create_case(0.0, f64::INFINITY);
        bad_create_case(f64::INFINITY, 0.0);
        bad_create_case(f64::NAN, 1.0);
        bad_create_case(1.0, f64::NAN);
        bad_create_case(f64::NAN, f64::NAN);
        bad_create_case(1.0, -1.0);
        bad_create_case(-1.0, 1.0);
        bad_create_case(-1.0, -1.0);
    }

    #[test]
    fn test_mean() {
        test_case(1.0, 1.0, 0.5, |x| x.mean());
        test_case(9.0, 1.0, 0.9, |x| x.mean());
        test_case(5.0, 100.0, 0.047619047619047619047616, |x| x.mean());
        test_case(1.0, f64::INFINITY, 0.0, |x| x.mean());
        test_case(f64::INFINITY, 1.0, 1.0, |x| x.mean());
        test_case(f64::INFINITY, f64::INFINITY, 0.5, |x| x.mean());
    }

    #[test]
    fn test_variance() {
        test_case(1.0, 1.0, 1.0 / 12.0, |x| x.variance());
        test_case(9.0, 1.0, 9.0 / 1100.0, |x| x.variance());
        test_case(5.0, 100.0, 500.0 / 1168650.0, |x| x.variance());
        test_is_nan(1.0, f64::INFINITY, |x| x.variance());
        test_is_nan(f64::INFINITY, 1.0, |x| x.variance());
        test_is_nan(f64::INFINITY, f64::INFINITY, |x| x.variance());
    }

    #[test]
    fn test_std_dev() {
        test_case(1.0, 1.0, (1f64 / 12.0).sqrt(), |x| x.std_dev());
        test_case(9.0, 1.0, (9f64 / 1100.0).sqrt(), |x| x.std_dev());
        test_case(5.0, 100.0, (500f64 / 1168650.0).sqrt(), |x| x.std_dev());
        test_is_nan(1.0, f64::INFINITY, |x| x.std_dev());
        test_is_nan(f64::INFINITY, 1.0, |x| x.std_dev());
        test_is_nan(f64::INFINITY, f64::INFINITY, |x| x.std_dev());
    }

    #[test]
    fn test_entropy() {
        test_almost(1.0, 1.0, 0.0, 1e-15, |x| x.entropy());
        test_almost(9.0, 1.0, -1.3083356884473304939016015849561625204060922267565917, 1e-13, |x| x.entropy());
        test_almost(5.0, 100.0, -2.5201623187602743679459255108827601222133603091753153, 1e-13, |x| x.entropy());
        test_case(1.0, f64::INFINITY, 0.0, |x| x.entropy());
        test_case(f64::INFINITY, 1.0, 0.0, |x| x.entropy());
        test_case(f64::INFINITY, f64::INFINITY, 0.0, |x| x.entropy());
    }

    #[test]
    fn test_skewness() {
        test_case(1.0, 1.0, 0.0, |x| x.skewness());
        test_almost(9.0, 1.0, -1.4740554623801777107177478829647496373009282424841579, 1e-15, |x| x.skewness());
        test_almost(5.0, 100.0, 0.81759410927553430354583159143895018978562196953345572, 1e-15, |x| x.skewness());
        test_case(1.0, f64::INFINITY, 2.0, |x| x.skewness());
        test_case(f64::INFINITY, 1.0, -2.0, |x| x.skewness());
        test_case(f64::INFINITY, f64::INFINITY, 0.0, |x| x.skewness());
    }

    #[test]
    fn test_mode() {
        test_case(5.0, 100.0, 0.038834951456310676243255386452801758423447608947753906, |x| x.mode());
        test_case(2.0, f64::INFINITY, 0.0, |x| x.mode());
        test_case(f64::INFINITY, 2.0, 1.0, |x| x.mode());
        test_case(f64::INFINITY, f64::INFINITY, 0.5, |x| x.mode());
    }

    #[test]
    #[should_panic]
    fn test_mode_shape_a_lte_one() {
        get_value(1.0, 5.0, |x| x.mode());
    }

    #[test]
    #[should_panic]
    fn test_mode_shape_b_lte_one() {
        get_value(5.0, 1.0, |x| x.mode());
    }

    #[test]
    fn test_min_max() {
        test_case(1.0, 1.0, 0.0, |x| x.min());
        test_case(1.0, 1.0, 1.0, |x| x.max());
    }

    #[test]
    fn test_pdf() {
        test_case(1.0, 1.0, 1.0, |x| x.pdf(0.0));
        test_case(1.0, 1.0, 1.0, |x| x.pdf(0.5));
        test_case(1.0, 1.0, 1.0, |x| x.pdf(1.0));
        test_case(9.0, 1.0, 0.0, |x| x.pdf(0.0));
        test_almost(9.0, 1.0, 0.03515625, 1e-15, |x| x.pdf(0.5));
        test_almost(9.0, 1.0, 9.0, 1e-13, |x| x.pdf(1.0));
        test_case(5.0, 100.0, 0.0, |x| x.pdf(0.0));
        test_almost(5.0, 100.0, 4.534102298350337661e-23, 1e-35, |x| x.pdf(0.5));
        test_case(5.0, 100.0, 0.0, |x| x.pdf(1.0));
        test_case(5.0, 100.0, 0.0, |x| x.pdf(1.0));
        test_case(1.0, f64::INFINITY, f64::INFINITY, |x| x.pdf(0.0));
        test_case(1.0, f64::INFINITY, 0.0, |x| x.pdf(0.5));
        test_case(1.0, f64::INFINITY, 0.0, |x| x.pdf(1.0));
        test_case(f64::INFINITY, 1.0, 0.0, |x| x.pdf(0.0));
        test_case(f64::INFINITY, 1.0, 0.0, |x| x.pdf(0.5));
        test_case(f64::INFINITY, 1.0, f64::INFINITY, |x| x.pdf(1.0));
        test_case(f64::INFINITY, f64::INFINITY, 0.0, |x| x.pdf(0.0));
        test_case(f64::INFINITY, f64::INFINITY, f64::INFINITY, |x| x.pdf(0.5));
        test_case(f64::INFINITY, f64::INFINITY, 0.0, |x| x.pdf(1.0));
    }

    #[test]
    #[should_panic]
    fn test_pdf_input_lt_zero() {
        get_value(1.0, 1.0, |x| x.pdf(-1.0));
    }

    #[test]
    #[should_panic]
    fn test_pdf_input_gt_one() {
        get_value(1.0, 1.0, |x| x.pdf(2.0));
    }

    #[test]
    fn test_ln_pdf() {
        test_case(1.0, 1.0, 0.0, |x| x.ln_pdf(0.0));
        test_case(1.0, 1.0, 0.0, |x| x.ln_pdf(0.5));
        test_case(1.0, 1.0, 0.0, |x| x.ln_pdf(1.0));
        test_case(9.0, 1.0, f64::NEG_INFINITY, |x| x.ln_pdf(0.0));
        test_almost(9.0, 1.0, -3.3479528671433430925473664978203611353090199592365458, 1e-13, |x| x.ln_pdf(0.5));
        test_almost(9.0, 1.0, 2.1972245773362193827904904738450514092949811156454996, 1e-13, |x| x.ln_pdf(1.0));
        test_case(5.0, 100.0, f64::NEG_INFINITY, |x| x.ln_pdf(0.0));
        test_almost(5.0, 100.0, -51.447830024537682154565870837960406410586196074573801, 1e-12, |x| x.ln_pdf(0.5));
        test_case(5.0, 100.0, f64::NEG_INFINITY, |x| x.ln_pdf(1.0));
        test_case(1.0, f64::INFINITY, f64::INFINITY, |x| x.ln_pdf(0.0));
        test_case(1.0, f64::INFINITY, f64::NEG_INFINITY, |x| x.ln_pdf(0.5));
        test_case(1.0, f64::INFINITY, f64::NEG_INFINITY, |x| x.ln_pdf(1.0));
        test_case(f64::INFINITY, 1.0, f64::NEG_INFINITY, |x| x.ln_pdf(0.0));
        test_case(f64::INFINITY, 1.0, f64::NEG_INFINITY, |x| x.ln_pdf(0.5));
        test_case(f64::INFINITY, 1.0, f64::INFINITY, |x| x.ln_pdf(1.0));
        test_case(f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, |x| x.ln_pdf(0.0));
        test_case(f64::INFINITY, f64::INFINITY, f64::INFINITY, |x| x.ln_pdf(0.5));
        test_case(f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, |x| x.ln_pdf(1.0));
    }

    #[test]
    #[should_panic]
    fn test_ln_pdf_input_lt_zero() {
        get_value(1.0, 1.0, |x| x.ln_pdf(-1.0));
    }

    #[test]
    #[should_panic]
    fn test_ln_pdf_input_gt_one() {
        get_value(1.0, 1.0, |x| x.ln_pdf(2.0));
    }

    #[test]
    fn test_cdf() {
        test_case(1.0, 1.0, 0.0, |x| x.cdf(0.0));
        test_case(1.0, 1.0, 0.5, |x| x.cdf(0.5));
        test_case(1.0, 1.0, 1.0, |x| x.cdf(1.0));
        test_case(9.0, 1.0, 0.0, |x| x.cdf(0.0));
        test_almost(9.0, 1.0, 0.001953125, 1e-16, |x| x.cdf(0.5));
        test_case(9.0, 1.0, 1.0, |x| x.cdf(1.0));
        test_case(5.0, 100.0, 0.0, |x| x.cdf(0.0));
        test_case(5.0, 100.0, 1.0, |x| x.cdf(0.5));
        test_case(5.0, 100.0, 1.0, |x| x.cdf(1.0));
        test_case(1.0, f64::INFINITY, 1.0, |x| x.cdf(0.0));
        test_case(1.0, f64::INFINITY, 1.0, |x| x.cdf(0.5));
        test_case(1.0, f64::INFINITY, 1.0, |x| x.cdf(1.0));
        test_case(f64::INFINITY, 1.0, 0.0, |x| x.cdf(0.0));
        test_case(f64::INFINITY, 1.0, 0.0, |x| x.cdf(0.5));
        test_case(f64::INFINITY, 1.0, 1.0, |x| x.cdf(1.0));
        test_case(f64::INFINITY, f64::INFINITY, 0.0, |x| x.cdf(0.0));
        test_case(f64::INFINITY, f64::INFINITY, 1.0, |x| x.cdf(0.5));
        test_case(f64::INFINITY, f64::INFINITY, 1.0, |x| x.cdf(1.0));
    }

    #[test]
    #[should_panic]
    fn test_cdf_input_lt_zero() {
        get_value(1.0, 1.0, |x| x.cdf(-1.0));
    }

    #[test]
    #[should_panic]
    fn test_cdf_input_gt_zero() {
        get_value(1.0, 1.0, |x| x.cdf(2.0));
    }
}
