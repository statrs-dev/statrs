use crate::distribution::{Binomial, BinomialError, Discrete, DiscreteCDF};
use crate::statistics::*;

/// Implements the
/// [Bernoulli](https://en.wikipedia.org/wiki/Bernoulli_distribution)
/// distribution which is a special case of the
/// [Binomial](https://en.wikipedia.org/wiki/Binomial_distribution)
/// distribution where `n = 1` (referenced [Here](./struct.Binomial.html))
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Bernoulli, BinomialError, Discrete};
/// use statrs::statistics::*;
///
/// let n = Bernoulli::new(0.5)?;
/// assert_eq!(n.mean(), 0.5);
/// assert_eq!(n.pmf(0), 0.5);
/// assert_eq!(n.pmf(1), 0.5);
/// # Ok::<(), BinomialError>(())
/// ```
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Bernoulli {
    b: Binomial,
}

impl Bernoulli {
    /// Constructs a new bernoulli distribution with
    /// the given `p` probability of success.
    ///
    /// # Errors
    ///
    /// Returns an error if `p` is `NaN`, less than `0.0`
    /// or greater than `1.0`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Bernoulli;
    ///
    /// let mut result = Bernoulli::new(0.5);
    /// assert!(result.is_ok());
    ///
    /// result = Bernoulli::new(-0.5);
    /// assert!(result.is_err());
    /// ```
    pub fn new(p: f64) -> Result<Bernoulli, BinomialError> {
        Binomial::new(p, 1).map(|b| Bernoulli { b })
    }

    /// Returns the probability of success `p` of the
    /// bernoulli distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{Bernoulli, BinomialError};
    ///
    /// let n = Bernoulli::new(0.5)?;
    /// assert_eq!(n.p(), 0.5);
    /// # Ok::<(), BinomialError>(())
    /// ```
    pub fn p(&self) -> f64 {
        self.b.p()
    }

    /// Returns the number of trials `n` of the
    /// bernoulli distribution. Will always be `1.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{Bernoulli, BinomialError};
    ///
    /// let n = Bernoulli::new(0.5)?;
    /// assert_eq!(n.n(), 1);
    /// # Ok::<(), BinomialError>(())
    /// ```
    pub fn n(&self) -> u64 {
        1
    }
}

impl std::fmt::Display for Bernoulli {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Bernoulli({})", self.p())
    }
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl ::rand::distributions::Distribution<bool> for Bernoulli {
    fn sample<R: ::rand::Rng + ?Sized>(&self, rng: &mut R) -> bool {
        rng.gen_bool(self.p())
    }
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl ::rand::distributions::Distribution<f64> for Bernoulli {
    fn sample<R: ::rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        rng.sample::<bool, _>(self) as u8 as f64
    }
}

impl DiscreteCDF<u64, f64> for Bernoulli {
    /// Calculates the cumulative distribution
    /// function for the bernoulli distribution at `x`.
    ///
    /// # Formula
    ///
    /// ```text
    /// if x < 0 { 0 }
    /// else if x >= 1 { 1 }
    /// else { 1 - p }
    /// ```
    fn cdf(&self, x: u64) -> f64 {
        self.b.cdf(x)
    }

    /// Calculates the survival function for the
    /// bernoulli distribution at `x`.
    ///
    /// # Formula
    ///
    /// ```text
    /// if x < 0 { 1 }
    /// else if x >= 1 { 0 }
    /// else { p }
    /// ```
    fn sf(&self, x: u64) -> f64 {
        self.b.sf(x)
    }
}

impl Min<u64> for Bernoulli {
    /// Returns the minimum value in the domain of the
    /// bernoulli distribution representable by a 64-
    /// bit integer
    ///
    /// # Formula
    ///
    /// ```text
    /// 0
    /// ```
    fn min(&self) -> u64 {
        0
    }
}

impl Max<u64> for Bernoulli {
    /// Returns the maximum value in the domain of the
    /// bernoulli distribution representable by a 64-
    /// bit integer
    ///
    /// # Formula
    ///
    /// ```text
    /// 1
    /// ```
    fn max(&self) -> u64 {
        1
    }
}

impl Median<f64> for Bernoulli {
    /// Returns the median of the bernoulli
    /// distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// if p < 0.5 { 0 }
    /// else if p > 0.5 { 1 }
    /// else { 0.5 }
    /// ```
    fn median(&self) -> f64 {
        self.b.median()
    }
}

impl Mode<Option<u64>> for Bernoulli {
    /// Returns the mode of the bernoulli distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// if p < 0.5 { 0 }
    /// else { 1 }
    /// ```
    fn mode(&self) -> Option<u64> {
        self.b.mode()
    }
}

impl Discrete<u64, f64> for Bernoulli {
    /// Calculates the probability mass function for the
    /// bernoulli distribution at `x`.
    ///
    /// # Formula
    ///
    /// ```text
    /// if x == 0 { 1 - p }
    /// else { p }
    /// ```
    fn pmf(&self, x: u64) -> f64 {
        self.b.pmf(x)
    }

    /// Calculates the log probability mass function for the
    /// bernoulli distribution at `x`.
    ///
    /// # Formula
    ///
    /// ```text
    /// else if x == 0 { ln(1 - p) }
    /// else { ln(p) }
    /// ```
    fn ln_pmf(&self, x: u64) -> f64 {
        self.b.ln_pmf(x)
    }
}

/// returns the mean of a bernoulli variable
///
/// this is also it's probability parameter

impl Mean for Bernoulli {
    type Mu = f64;
    fn mean(&self) -> Self::Mu {
        self.p()
    }
}

/// returns the variance of a bernoulli variable
///
/// # Formula
/// ```text
/// p (1 - p)
/// ```

impl Variance for Bernoulli {
    type Var = f64;
    fn variance(&self) -> Self::Var {
        let p = self.p();
        p.mul_add(-p, p)
    }
}

/// Returns the skewness of a bernoulli variable
///
/// # Formula
///
/// ```text
/// (1 - 2p) / sqrt(p * (1 - p)))
/// ```

impl Skewness for Bernoulli {
    type Skew = f64;
    fn skewness(&self) -> Self::Skew {
        let d = 0.5 - self.p();
        2.0 * d / (0.25 - d * d).sqrt()
    }
}

/// Returns the excess kurtosis of a bernoulli variable
///
/// # Formula
///
/// ```text
/// pq^-1 - 6; pq = p (1-p)
/// ```

impl ExcessKurtosis for Bernoulli {
    type Kurt = f64;
    fn excess_kurtosis(&self) -> Self::Kurt {
        let p = self.p();
        let pq = p.mul_add(-p, p);
        pq.recip() - 6.0
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod testing {
    use super::*;
    use crate::testing_boiler;

    testing_boiler!(p: f64; Bernoulli; BinomialError);

    #[test]
    fn test_create() {
        create_ok(0.0);
        create_ok(0.3);
        create_ok(1.0);
    }

    #[test]
    fn test_bad_create() {
        create_err(f64::NAN);
        create_err(-1.0);
        create_err(2.0);
    }

    #[test]
    fn test_cdf_upper_bound() {
        let cdf = |arg: u64| move |x: Bernoulli| x.cdf(arg);
        test_relative(0.3, 1., cdf(1));
    }

    #[test]
    fn test_sf_upper_bound() {
        let sf = |arg: u64| move |x: Bernoulli| x.sf(arg);
        test_relative(0.3, 0., sf(1));
    }

    #[test]
    fn test_cdf() {
        let cdf = |arg: u64| move |x: Bernoulli| x.cdf(arg);
        test_relative(0.0, 1.0, cdf(0));
        test_relative(0.0, 1.0, cdf(1));
        test_absolute(0.3, 0.7, 1e-15, cdf(0));
        test_absolute(0.7, 0.3, 1e-15, cdf(0));
    }

    #[test]
    fn test_sf() {
        let sf = |arg: u64| move |x: Bernoulli| x.sf(arg);
        test_relative(0.0, 0.0, sf(0));
        test_relative(0.0, 0.0, sf(1));
        test_absolute(0.3, 0.3, 1e-15, sf(0));
        test_absolute(0.7, 0.7, 1e-15, sf(0));
    }
}
