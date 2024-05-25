use crate::distribution::{Binomial, Discrete, DiscreteCDF};
use crate::statistics::*;
use crate::Result;
use rand::Rng;

/// Implements the
/// [Bernoulli](https://en.wikipedia.org/wiki/Bernoulli_distribution)
/// distribution which is a special case of the
/// [Binomial](https://en.wikipedia.org/wiki/Binomial_distribution)
/// distribution where `n = 1` (referenced [Here](./struct.Binomial.html))
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Bernoulli, Discrete};
/// use statrs::statistics::Distribution;
///
/// let n = Bernoulli::new(0.5).unwrap();
/// assert_eq!(n.mean().unwrap(), 0.5);
/// assert_eq!(n.pmf(0), 0.5);
/// assert_eq!(n.pmf(1), 0.5);
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
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
    pub fn new(p: f64) -> Result<Bernoulli> {
        Binomial::new(p, 1).map(|b| Bernoulli { b })
    }

    /// Returns the probability of success `p` of the
    /// bernoulli distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Bernoulli;
    ///
    /// let n = Bernoulli::new(0.5).unwrap();
    /// assert_eq!(n.p(), 0.5);
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
    /// use statrs::distribution::Bernoulli;
    ///
    /// let n = Bernoulli::new(0.5).unwrap();
    /// assert_eq!(n.n(), 1);
    /// ```
    pub fn n(&self) -> u64 {
        1
    }
}

impl ::rand::distributions::Distribution<f64> for Bernoulli {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        rng.gen_bool(self.p()) as u8 as f64
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

impl Distribution<f64> for Bernoulli {
    /// Returns the mean of the bernoulli
    /// distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// p
    /// ```
    fn mean(&self) -> Option<f64> {
        self.b.mean()
    }

    /// Returns the variance of the bernoulli
    /// distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// p * (1 - p)
    /// ```
    fn variance(&self) -> Option<f64> {
        self.b.variance()
    }

    /// Returns the entropy of the bernoulli
    /// distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// q = (1 - p)
    /// -q * ln(q) - p * ln(p)
    /// ```
    fn entropy(&self) -> Option<f64> {
        self.b.entropy()
    }

    /// Returns the skewness of the bernoulli
    /// distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// q = (1 - p)
    /// (1 - 2p) / sqrt(p * q)
    /// ```
    fn skewness(&self) -> Option<f64> {
        self.b.skewness()
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

#[rustfmt::skip]
#[cfg(test)]
mod testing {
    use std::fmt::Debug;
    use crate::distribution::DiscreteCDF;
    use super::Bernoulli;

    fn try_create(p: f64) -> Bernoulli {
        let n = Bernoulli::new(p);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(p: f64) {
        let dist = try_create(p);
        assert_eq!(p, dist.p());
    }

    fn bad_create_case(p: f64) {
        let n = Bernoulli::new(p);
        assert!(n.is_err());
    }

    fn get_value<T, F>(p: f64, eval: F) -> T
        where T: PartialEq + Debug,
              F: Fn(Bernoulli) -> T
    {
        let n = try_create(p);
        eval(n)
    }

    fn test_case<T, F>(p: f64, expected: T, eval: F)
        where T: PartialEq + Debug,
              F: Fn(Bernoulli) -> T
    {
        let x = get_value(p, eval);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(p: f64, expected: f64, acc: f64, eval: F)
        where F: Fn(Bernoulli) -> f64
    {
        let x = get_value(p, eval);
        assert_almost_eq!(expected, x, acc);
    }

    #[test]
    fn test_create() {
        create_case(0.0);
        create_case(0.3);
        create_case(1.0);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(f64::NAN);
        bad_create_case(-1.0);
        bad_create_case(2.0);
    }

    #[test]
    fn test_cdf_upper_bound() {
        let cdf = |arg: u64| move |x: Bernoulli| x.cdf(arg);
        test_case(0.3, 1., cdf(1));
    }

    #[test]
    fn test_sf_upper_bound() {
        let sf = |arg: u64| move |x: Bernoulli| x.sf(arg);
        test_case(0.3, 0., sf(1));
    }

    #[test]
    fn test_cdf() {
        let cdf = |arg: u64| move |x: Bernoulli| x.cdf(arg);
        test_case(0.0, 1.0, cdf(0));
        test_case(0.0, 1.0, cdf(1));
        test_almost(0.3, 0.7, 1e-15, cdf(0));
        test_almost(0.7, 0.3, 1e-15, cdf(0));
    }

    #[test]
    fn test_sf() {
        let sf = |arg: u64| move |x: Bernoulli| x.sf(arg);
        test_case(0.0, 0.0, sf(0));
        test_case(0.0, 0.0, sf(1));
        test_almost(0.3, 0.3, 1e-15, sf(0));
        test_almost(0.7, 0.7, 1e-15, sf(0));
    }
}
