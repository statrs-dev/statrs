use crate::distribution::{Discrete, Univariate};
use crate::function::{factorial, gamma};
use crate::statistics::*;
use crate::{Result, StatsError};
use rand::distributions::Distribution;
use rand::Rng;
use std::f64;
use std::u64;

/// Implements the [Poisson](https://en.wikipedia.org/wiki/Poisson_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Poisson, Discrete};
/// use statrs::statistics::Mean;
/// use statrs::prec;
///
/// let n = Poisson::new(1.0).unwrap();
/// assert_eq!(n.mean(), 1.0);
/// assert!(prec::almost_eq(n.pmf(1), 0.367879441171442, 1e-15));
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Poisson {
    lambda: f64,
}

impl Poisson {
    /// Constructs a new poisson distribution with a rate (λ)
    /// of `lambda`
    ///
    /// # Errors
    ///
    /// Returns an error if `lambda` is `NaN` or `lambda <= 0.0`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Poisson;
    ///
    /// let mut result = Poisson::new(1.0);
    /// assert!(result.is_ok());
    ///
    /// result = Poisson::new(0.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(lambda: f64) -> Result<Poisson> {
        if lambda.is_nan() || lambda <= 0.0 {
            Err(StatsError::BadParams)
        } else {
            Ok(Poisson { lambda })
        }
    }

    /// Returns the rate (λ) of the poisson distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Poisson;
    ///
    /// let n = Poisson::new(1.0).unwrap();
    /// assert_eq!(n.lambda(), 1.0);
    /// ```
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

impl Distribution<f64> for Poisson {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> f64 {
        sample_unchecked(r, self.lambda)
    }
}

impl Univariate<u64, f64> for Poisson {
    /// Calculates the cumulative distribution function for the poisson
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 1 - P(x + 1, λ)
    /// ```
    ///
    /// where `λ` is the rate and `P` is the lower regularized gamma function
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else if x.is_infinite() {
            1.0
        } else {
            1.0 - gamma::gamma_lr(x.floor() + 1.0, self.lambda)
        }
    }
}

impl Min<u64> for Poisson {
    /// Returns the minimum value in the domain of the poisson distribution
    /// representable by a 64-bit integer
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 0
    /// ```
    fn min(&self) -> u64 {
        0
    }
}

impl Max<u64> for Poisson {
    /// Returns the maximum value in the domain of the poisson distribution
    /// representable by a 64-bit integer
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 2^63 - 1
    /// ```
    fn max(&self) -> u64 {
        u64::MAX
    }
}

impl Mean<f64> for Poisson {
    /// Returns the mean of the poisson distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// λ
    /// ```
    ///
    /// where `λ` is the rate
    fn mean(&self) -> f64 {
        self.lambda
    }
}

impl Variance<f64> for Poisson {
    /// Returns the variance of the poisson distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// λ
    /// ```
    ///
    /// where `λ` is the rate
    fn variance(&self) -> f64 {
        self.lambda
    }

    /// Returns the standard deviation of the poisson distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// sqrt(λ)
    /// ```
    ///
    /// where `λ` is the rate
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

impl Entropy<f64> for Poisson {
    /// Returns the entropy of the poisson distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (1 / 2) * ln(2πeλ) - 1 / (12λ) - 1 / (24λ^2) - 19 / (360λ^3)
    /// ```
    ///
    /// where `λ` is the rate
    fn entropy(&self) -> f64 {
        0.5 * (2.0 * f64::consts::PI * f64::consts::E * self.lambda).ln()
            - 1.0 / (12.0 * self.lambda)
            - 1.0 / (24.0 * self.lambda * self.lambda)
            - 19.0 / (360.0 * self.lambda * self.lambda * self.lambda)
    }
}

impl Skewness<f64> for Poisson {
    /// Returns the skewness of the poisson distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// λ^(-1/2)
    /// ```
    ///
    /// where `λ` is the rate
    fn skewness(&self) -> f64 {
        1.0 / self.lambda.sqrt()
    }
}

impl Median<f64> for Poisson {
    /// Returns the median of the poisson distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// floor(λ + 1 / 3 - 0.02 / λ)
    /// ```
    ///
    /// where `λ` is the rate
    fn median(&self) -> f64 {
        (self.lambda + 1.0 / 3.0 - 0.02 / self.lambda).floor()
    }
}

impl Mode<u64> for Poisson {
    /// Returns the mode of the poisson distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// floor(λ)
    /// ```
    ///
    /// where `λ` is the rate
    fn mode(&self) -> u64 {
        self.lambda.floor() as u64
    }
}

impl Discrete<u64, f64> for Poisson {
    /// Calculates the probability mass function for the poisson distribution at
    /// `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (λ^k * e^(-λ)) / x!
    /// ```
    ///
    /// where `λ` is the rate
    fn pmf(&self, x: u64) -> f64 {
        (-self.lambda + x as f64 * self.lambda.ln() - factorial::ln_factorial(x as u64)).exp()
    }

    /// Calculates the log probability mass function for the poisson
    /// distribution at
    /// `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln((λ^k * e^(-λ)) / x!)
    /// ```
    ///
    /// where `λ` is the rate
    fn ln_pmf(&self, x: u64) -> f64 {
        -self.lambda + x as f64 * self.lambda.ln() - factorial::ln_factorial(x as u64)
    }
}

/// Generates one sample from the Poisson distribution either by
/// Knuth's method if lambda < 30.0 or Rejection method PA by
/// A. C. Atkinson from the Journal of the Royal Statistical Society
/// Series C (Applied Statistics) Vol. 28 No. 1. (1979) pp. 29 - 35
/// otherwise
pub fn sample_unchecked<R: Rng + ?Sized>(rng: &mut R, lambda: f64) -> f64 {
    if lambda < 30.0 {
        let limit = (-lambda).exp();
        let mut count = 0.0;
        let mut product: f64 = rng.gen();
        while product >= limit {
            count += 1.0;
            product *= rng.gen::<f64>();
        }
        count
    } else {
        let c = 0.767 - 3.36 / lambda;
        let beta = f64::consts::PI / (3.0 * lambda).sqrt();
        let alpha = beta * lambda;
        let k = c.ln() - lambda - beta.ln();

        loop {
            let u: f64 = rng.gen();
            let x = (alpha - ((1.0 - u) / u).ln()) / beta;
            let n = (x + 0.5).floor();
            if n < 0.0 {
                continue;
            }

            let v: f64 = rng.gen();
            let y = alpha - beta * x;
            let temp = 1.0 + y.exp();
            let lhs = y + (v / (temp * temp)).ln();
            let rhs = k + n * lambda.ln() - factorial::ln_factorial(n as u64);
            if lhs <= rhs {
                return n;
            }
        }
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod test {
    use std::fmt::Debug;
    use std::f64;
    use std::u64;
    use crate::statistics::*;
    use crate::distribution::{Univariate, Discrete, Poisson};
    use crate::distribution::internal::*;

    fn try_create(lambda: f64) -> Poisson {
        let n = Poisson::new(lambda);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(lambda: f64) {
        let n = try_create(lambda);
        assert_eq!(lambda, n.lambda());
    }

    fn bad_create_case(lambda: f64) {
        let n = Poisson::new(lambda);
        assert!(n.is_err());
    }

    fn get_value<T, F>(lambda: f64, eval: F) -> T
        where T: PartialEq + Debug,
              F: Fn(Poisson) -> T
    {
        let n = try_create(lambda);
        eval(n)
    }

    fn test_case<T, F>(lambda: f64, expected: T, eval: F)
        where T: PartialEq + Debug,
              F: Fn(Poisson) -> T
    {
        let x = get_value(lambda, eval);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(lambda: f64, expected: f64, acc: f64, eval: F)
        where F: Fn(Poisson) -> f64
    {
        let x = get_value(lambda, eval);
        assert_almost_eq!(expected, x, acc);
    }

    #[test]
    fn test_create() {
        create_case(1.5);
        create_case(5.4);
        create_case(10.8);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(f64::NAN);
        bad_create_case(-1.5);
        bad_create_case(0.0);
    }

    #[test]
    fn test_mean() {
        test_case(1.5, 1.5, |x| x.mean());
        test_case(5.4, 5.4, |x| x.mean());
        test_case(10.8, 10.8, |x| x.mean());
    }

    #[test]
    fn test_variance() {
        test_case(1.5, 1.5, |x| x.variance());
        test_case(5.4, 5.4, |x| x.variance());
        test_case(10.8, 10.8, |x| x.variance());
    }

    #[test]
    fn test_std_dev() {
        test_case(1.5, (1.5f64).sqrt(), |x| x.std_dev());
        test_case(5.4, (5.4f64).sqrt(), |x| x.std_dev());
        test_case(10.8, (10.8f64).sqrt(), |x| x.std_dev());
    }

    #[test]
    fn test_entropy() {
        test_almost(1.5, 1.531959153102376331946, 1e-15, |x| x.entropy());
        test_almost(5.4, 2.244941839577643504608, 1e-15, |x| x.entropy());
        test_case(10.8, 2.600596429676975222694, |x| x.entropy());
    }

    #[test]
    fn test_skewness() {
        test_almost(1.5, 0.8164965809277260327324, 1e-15, |x| x.skewness());
        test_almost(5.4, 0.4303314829119352094644, 1e-16, |x| x.skewness());
        test_almost(10.8, 0.3042903097250922852539, 1e-16, |x| x.skewness());
    }

    #[test]
    fn test_median() {
        test_case(1.5, 1.0, |x| x.median());
        test_case(5.4, 5.0, |x| x.median());
        test_case(10.8, 11.0, |x| x.median());
    }

    #[test]
    fn test_mode() {
        test_case(1.5, 1, |x| x.mode());
        test_case(5.4, 5, |x| x.mode());
        test_case(10.8, 10, |x| x.mode());
    }

    #[test]
    fn test_min_max() {
        test_case(1.5, 0, |x| x.min());
        test_case(5.4, 0, |x| x.min());
        test_case(10.8, 0, |x| x.min());
        test_case(1.5, u64::MAX, |x| x.max());
        test_case(5.4, u64::MAX, |x| x.max());
        test_case(10.8, u64::MAX, |x| x.max());
    }

    #[test]
    fn test_pmf() {
        test_almost(1.5, 0.334695240222645000000000000000, 1e-15, |x| x.pmf(1));
        test_almost(1.5, 0.000003545747740570180000000000, 1e-20, |x| x.pmf(10));
        test_almost(1.5, 0.000000000000000304971208961018, 1e-30, |x| x.pmf(20));
        test_almost(5.4, 0.024389537090108400000000000000, 1e-17, |x| x.pmf(1));
        test_almost(5.4, 0.026241240591792300000000000000, 1e-16, |x| x.pmf(10));
        test_almost(5.4, 0.000000825202200316548000000000, 1e-20, |x| x.pmf(20));
        test_almost(10.8, 0.000220314636840657000000000000, 1e-18, |x| x.pmf(1));
        test_almost(10.8, 0.121365183659420000000000000000, 1e-15, |x| x.pmf(10));
        test_almost(10.8, 0.003908139778574110000000000000, 1e-16, |x| x.pmf(20));
    }

    #[test]
    fn test_ln_pmf() {
        test_almost(1.5, -1.09453489189183485135413967177, 1e-15, |x| x.ln_pmf(1));
        test_almost(1.5, -12.5497614919938728510400000000, 1e-14, |x| x.ln_pmf(10));
        test_almost(1.5, -35.7263142985901000000000000000, 1e-13, |x| x.ln_pmf(20));
        test_case(5.4, -3.71360104642977159156055355910, |x| x.ln_pmf(1));
        test_almost(5.4, -3.64042303737322774736223038530, 1e-15, |x| x.ln_pmf(10));
        test_almost(5.4, -14.0076373893489089949388000000, 1e-14, |x| x.ln_pmf(20));
        test_almost(10.8, -8.42045386586982559781714423000, 1e-14, |x| x.ln_pmf(1));
        test_almost(10.8, -2.10895123177378079525424989992, 1e-14, |x| x.ln_pmf(10));
        test_almost(10.8, -5.54469377815000936289610059500, 1e-14, |x| x.ln_pmf(20));
    }

    #[test]
    fn test_cdf() {
        test_almost(1.5, 0.5578254003710750000000, 1e-15, |x| x.cdf(1.0));
        test_almost(1.5, 0.9999994482467640000000, 1e-15, |x| x.cdf(10.0));
        test_case(1.5, 1.0, |x| x.cdf(20.0));
        test_almost(5.4, 0.0289061180327211000000, 1e-16, |x| x.cdf(1.0));
        test_almost(5.4, 0.9774863006897650000000, 1e-15, |x| x.cdf(10.0));
        test_almost(5.4, 0.9999997199928290000000, 1e-15, |x| x.cdf(20.0));
        test_almost(10.8, 0.0002407141402518290000, 1e-16, |x| x.cdf(1.0));
        test_almost(10.8, 0.4839692359955690000000, 1e-15, |x| x.cdf(10.0));
        test_almost(10.8, 0.9961800769608090000000, 1e-15, |x| x.cdf(20.0));
    }

    #[test]
    fn test_neg_cdf() {
        test_case(1.5, 0.0, |x| x.cdf(-1.0));
    }

    #[test]
    fn test_discrete() {
        test::check_discrete_distribution(&try_create(0.3), 10);
        test::check_discrete_distribution(&try_create(4.5), 30);
    }
}
