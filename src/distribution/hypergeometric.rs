use distribution::{Discrete, Univariate};
use function::factorial;
use rand::distributions::Distribution;
use rand::Rng;
use statistics::*;
use std::cmp;
use std::f64;
use {Result, StatsError};

/// Implements the
/// [Hypergeometric](http://en.wikipedia.org/wiki/Hypergeometric_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Hypergeometric {
    population: u64,
    successes: u64,
    draws: u64,
}

impl Hypergeometric {
    /// Constructs a new hypergeometric distribution
    /// with a population (N) of `population`, number
    /// of successes (K) of `successes`, and number of draws
    /// (n) of `draws`
    ///
    /// # Errors
    ///
    /// If `successes > population` or `draws > population`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Hypergeometric;
    ///
    /// let mut result = Hypergeometric::new(2, 2, 2);
    /// assert!(result.is_ok());
    ///
    /// result = Hypergeometric::new(2, 3, 2);
    /// assert!(result.is_err());
    /// ```
    pub fn new(population: u64, successes: u64, draws: u64) -> Result<Hypergeometric> {
        if successes > population || draws > population {
            Err(StatsError::BadParams)
        } else {
            Ok(Hypergeometric {
                population: population,
                successes: successes,
                draws: draws,
            })
        }
    }

    /// Returns the population size of the hypergeometric
    /// distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Hypergeometric;
    ///
    /// let n = Hypergeometric::new(10, 5, 3).unwrap();
    /// assert_eq!(n.population(), 10);
    /// ```
    pub fn population(&self) -> u64 {
        self.population
    }

    /// Returns the number of observed successes of the hypergeometric
    /// distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Hypergeometric;
    ///
    /// let n = Hypergeometric::new(10, 5, 3).unwrap();
    /// assert_eq!(n.successes(), 5);
    /// ```
    pub fn successes(&self) -> u64 {
        self.successes
    }

    /// Returns the number of draws of the hypergeometric
    /// distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Hypergeometric;
    ///
    /// let n = Hypergeometric::new(10, 5, 3).unwrap();
    /// assert_eq!(n.draws(), 3);
    /// ```
    pub fn draws(&self) -> u64 {
        self.draws
    }

    /// Returns population, successes, and draws in that order
    /// as a tuple of doubles
    fn values_f64(&self) -> (f64, f64, f64) {
        (
            self.population as f64,
            self.successes as f64,
            self.draws as f64,
        )
    }
}

impl Distribution<f64> for Hypergeometric {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> f64 {
        let mut population = self.population as f64;
        let mut successes = self.successes as f64;
        let mut draws = self.draws;
        let mut x = 0.0;
        loop {
            let p = successes / population;
            let next: f64 = r.gen();
            if next < p {
                x += 1.0;
                successes -= 1.0;
            }
            population -= 1.0;
            draws -= 1;
            if draws == 0 {
                break;
            }
        }
        x
    }
}

impl Univariate<u64, f64> for Hypergeometric {
    /// Calculates the cumulative distribution function for the hypergeometric
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 1 - ((n choose k+1) * (N-n choose K-k-1)) / (N choose K) * 3_F_2(1,
    /// k+1-K, k+1-n; k+2, N+k+2-K-n; 1)
    /// ```
    ///
    // where `N` is population, `K` is successes, `n` is draws,
    /// and `p_F_q` is the [generalized hypergeometric
    /// function](https://en.wikipedia.
    /// org/wiki/Generalized_hypergeometric_function)
    fn cdf(&self, x: f64) -> f64 {
        if x < self.min() as f64 {
            0.0
        } else if x >= self.max() as f64 {
            1.0
        } else {
            let k = x.floor() as u64;
            let ln_denom = factorial::ln_binomial(self.population, self.draws);
            (0..k + 1).fold(0.0, |acc, i| {
                acc + (factorial::ln_binomial(self.successes, i)
                    + factorial::ln_binomial(self.population - self.successes, self.draws - i)
                    - ln_denom)
                    .exp()
            })
        }
    }
}

impl Min<u64> for Hypergeometric {
    /// Returns the minimum value in the domain of the
    /// hypergeometric distribution representable by a 64-bit
    /// integer
    ///
    /// # Formula
    ///
    /// ```ignore
    /// max(0, n + K - N)
    /// ```
    ///
    /// where `N` is population, `K` is successes, and `n` is draws
    fn min(&self) -> u64 {
        (self.draws + self.successes).saturating_sub(self.population)
    }
}

impl Max<u64> for Hypergeometric {
    /// Returns the maximum value in the domain of the
    /// hypergeometric distribution representable by a 64-bit
    /// integer
    ///
    /// # Formula
    ///
    /// ```ignore
    /// min(K, n)
    /// ```
    ///
    /// where `K` is successes and `n` is draws
    fn max(&self) -> u64 {
        cmp::min(self.successes, self.draws)
    }
}

impl Mean<f64> for Hypergeometric {
    /// Returns the mean of the hypergeometric distribution
    ///
    /// # Panics
    ///
    /// If `N` is `0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// K * n / N
    /// ```
    ///
    /// where `N` is population, `K` is successes, and `n` is draws
    fn mean(&self) -> f64 {
        self.checked_mean().unwrap()
    }
}

impl CheckedMean<f64> for Hypergeometric {
    /// Returns the mean of the hypergeometric distribution
    ///
    /// # Errors
    ///
    /// If `N` is `0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// K * n / N
    /// ```
    ///
    /// where `N` is population, `K` is successes, and `n` is draws
    fn checked_mean(&self) -> Result<f64> {
        if self.population == 0 {
            Err(StatsError::ArgGt("population", 0.0))
        } else {
            Ok(self.successes as f64 * self.draws as f64 / self.population as f64)
        }
    }
}

impl Variance<f64> for Hypergeometric {
    /// Returns the variance of the hypergeometric distribution
    ///
    /// # Panics
    ///
    /// If `N <= 1`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// n * (K / N) * ((N - K) / N) * ((N - n) / (N - 1))
    /// ```
    ///
    /// where `N` is population, `K` is successes, and `n` is draws
    fn variance(&self) -> f64 {
        self.checked_variance().unwrap()
    }

    /// Returns the standard deviation of the hypergeometric distribution
    ///
    /// # Panics
    ///
    /// If `N <= 1`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// sqrt(n * (K / N) * ((N - K) / N) * ((N - n) / (N - 1)))
    /// ```
    ///
    /// where `N` is population, `K` is successes, and `n` is draws
    fn std_dev(&self) -> f64 {
        self.checked_std_dev().unwrap()
    }
}

impl CheckedVariance<f64> for Hypergeometric {
    /// Returns the variance of the hypergeometric distribution
    ///
    /// # Errors
    ///
    /// If `N <= 1`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// n * (K / N) * ((N - K) / N) * ((N - n) / (N - 1))
    /// ```
    ///
    /// where `N` is population, `K` is successes, and `n` is draws
    fn checked_variance(&self) -> Result<f64> {
        if self.population <= 1 {
            Err(StatsError::ArgGt("population", 1.0))
        } else {
            let (population, successes, draws) = self.values_f64();
            let val = draws * successes * (population - draws) * (population - successes)
                / (population * population * (population - 1.0));
            Ok(val)
        }
    }

    /// Returns the standard deviation of the hypergeometric distribution
    ///
    /// # Errors
    ///
    /// If `N <= 1`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// sqrt(n * (K / N) * ((N - K) / N) * ((N - n) / (N - 1)))
    /// ```
    ///
    /// where `N` is population, `K` is successes, and `n` is draws
    fn checked_std_dev(&self) -> Result<f64> {
        self.checked_variance().map(|x| x.sqrt())
    }
}

impl Skewness<f64> for Hypergeometric {
    /// Returns the skewness of the hypergeometric distribution
    ///
    /// # Panics
    ///
    /// If `N <= 2`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ((N - 2K) * (N - 1)^(1 / 2) * (N - 2n)) / ([n * K * (N - K) * (N -
    /// n)]^(1 / 2) * (N - 2))
    /// ```
    ///
    /// where `N` is population, `K` is successes, and `n` is draws
    fn skewness(&self) -> f64 {
        self.checked_skewness().unwrap()
    }
}

impl CheckedSkewness<f64> for Hypergeometric {
    /// Returns the skewness of the hypergeometric distribution
    ///
    /// # Errors
    ///
    /// If `N <= 2`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ((N - 2K) * (N - 1)^(1 / 2) * (N - 2n)) / ([n * K * (N - K) * (N -
    /// n)]^(1 / 2) * (N - 2))
    /// ```
    ///
    /// where `N` is population, `K` is successes, and `n` is draws
    fn checked_skewness(&self) -> Result<f64> {
        if self.population <= 2 {
            Err(StatsError::ArgGt("population", 2.0))
        } else {
            let (population, successes, draws) = self.values_f64();
            let val = (population - 1.0).sqrt()
                * (population - 2.0 * draws)
                * (population - 2.0 * successes)
                / ((draws * successes * (population - successes) * (population - draws)).sqrt()
                    * (population - 2.0));
            Ok(val)
        }
    }
}

impl Mode<u64> for Hypergeometric {
    /// Returns the mode of the hypergeometric distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// floor((n + 1) * (k + 1) / (N + 2))
    /// ```
    ///
    /// where `N` is population, `K` is successes, and `n` is draws
    fn mode(&self) -> u64 {
        ((self.draws + 1) * (self.successes + 1) / (self.population + 2)) as u64
    }
}

impl Discrete<u64, f64> for Hypergeometric {
    /// Calculates the probability mass function for the hypergeometric
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (K choose x) * (N-K choose n-x) / (N choose n)
    /// ```
    ///
    /// where `N` is population, `K` is successes, and `n` is draws
    fn pmf(&self, x: u64) -> f64 {
        if x > self.draws {
            0.0
        } else {
            factorial::binomial(self.successes, x)
                * factorial::binomial(self.population - self.successes, self.draws - x)
                / factorial::binomial(self.population, self.draws)
        }
    }

    /// Calculates the log probability mass function for the hypergeometric
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln((K choose x) * (N-K choose n-x) / (N choose n))
    /// ```
    ///
    /// where `N` is population, `K` is successes, and `n` is draws
    fn ln_pmf(&self, x: u64) -> f64 {
        factorial::ln_binomial(self.successes, x)
            + factorial::ln_binomial(self.population - self.successes, self.draws - x)
            - factorial::ln_binomial(self.population, self.draws)
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use std::f64;
    use std::fmt::Debug;
    use statistics::*;
    use distribution::{Univariate, Discrete, Hypergeometric};
    use distribution::internal::*;

    fn try_create(population: u64, successes: u64, draws: u64) -> Hypergeometric {
        let n = Hypergeometric::new(population, successes, draws);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(population: u64, successes: u64, draws: u64) {
        let n = try_create(population, successes, draws);
        assert_eq!(population, n.population());
        assert_eq!(successes, n.successes());
        assert_eq!(draws, n.draws());
    }

    fn bad_create_case(population: u64, successes: u64, draws: u64) {
        let n = Hypergeometric::new(population, successes, draws);
        assert!(n.is_err());
    }

    fn get_value<T, F>(population: u64, successes: u64, draws: u64, eval: F) -> T
        where T: PartialEq + Debug,
              F: Fn(Hypergeometric) -> T
    {
        let n = try_create(population, successes, draws);
        eval(n)
    }

    fn test_case<T, F>(population: u64, successes: u64, draws: u64, expected: T, eval: F)
        where T: PartialEq + Debug,
              F: Fn(Hypergeometric) -> T
    {
        let x = get_value(population, successes, draws, eval);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(population: u64, successes: u64, draws: u64, expected: f64, acc: f64, eval: F)
        where F: Fn(Hypergeometric) -> f64
    {
        let x = get_value(population, successes, draws, eval);
        assert_almost_eq!(expected, x, acc);
    }

    #[test]
    fn test_create() {
        create_case(0, 0, 0);
        create_case(1, 1, 1,);
        create_case(2, 1, 1);
        create_case(2, 2, 2);
        create_case(10, 1, 1);
        create_case(10, 5, 3);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(2, 3, 2);
        bad_create_case(10, 5, 20);
        bad_create_case(0, 1, 1);
    }

    #[test]
    fn test_mean() {
        test_case(1, 1, 1, 1.0, |x| x.mean());
        test_case(2, 1, 1, 0.5, |x| x.mean());
        test_case(2, 2, 2, 2.0, |x| x.mean());
        test_case(10, 1, 1, 0.1, |x| x.mean());
        test_case(10, 5, 3, 15.0 / 10.0, |x| x.mean());
    }

    #[test]
    #[should_panic]
    fn test_mean_with_population_0() {
        get_value(0, 0, 0, |x| x.mean());
    }

    #[test]
    fn test_checked_mean_with_population_0() {
        let n = try_create(0, 0, 0);
        assert!(n.checked_mean().is_err());
    }

    #[test]
    fn test_variance() {
        test_case(2, 1, 1, 0.25, |x| x.variance());
        test_case(2, 2, 2, 0.0, |x| x.variance());
        test_case(10, 1, 1, 81.0 / 900.0, |x| x.variance());
        test_case(10, 5, 3, 525.0 / 900.0, |x| x.variance());
    }

    #[test]
    #[should_panic]
    fn test_variance_with_pop_lte_1() {
        get_value(1, 1, 1, |x| x.variance());
    }

    #[test]
    fn test_checked_variance_with_pop_lte_1() {
        let n = try_create(1, 1, 1);
        assert!(n.checked_variance().is_err());
    }

    #[test]
    fn test_std_dev() {
        test_case(2, 1, 1, 0.25f64.sqrt(), |x| x.std_dev());
        test_case(2, 2, 2, 0.0, |x| x.std_dev());
        test_case(10, 1, 1, (81f64 / 900.0).sqrt(), |x| x.std_dev());
        test_case(10, 5, 3, (525f64 / 900.0).sqrt(), |x| x.std_dev());
    }

    #[test]
    #[should_panic]
    fn test_std_dev_with_pop_lte_1() {
        get_value(1, 1, 1, |x| x.std_dev());
    }

    #[test]
    fn test_checked_std_dev_with_pop_lte_1() {
        let n = try_create(1, 1, 1);
        assert!(n.checked_std_dev().is_err());
    }

    #[test]
    fn test_skewness() {
        test_case(10, 1, 1, 8.0 / 3.0, |x| x.skewness());
        test_case(10, 5, 3, 0.0, |x| x.skewness());
    }

    #[test]
    #[should_panic]
    fn test_skewness_with_pop_lte_2() {
        get_value(2, 2, 2, |x| x.skewness());
    }

    #[test]
    fn test_checked_skewness_with_pop_lte_2() {
        let n = try_create(2, 2, 2);
        assert!(n.checked_skewness().is_err());
    }

    #[test]
    fn test_mode() {
        test_case(0, 0, 0, 0, |x| x.mode());
        test_case(1, 1, 1, 1, |x| x.mode());
        test_case(2, 1, 1, 1, |x| x.mode());
        test_case(2, 2, 2, 2, |x| x.mode());
        test_case(10, 1, 1, 0, |x| x.mode());
        test_case(10, 5, 3, 2, |x| x.mode());
    }

    #[test]
    fn test_min() {
        test_case(0, 0, 0, 0, |x| x.min());
        test_case(1, 1, 1, 1, |x| x.min());
        test_case(2, 1, 1, 0, |x| x.min());
        test_case(2, 2, 2, 2, |x| x.min());
        test_case(10, 1, 1, 0, |x| x.min());
        test_case(10, 5, 3, 0, |x| x.min());
    }

    #[test]
    fn test_max() {
        test_case(0, 0, 0, 0, |x| x.max());
        test_case(1, 1, 1, 1, |x| x.max());
        test_case(2, 1, 1, 1, |x| x.max());
        test_case(2, 2, 2, 2, |x| x.max());
        test_case(10, 1, 1, 1, |x| x.max());
        test_case(10, 5, 3, 3, |x| x.max());
    }

    #[test]
    fn test_pmf() {
        test_case(0, 0, 0, 1.0, |x| x.pmf(0));
        test_case(1, 1, 1, 1.0, |x| x.pmf(1));
        test_case(2, 1, 1, 0.5, |x| x.pmf(0));
        test_case(2, 1, 1, 0.5, |x| x.pmf(1));
        test_case(2, 2, 2, 1.0, |x| x.pmf(2));
        test_case(10, 1, 1, 0.9, |x| x.pmf(0));
        test_case(10, 1, 1, 0.1, |x| x.pmf(1));
        test_case(10, 5, 3, 0.41666666666666666667, |x| x.pmf(1));
        test_case(10, 5, 3, 0.083333333333333333333, |x| x.pmf(3));
    }

    #[test]
    fn test_ln_pmf() {
        test_case(0, 0, 0, 0.0, |x| x.ln_pmf(0));
        test_case(1, 1, 1, 0.0, |x| x.ln_pmf(1));
        test_case(2, 1, 1, -0.6931471805599453094172, |x| x.ln_pmf(0));
        test_case(2, 1, 1, -0.6931471805599453094172, |x| x.ln_pmf(1));
        test_case(2, 2, 2, 0.0, |x| x.ln_pmf(2));
        test_almost(10, 1, 1, -0.1053605156578263012275, 1e-14, |x| x.ln_pmf(0));
        test_almost(10, 1, 1, -2.302585092994045684018, 1e-14, |x| x.ln_pmf(1));
        test_almost(10, 5, 3, -0.875468737353899935621, 1e-14, |x| x.ln_pmf(1));
        test_almost(10, 5, 3, -2.484906649788000310234, 1e-14, |x| x.ln_pmf(3));
    }

    #[test]
    fn test_cdf() {
        test_case(2, 1, 1, 0.5, |x| x.cdf(0.3));
        test_almost(10, 1, 1, 0.9, 1e-14, |x| x.cdf(0.3));
        test_almost(10, 5, 3, 0.5, 1e-15, |x| x.cdf(1.1));
        test_almost(10, 5, 3, 11.0 / 12.0, 1e-14, |x| x.cdf(2.0));
        test_almost(10000, 2, 9800, 199.0 / 499950.0, 1e-14, |x| x.cdf(0.0));
        test_almost(10000, 2, 9800, 199.0 / 499950.0, 1e-14, |x| x.cdf(0.5));
        test_almost(10000, 2, 9800, 19799.0 / 499950.0, 1e-12, |x| x.cdf(1.5));
    }

    #[test]
    fn test_cdf_arg_too_big() {
        test_case(0, 0, 0, 1.0, |x| x.cdf(0.5));
    }

    #[test]
    fn test_cdf_arg_too_small() {
        test_case(2, 2, 2, 0.0, |x| x.cdf(0.0));
    }

    #[test]
    fn test_discrete() {
        test::check_discrete_distribution(&try_create(5, 4, 3), 4);
        test::check_discrete_distribution(&try_create(3, 2, 1), 2);
    }
}
