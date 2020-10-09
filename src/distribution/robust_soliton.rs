use crate::distribution::{Soliton, IdealSoliton};
use crate::statistics::*;
use crate::{Result, StatsError};
use rand::distributions::Distribution;
use rand::Rng;
use std::f64;

/// Implements the [Discrete
/// Uniform](https://en.wikipedia.org/wiki/Discrete_uniform_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{DiscreteUniform, Discrete};
/// use statrs::statistics::Mean;
///
/// let n = DiscreteUniform::new(0, 5).unwrap();
/// assert_eq!(n.mean(), 2.5);
/// assert_eq!(n.pmf(3), 1.0 / 6.0);
/// ```

#[derive(Debug, Clone, PartialEq)]
pub struct RobustSoliton {
    min: i64,
    max: i64,
    spike: Option<i64>,
    cumulative_probability_table: Vec<f64>,
    ripple: Ripple,
    fail_probability: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RobustSolitonDistribution {
    ripple: Ripple,
    fail_probability: f64,
}

impl RobustSoliton {
    /// Constructs a new discrete uniform distribution with a minimum value
    /// of `min` and a maximum value of `max`.
    ///
    /// # Errors
    ///
    /// Returns an error if `max < min`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::DiscreteUniform;
    ///
    /// let mut result = DiscreteUniform::new(0, 5);
    /// assert!(result.is_ok());
    ///
    /// result = DiscreteUniform::new(5, 0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(max: i64, heuristic: bool, ripple: f64, fail_probability: f64) -> Result<RobustSoliton> {
        if max < 1 {
            Err(StatsError::BadParams)
        } else {
            let pmf_table: Vec<f64> = Vec::with_capacity(max as usize);
            let mut rs = RobustSoliton {
                min: 1,
                max,
                spike: None,
                cumulative_probability_table: pmf_table,
                ripple: match heuristic {
                    false => Ripple::Fixed(ripple),
                    true => Ripple::Heuristic(ripple),
                },
                fail_probability,
            };

            let mut cumulative_probability = 0.0;
            for i in 1..(max + 1) {
                cumulative_probability += rs.soliton(i);
                rs.cumulative_probability_table.push(cumulative_probability);
            }
            Ok(rs)
        }
    }

    pub fn query_table<R: Rng + ?Sized>(&self, r: &mut R) -> Result<i64> {
        let n = self.sample(r);

        for i in 1..(self.max+1) {
            if n < self.cumulative_probability_table[i as usize] {
                return Ok(i);
            }
        }
        Err(StatsError::ContainerExpectedSum("Elements in probability table expected to sum to 1 but didn't! Limit is", self.max as f64))
    }

    pub fn query_table_int<R: Rng + ?Sized>(&self, r: &mut R) -> Result<usize> {
        let n = r.gen_range(self.min as usize, self.max as usize);
        Ok(n)
    }

}

#[derive(PartialEq, Clone, Debug)]
enum Ripple {
    Fixed(f64),
    Heuristic(f64)
}

impl Ripple {
    fn size(&self, max: i64, fail_probability: f64) -> f64 {
        match self {
            &Ripple::Fixed(n) => n,
            &Ripple::Heuristic(n) => {
                n * (max as f64 / fail_probability).ln() * (max as f64).sqrt()
            }
        }
    }
}

impl Distribution<f64> for RobustSoliton {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> f64 {
        r.gen_range(0, 1) as f64
    }
}

impl Min<i64> for RobustSoliton {
    /// Returns the minimum value in the domain of the robust soliton
    /// distribution
    ///
    /// # Remarks
    ///
    /// This is the same value as the minimum passed into the constructor
    fn min(&self) -> i64 {
        self.min
    }
}

impl Max<i64> for RobustSoliton {
    /// Returns the maximum value in the domain of the robust soliton
    /// distribution
    ///
    /// # Remarks
    ///
    /// This is the same value as the maximum passed into the constructor
    fn max(&self) -> i64 {
        self.max
    }
}

impl Mean<f64> for RobustSoliton {
    /// Returns the mean of the robust soliton distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// sum(Xn)/n
    /// ```
    fn mean(&self) -> f64 {
        let sum: f64 = Iterator::sum(self.cumulative_probability_table.iter());
        let mean = sum / self.cumulative_probability_table.len() as f64;
        mean
    }
}

impl Variance<f64> for RobustSoliton {
    /// Returns the variance of the robust soliton distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// sum((Xn - mean)^2) / n  n from 0..max
    /// ```
    fn variance(&self) -> f64 {
        let mut sumsq = 0.0;
        let mean = self.mean();
        for i in &self.cumulative_probability_table {
            sumsq += (i - mean)*(i - mean);
        }
        let var = sumsq / self.cumulative_probability_table.len() as f64;
        var
    }

    /// Returns the standard deviation of the robust soliton distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// sqrt(variance)
    /// ```
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

impl Soliton<i64, f64> for RobustSoliton {
    /// Calculates the ideal soliton for the
    /// discrete uniform distribution at `x`
    ///
    /// # Remarks
    ///
    /// Returns `0.0` if `x` is not in `[min, max]`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ```
    fn soliton(&self, x: i64) -> f64 {
        if x > 1 && x < self.max {
            let ideal_sol = IdealSoliton::new(self.max).unwrap();
            let ideal_val = ideal_sol.soliton(x);
            //println!("I: {:?} | A: {:?} | N: {:?}", ideal_val, self.additive_probability(x), self.normalization_factor());
            (ideal_val + self.additive_probability(x))/(self.normalization_factor())
        } else if x == 1 {
            1.0
        } else {
            // Point must be in range (0, limit]
            0.0
        }
    }

    fn normalization_factor(&self) -> f64 {
        let mut normalization_factor: f64 = 0.0;
        let ideal_sol = IdealSoliton::new(self.max).unwrap();
        for i in 1..(self.max+1) {
            normalization_factor += ideal_sol.soliton(i);
            normalization_factor += self.additive_probability(i);
        }
        normalization_factor
    }

    fn additive_probability(&self, val: i64) -> f64 {
        let ripple_size = self.ripple.size(self.max, self.fail_probability);
        let swap = self.max as f64 / ripple_size;
        if val == 0 || val > self.max {
            panic!("Point must be in range (0,max]. Given {} - {}", val, self.max);
        } else if (val as f64) < swap {
            ripple_size / ((val * self.max) as f64)
        } else if (val as f64) == swap {
            (ripple_size * (ripple_size / self.fail_probability).ln()) / (self.max as f64)
        } else {
            0.0
        }
    }


}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use std::fmt::Debug;
    use std::f64;
    use crate::statistics::*;
    use crate::distribution::{Soliton, RobustSoliton};
    fn try_create(max: i64, heuristic: bool, ripple: f64, fail_probability: f64) -> RobustSoliton {
        let n = RobustSoliton::new(max, heuristic, ripple, fail_probability);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(max: i64, heuristic: bool, ripple: f64, fail_probability: f64) {
        let n = try_create(max, heuristic, ripple, fail_probability);
        assert_eq!(1, n.min());
        assert_eq!(max, n.max());
    }

    fn bad_create_case(max: i64, heuristic: bool, ripple: f64, fail_probability: f64) {
        let n = RobustSoliton::new(max, heuristic, ripple, fail_probability);
        assert!(n.is_err());
    }

    fn get_value<T, F>(max: i64, heuristic: bool, ripple: f64, fail_probability: f64, eval: F) -> T
        where T: PartialEq + Debug,
              F: Fn(RobustSoliton) -> T
    {
        let n = try_create(max, heuristic, ripple, fail_probability);
        eval(n)
    }

    fn test_case<T, F>(max: i64, heuristic: bool, ripple: f64, fail_probability: f64, expected: T, eval: F)
        where T: PartialEq + Debug,
              F: Fn(RobustSoliton) -> T
    {
        let sol = get_value(max, heuristic, ripple, fail_probability, eval);
        assert_eq!(expected, sol);
    }

    fn test_case_greater<T, F>(max: i64, heuristic: bool, ripple: f64, fail_probability: f64, expected: T, eval: F)
        where T: PartialEq + Debug + Into<f64>,
              F: Fn(RobustSoliton) -> T
    {
        let sol = get_value(max, heuristic, ripple, fail_probability, eval);
        let a: f64 = sol.into();
        let b = expected.into();
        assert!(a > b, "{} greater than {}", a, b);
    }

    fn test_case_query(max: i64, heuristic: bool, ripple: f64, fail_probability: f64)
    {
        let mut rng = rand::thread_rng();
        let sol = try_create(max, heuristic, ripple, fail_probability);
        assert!(sol.query_table(&mut rng).is_ok());
    }

    #[test]
    fn test_create() {
        create_case(5, true, 0.1, 0.1);
        create_case(10, true, 0.1, 0.1);
        create_case(25, true, 0.1, 0.1);
        create_case(50, true, 0.1, 0.1);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(-1, true, 0.1, 0.1);
        bad_create_case(0, true, 0.1, 0.1);
    }

    #[test]
    fn test_mean() {
        test_case_greater(10, true, 0.1, 0.1, 0.9, |x| x.mean());
        test_case_greater(20, true, 0.1, 0.1, 0.9, |x| x.mean());
        test_case_greater(30, true, 0.1, 0.1, 0.9, |x| x.mean());
        test_case_greater(40, true, 0.1, 0.1, 0.9, |x| x.mean());
    }

    #[test]
    fn test_variance() {
        test_case(10, true, 0.1, 0.1, 0.0601467074373455, |x| x.variance());
    }

    #[test]
    fn test_std_dev() {
        test_case(10, true, 0.1, 0.1, (0.0601467074373455f64).sqrt(), |x| x.std_dev());
    }

    #[test]
    fn test_solition() {
        test_case(10, true, 0.1, 0.1, 0.0, |x| x.soliton(-1));
        test_case(10, true, 0.1, 0.1, 0.05879983550709325, |x| x.soliton(5));
        test_case(10, true, 0.1, 0.1, 0.0, |x| x.soliton(10));
        test_case(10, true, 0.1, 0.1, 1.0, |x| x.soliton(1));
    }

    #[test]
    fn test_query_table() {
        test_case_query(10, true, 0.1, 0.1);
        test_case_query(20, true, 0.1, 0.3);
    }
}
