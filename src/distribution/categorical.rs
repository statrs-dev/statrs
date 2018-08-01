use distribution::{CheckedInverseCDF, Discrete, InverseCDF, Univariate};
use rand::distributions::Distribution;
use rand::Rng;
use statistics::*;
use std::f64;
use {Result, StatsError};

/// Implements the
/// [Categorical](https://en.wikipedia.org/wiki/Categorical_distribution)
/// distribution, also known as the generalized Bernoulli or discrete
/// distribution
///
/// # Examples
///
/// ```
///
/// use statrs::distribution::{Categorical, Discrete};
/// use statrs::statistics::Mean;
/// use statrs::prec;
///
/// let n = Categorical::new(&[0.0, 1.0, 2.0]).unwrap();
/// assert!(prec::almost_eq(n.mean(), 5.0 / 3.0, 1e-15));
/// assert_eq!(n.pmf(1), 1.0 / 3.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Categorical {
    norm_pmf: Vec<f64>,
    cdf: Vec<f64>,
}

impl Categorical {
    /// Constructs a new categorical distribution
    /// with the probabilities masses defined by `prob_mass`
    ///
    /// # Errors
    ///
    /// Returns an error if `prob_mass` is empty, the sum of
    /// the elements in `prob_mass` is 0, or any element is less than
    /// 0 or is `f64::NAN`
    ///
    /// # Note
    ///
    /// The elements in `prob_mass` do not need to be normalized
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Categorical;
    ///
    /// let mut result = Categorical::new(&[0.0, 1.0, 2.0]);
    /// assert!(result.is_ok());
    ///
    /// result = Categorical::new(&[0.0, -1.0, 2.0]);
    /// assert!(result.is_err());
    /// ```
    pub fn new(prob_mass: &[f64]) -> Result<Categorical> {
        if !super::internal::is_valid_multinomial(prob_mass, true) {
            Err(StatsError::BadParams)
        } else {
            // extract un-normalized cdf
            let cdf = prob_mass_to_cdf(prob_mass);
            // extract normalized probability mass
            let sum = cdf[cdf.len() - 1];
            let mut norm_pmf = vec![0.0; prob_mass.len()];
            for i in 0..prob_mass.len() {
                unsafe {
                    let elem = norm_pmf.get_unchecked_mut(i);
                    *elem = prob_mass.get_unchecked(i) / sum;
                }
            }
            Ok(Categorical {
                norm_pmf: norm_pmf,
                cdf: cdf,
            })
        }
    }

    fn cdf_max(&self) -> f64 {
        *unsafe { self.cdf.get_unchecked(self.cdf.len() - 1) }
    }
}

impl Distribution<f64> for Categorical {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> f64 {
        sample_unchecked(r, &self.cdf)
    }
}

impl Univariate<u64, f64> for Categorical {
    /// Calculates the cumulative distribution function for the categorical
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// sum(p_j) from 0..x
    /// ```
    ///
    /// where `p_j` is the probability mass for the `j`th category
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else if x >= self.cdf.len() as f64 {
            1.0
        } else {
            unsafe { self.cdf.get_unchecked(x as usize) / self.cdf_max() }
        }
    }
}

impl InverseCDF<f64> for Categorical {
    /// Calculates the inverse cumulative distribution function for the
    /// categorical
    /// distribution at `x`
    ///
    /// # Panics
    ///
    /// If `x <= 0.0` or `x >= 1.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// i
    /// ```
    ///
    /// where `i` is the first index such that `x < f(i)`
    /// and `f(x)` is defined as `p_x + f(x - 1)` and `f(0) = p_0` where
    /// `p_x` is the `x`th probability mass
    fn inverse_cdf(&self, x: f64) -> f64 {
        self.checked_inverse_cdf(x).unwrap()
    }
}

impl CheckedInverseCDF<f64> for Categorical {
    /// Calculates the inverse cumulative distribution function for the
    /// categorical
    /// distribution at `x`
    ///
    /// # Errors
    ///
    /// If `x <= 0.0` or `x >= 1.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// i
    /// ```
    ///
    /// where `i` is the first index such that `x < f(i)`
    /// and `f(x)` is defined as `p_x + f(x - 1)` and `f(0) = p_0` where
    /// `p_x` is the `x`th probability mass
    fn checked_inverse_cdf(&self, x: f64) -> Result<f64> {
        if x <= 0.0 || x >= 1.0 {
            Err(StatsError::ArgIntervalExcl("x", 0.0, 1.0))
        } else {
            let denorm_prob = x * self.cdf_max();
            Ok(binary_index(&self.cdf, denorm_prob) as f64)
        }
    }
}

impl Min<u64> for Categorical {
    /// Returns the minimum value in the domain of the
    /// categorical distribution representable by a 64-bit
    /// integer
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

impl Max<u64> for Categorical {
    /// Returns the maximum value in the domain of the
    /// categorical distribution representable by a 64-bit
    /// integer
    ///
    /// # Formula
    ///
    /// ```ignore
    /// n
    /// ```
    fn max(&self) -> u64 {
        self.cdf.len() as u64 - 1
    }
}

impl Mean<f64> for Categorical {
    /// Returns the mean of the categorical distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// Σ(j * p_j)
    /// ```
    ///
    /// where `p_j` is the `j`th probability mass,
    /// `Σ` is the sum from `0` to `k - 1`,
    /// and `k` is the number of categories
    fn mean(&self) -> f64 {
        self.norm_pmf
            .iter()
            .enumerate()
            .fold(0.0, |acc, (idx, &val)| acc + idx as f64 * val)
    }
}

impl Variance<f64> for Categorical {
    /// Returns the variance of the categorical distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// Σ(p_j * (j - μ)^2)
    /// ```
    ///
    /// where `p_j` is the `j`th probability mass, `μ` is the mean,
    /// `Σ` is the sum from `0` to `k - 1`,
    /// and `k` is the number of categories
    fn variance(&self) -> f64 {
        let mu = self.mean();
        self.norm_pmf
            .iter()
            .enumerate()
            .fold(0.0, |acc, (idx, &val)| {
                let r = idx as f64 - mu;
                acc + r * r * val
            })
    }

    /// Returns the standard deviation of the categorical distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// sqrt(Σ(p_j * (j - μ)^2))
    /// ```
    ///
    /// where `p_j` is the `j`th probability mass, `μ` is the mean,
    /// `Σ` is the sum from `0` to `k - 1`,
    /// and `k` is the number of categories
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

impl Entropy<f64> for Categorical {
    /// Returns the entropy of the categorical distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// -Σ(p_j * ln(p_j))
    /// ```
    ///
    /// where `p_j` is the `j`th probability mass,
    /// `Σ` is the sum from `0` to `k - 1`,
    /// and `k` is the number of categories
    fn entropy(&self) -> f64 {
        -self
            .norm_pmf
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|p| p * p.ln())
            .sum::<f64>()
    }
}

impl Median<f64> for Categorical {
    /// Returns the median of the categorical distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// CDF^-1(0.5)
    /// ```
    fn median(&self) -> f64 {
        self.inverse_cdf(0.5)
    }
}

impl Discrete<u64, f64> for Categorical {
    /// Calculates the probability mass function for the categorical
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// p_x
    /// ```
    fn pmf(&self, x: u64) -> f64 {
        if x >= self.norm_pmf.len() as u64 {
            0.0
        } else {
            unsafe { *self.norm_pmf.get_unchecked(x as usize) }
        }
    }

    /// Calculates the log probability mass function for the categorical
    /// distribution at `x`
    fn ln_pmf(&self, x: u64) -> f64 {
        self.pmf(x).ln()
    }
}

/// Draws a sample from the categorical distribution described by `cdf`
/// without doing any bounds checking
pub fn sample_unchecked<R: Rng + ?Sized>(r: &mut R, cdf: &[f64]) -> f64 {
    let draw = r.gen::<f64>() * unsafe { cdf.get_unchecked(cdf.len() - 1) };
    let mut idx = 0;

    if draw == 0.0 {
        // skip zero-probability categories
        let mut el = unsafe { cdf.get_unchecked(idx) };
        while *el == 0.0 {
            // don't need bounds checking because we do not allow
            // creating Categorical distributions with all 0.0 probs
            idx += 1;
            el = unsafe { cdf.get_unchecked(idx) }
        }
    }
    let mut el = unsafe { cdf.get_unchecked(idx) };
    while draw > *el {
        idx += 1;
        el = unsafe { cdf.get_unchecked(idx) };
    }
    idx as f64
}

/// Computes the cdf from the given probability masses. Performs
/// no parameter or bounds checking.
pub fn prob_mass_to_cdf(prob_mass: &[f64]) -> Vec<f64> {
    let mut cdf = vec![0.0; prob_mass.len()];
    cdf[0] = prob_mass[0];
    for i in 1..prob_mass.len() {
        unsafe {
            let val = cdf.get_unchecked(i - 1) + prob_mass.get_unchecked(i);
            let elem = cdf.get_unchecked_mut(i);
            *elem = val;
        }
    }
    cdf
}

// Returns the index of val if placed into the sorted search array.
// If val is greater than all elements, it therefore would return
// the length of the array (N). If val is less than all elements, it would
// return 0. Otherwise val returns the index of the first element larger than
// it within the search array.
fn binary_index(search: &[f64], val: f64) -> usize {
    use std::cmp;

    let mut low = 0 as isize;
    let mut high = search.len() as isize - 1;
    while low <= high {
        let mid = low + ((high - low) / 2);
        let el = *unsafe { search.get_unchecked(mid as usize) };
        if el > val {
            high = mid - 1;
        } else if el < val {
            low = mid.saturating_add(1);
        } else {
            return mid as usize;
        }
    }
    cmp::min(search.len(), cmp::max(low, 0) as usize)
}

#[test]
fn test_prob_mass_to_cdf() {
    let arr = [0.0, 0.5, 0.5, 3.0, 1.1];
    let res = prob_mass_to_cdf(&arr);
    assert_eq!(res, [0.0, 0.5, 1.0, 4.0, 5.1]);
}

#[test]
fn test_binary_index() {
    let arr = [0.0, 3.0, 5.0, 9.0, 10.0];
    assert_eq!(0, binary_index(&arr, -1.0));
    assert_eq!(2, binary_index(&arr, 5.0));
    assert_eq!(3, binary_index(&arr, 5.2));
    assert_eq!(5, binary_index(&arr, 10.1));
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use std::f64;
    use std::fmt::Debug;
    use statistics::*;
    use distribution::{Categorical, CheckedInverseCDF, Discrete, InverseCDF, Univariate};
    use distribution::internal::*;

    fn try_create(prob_mass: &[f64]) -> Categorical {
        let n = Categorical::new(prob_mass);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(prob_mass: &[f64]) {
        try_create(prob_mass);
    }

    fn bad_create_case(prob_mass: &[f64]) {
        let n = Categorical::new(prob_mass);
        assert!(n.is_err());
    }

    fn get_value<T, F>(prob_mass: &[f64], eval: F) -> T
        where T: PartialEq + Debug,
              F: Fn(Categorical) -> T
    {
        let n = try_create(prob_mass);
        eval(n)
    }

    fn test_case<T, F>(prob_mass: &[f64], expected: T, eval: F)
        where T: PartialEq + Debug,
              F: Fn(Categorical) -> T
    {
        let x = get_value(prob_mass, eval);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(prob_mass: &[f64], expected: f64, acc: f64, eval: F)
        where F: Fn(Categorical) -> f64
    {
        let x = get_value(prob_mass, eval);
        assert_almost_eq!(expected, x, acc);
    }

    #[test]
    fn test_create() {
        create_case(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(&[-1.0, 1.0]);
        bad_create_case(&[0.0, 0.0]);
    }

    #[test]
    fn test_mean() {
        test_case(&[0.0, 0.25, 0.5, 0.25], 2.0, |x| x.mean());
        test_case(&[0.0, 1.0, 2.0, 1.0], 2.0, |x| x.mean());
        test_case(&[0.0, 0.5, 0.5], 1.5, |x| x.mean());
        test_case(&[0.75, 0.25], 0.25, |x| x.mean());
        test_case(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 5.0, |x| x.mean());
    }

    #[test]
    fn test_variance() {
        test_case(&[0.0, 0.25, 0.5, 0.25], 0.5, |x| x.variance());
        test_case(&[0.0, 1.0, 2.0, 1.0], 0.5, |x| x.variance());
        test_case(&[0.0, 0.5, 0.5], 0.25, |x| x.variance());
        test_case(&[0.75, 0.25], 0.1875, |x| x.variance());
        test_case(&[1.0, 0.0, 1.0], 1.0, |x| x.variance());
    }

    #[test]
    fn test_std_dev() {
        test_case(&[0.0, 0.25, 0.5, 0.25], 0.70710678118654752440084436210485, |x| x.std_dev());
        test_case(&[0.0, 1.0, 2.0, 1.0], 0.70710678118654752440084436210485, |x| x.std_dev());
        test_case(&[0.0, 0.5, 0.5], 0.5, |x| x.std_dev());
        test_case(&[0.75, 0.25], 0.43301270189221932338186158537647, |x| x.std_dev());
        test_case(&[1.0, 0.0, 1.0], 1.0, |x| x.std_dev());
    }

    #[test]
    fn test_entropy() {
        test_case(&[0.0, 1.0], 0.0, |x| x.entropy());
        test_almost(&[0.0, 1.0, 1.0], 2f64.ln(), 1e-15, |x| x.entropy());
        test_almost(&[1.0, 1.0, 1.0], 3f64.ln(), 1e-15, |x| x.entropy());
        test_almost(&vec![1.0; 100], 100f64.ln(), 1e-14, |x| x.entropy());
        test_almost(&[0.0, 0.25, 0.5, 0.25], 1.0397207708399179, 1e-15, |x| x.entropy());
    }

    #[test]
    fn test_median() {
        test_case(&[0.0, 3.0, 1.0, 1.0], 1.0, |x| x.median());
        test_case(&[4.0, 2.5, 2.5, 1.0], 1.0, |x| x.median());
    }

    #[test]
    fn test_min_max() {
        test_case(&[4.0, 2.5, 2.5, 1.0], 0, |x| x.min());
        test_case(&[4.0, 2.5, 2.5, 1.0], 3, |x| x.max());
    }

    #[test]
    fn test_pmf() {
        test_case(&[0.0, 0.25, 0.5, 0.25], 0.0, |x| x.pmf(0));
        test_case(&[0.0, 0.25, 0.5, 0.25], 0.25, |x| x.pmf(1));
        test_case(&[0.0, 0.25, 0.5, 0.25], 0.25, |x| x.pmf(3));
    }

    #[test]
    fn test_pmf_x_too_high() {
        test_case(&[4.0, 2.5, 2.5, 1.0], 0.0, |x| x.pmf(4));
    }

    #[test]
    fn test_ln_pmf() {
        test_case(&[0.0, 0.25, 0.5, 0.25], 0f64.ln(), |x| x.ln_pmf(0));
        test_case(&[0.0, 0.25, 0.5, 0.25], 0.25f64.ln(), |x| x.ln_pmf(1));
        test_case(&[0.0, 0.25, 0.5, 0.25], 0.25f64.ln(), |x| x.ln_pmf(3));
    }

    #[test]
    fn test_ln_pmf_x_too_high() {
        test_case(&[4.0, 2.5, 2.5, 1.0], f64::NEG_INFINITY, |x| x.ln_pmf(4));
    }

    #[test]
    fn test_cdf() {
        test_case(&[0.0, 3.0, 1.0, 1.0], 3.0 / 5.0, |x| x.cdf(1.5));
        test_case(&[1.0, 1.0, 1.0, 1.0], 0.25, |x| x.cdf(0.0));
        test_case(&[4.0, 2.5, 2.5, 1.0], 0.4, |x| x.cdf(0.8));
        test_case(&[4.0, 2.5, 2.5, 1.0], 1.0, |x| x.cdf(3.2));
        test_case(&[4.0, 2.5, 2.5, 1.0], 1.0, |x| x.cdf(4.0));
    }

    #[test]
    fn test_cdf_input_low() {
        test_case(&[4.0, 2.5, 2.5, 1.0], 0.0, |x| x.cdf(-1.0));
    }

    #[test]
    fn test_cdf_input_high() {
        test_case(&[4.0, 2.5, 2.5, 1.0], 1.0, |x| x.cdf(4.5));
    }

    #[test]
    fn test_inverse_cdf() {
        test_case(&[0.0, 3.0, 1.0, 1.0], 1.0, |x| x.inverse_cdf(0.2));
        test_case(&[0.0, 3.0, 1.0, 1.0], 1.0, |x| x.inverse_cdf(0.5));
        test_case(&[0.0, 3.0, 1.0, 1.0], 3.0, |x| x.inverse_cdf(0.95));
        test_case(&[4.0, 2.5, 2.5, 1.0], 0.0, |x| x.inverse_cdf(0.2));
        test_case(&[4.0, 2.5, 2.5, 1.0], 1.0, |x| x.inverse_cdf(0.5));
        test_case(&[4.0, 2.5, 2.5, 1.0], 3.0, |x| x.inverse_cdf(0.95));
    }

    #[test]
    #[should_panic]
    fn test_inverse_cdf_input_low() {
        get_value(&[4.0, 2.5, 2.5, 1.0], |x| x.inverse_cdf(0.0));
    }

    #[test]
    #[should_panic]
    fn test_inverse_cdf_input_high() {
        get_value(&[4.0, 2.5, 2.5, 1.0], |x| x.inverse_cdf(1.0));
    }

    #[test]
    fn test_checked_inverse_cdf_input_low() {
        let n = try_create(&[4.0, 2.5, 2.5, 1.0]);
        assert!(n.checked_inverse_cdf(0.0).is_err());
    }

    #[test]
    fn test_checked_inverse_cdf_input_high() {
        let n = try_create(&[4.0, 2.5, 2.5, 1.0]);
        assert!(n.checked_inverse_cdf(1.0).is_err());
    }

    #[test]
    fn test_discrete() {
        test::check_discrete_distribution(&try_create(&[1.0, 2.0, 3.0, 4.0]), 4);
        test::check_discrete_distribution(&try_create(&[0.0, 1.0, 2.0, 3.0, 4.0]), 5);
    }
}
