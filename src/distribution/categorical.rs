use crate::distribution::{Discrete, DiscreteCDF};
use crate::statistics::*;
use crate::{Result, StatsError};
use rand::Rng;
use std::f64;

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
/// use statrs::statistics::Distribution;
/// use statrs::prec;
///
/// let n = Categorical::new(&[0.0, 1.0, 2.0]).unwrap();
/// assert!(prec::almost_eq(n.mean().unwrap(), 5.0 / 3.0, 1e-15));
/// assert_eq!(n.pmf(1), 1.0 / 3.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Categorical {
    norm_pmf: Vec<f64>,
    cdf: Vec<f64>,
    sf: Vec<f64>
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
            // extract un-normalized sf
            let sf = cdf_to_sf(&cdf);
            // extract normalized probability mass
            let sum = cdf[cdf.len() - 1];
            let mut norm_pmf = vec![0.0; prob_mass.len()];
            norm_pmf
                .iter_mut()
                .zip(prob_mass.iter())
                .for_each(|(np, pm)| *np = *pm / sum);
            Ok(Categorical { norm_pmf, cdf, sf })
        }
    }

    fn cdf_max(&self) -> f64 {
        *self.cdf.last().unwrap()
    }
}

impl ::rand::distributions::Distribution<f64> for Categorical {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        sample_unchecked(rng, &self.cdf)
    }
}

impl DiscreteCDF<u64, f64> for Categorical {
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
    fn cdf(&self, x: u64) -> f64 {
        if x >= self.cdf.len() as u64 {
            1.0
        } else {
            self.cdf.get(x as usize).unwrap() / self.cdf_max()
        }
    }

    /// Calculates the survival function for the categorical distribution
    /// at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// [ sum(p_j) from x..end ]
    /// ```
    fn sf(&self, x: u64) -> f64 {
        if x >= self.sf.len() as u64 {
            0.0
        } else {
            self.sf.get(x as usize).unwrap() / self.cdf_max()
        }
    }

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
    fn inverse_cdf(&self, x: f64) -> u64 {
        if x >= 1.0 || x <= 0.0 {
            panic!("x must be in [0, 1]")
        }
        let denorm_prob = x * self.cdf_max();
        binary_index(&self.cdf, denorm_prob) as u64
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

impl Distribution<f64> for Categorical {
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
    fn mean(&self) -> Option<f64> {
        Some(
            self.norm_pmf
                .iter()
                .enumerate()
                .fold(0.0, |acc, (idx, &val)| acc + idx as f64 * val),
        )
    }
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
    fn variance(&self) -> Option<f64> {
        let mu = self.mean()?;
        let var = self
            .norm_pmf
            .iter()
            .enumerate()
            .fold(0.0, |acc, (idx, &val)| {
                let r = idx as f64 - mu;
                acc + r * r * val
            });
        Some(var)
    }
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
    fn entropy(&self) -> Option<f64> {
        let entr = -self
            .norm_pmf
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|p| p * p.ln())
            .sum::<f64>();
        Some(entr)
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
        self.inverse_cdf(0.5) as f64
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
        *self.norm_pmf.get(x as usize).unwrap_or(&0.0)
    }

    /// Calculates the log probability mass function for the categorical
    /// distribution at `x`
    fn ln_pmf(&self, x: u64) -> f64 {
        self.pmf(x).ln()
    }
}

/// Draws a sample from the categorical distribution described by `cdf`
/// without doing any bounds checking
pub fn sample_unchecked<R: Rng + ?Sized>(rng: &mut R, cdf: &[f64]) -> f64 {
    let draw = rng.gen::<f64>() * cdf.last().unwrap();
    cdf.iter()
        .enumerate()
        .find(|(_, val)| **val >= draw)
        .map(|(i, _)| i)
        .unwrap() as f64
}

/// Computes the cdf from the given probability masses. Performs
/// no parameter or bounds checking.
pub fn prob_mass_to_cdf(prob_mass: &[f64]) -> Vec<f64> {
    let mut cdf = Vec::with_capacity(prob_mass.len());
    prob_mass.iter().fold(0.0, |s, p| {
        let sum = s + p;
        cdf.push(sum);
        sum
    });
    cdf
}

/// Computes the sf from the given cumulative densities. 
/// Performs no parameter or bounds checking.
pub fn cdf_to_sf(cdf: &[f64]) -> Vec<f64> {
    let max = *cdf.last().unwrap();
    cdf.iter().map(|x| max - x).collect()
}

// Returns the index of val if placed into the sorted search array.
// If val is greater than all elements, it therefore would return
// the length of the array (N). If val is less than all elements, it would
// return 0. Otherwise val returns the index of the first element larger than
// it within the search array.
fn binary_index(search: &[f64], val: f64) -> usize {
    use std::cmp;

    let mut low = 0_isize;
    let mut high = search.len() as isize - 1;
    while low <= high {
        let mid = low + ((high - low) / 2);
        let el = *search.get(mid as usize).unwrap();
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

#[rustfmt::skip]
#[cfg(all(test, feature = "nightly"))]
mod tests {
    use std::fmt::Debug;
    use crate::statistics::*;
    use crate::distribution::{Categorical, Discrete, DiscreteCDF};
    use crate::distribution::internal::*;
    use crate::consts::ACC;

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
        let mean = |x: Categorical| x.mean().unwrap();
        test_case(&[0.0, 0.25, 0.5, 0.25], 2.0, mean);
        test_case(&[0.0, 1.0, 2.0, 1.0], 2.0, mean);
        test_case(&[0.0, 0.5, 0.5], 1.5, mean);
        test_case(&[0.75, 0.25], 0.25, mean);
        test_case(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 5.0, mean);
    }

    #[test]
    fn test_variance() {
        let variance = |x: Categorical| x.variance().unwrap();
        test_case(&[0.0, 0.25, 0.5, 0.25], 0.5, variance);
        test_case(&[0.0, 1.0, 2.0, 1.0], 0.5, variance);
        test_case(&[0.0, 0.5, 0.5], 0.25, variance);
        test_case(&[0.75, 0.25], 0.1875, variance);
        test_case(&[1.0, 0.0, 1.0], 1.0, variance);
    }

    #[test]
    fn test_entropy() {
        let entropy = |x: Categorical| x.entropy().unwrap();
        test_case(&[0.0, 1.0], 0.0, entropy);
        test_almost(&[0.0, 1.0, 1.0], 2f64.ln(), 1e-15, entropy);
        test_almost(&[1.0, 1.0, 1.0], 3f64.ln(), 1e-15, entropy);
        test_almost(&vec![1.0; 100], 100f64.ln(), 1e-14, entropy);
        test_almost(&[0.0, 0.25, 0.5, 0.25], 1.0397207708399179, 1e-15, entropy);
    }

    #[test]
    fn test_median() {
        let median = |x: Categorical| x.median();
        test_case(&[0.0, 3.0, 1.0, 1.0], 1.0, median);
        test_case(&[4.0, 2.5, 2.5, 1.0], 1.0, median);
    }

    #[test]
    fn test_min_max() {
        let min = |x: Categorical| x.min();
        let max = |x: Categorical| x.max();
        test_case(&[4.0, 2.5, 2.5, 1.0], 0, min);
        test_case(&[4.0, 2.5, 2.5, 1.0], 3, max);
    }

    #[test]
    fn test_pmf() {
        let pmf = |arg: u64| move |x: Categorical| x.pmf(arg);
        test_case(&[0.0, 0.25, 0.5, 0.25], 0.0, pmf(0));
        test_case(&[0.0, 0.25, 0.5, 0.25], 0.25, pmf(1));
        test_case(&[0.0, 0.25, 0.5, 0.25], 0.25, pmf(3));
    }

    #[test]
    fn test_pmf_x_too_high() {
        let pmf = |arg: u64| move |x: Categorical| x.pmf(arg);
        test_case(&[4.0, 2.5, 2.5, 1.0], 0.0, pmf(4));
    }

    #[test]
    fn test_ln_pmf() {
        let ln_pmf = |arg: u64| move |x: Categorical| x.ln_pmf(arg);
        test_case(&[0.0, 0.25, 0.5, 0.25], 0f64.ln(), ln_pmf(0));
        test_case(&[0.0, 0.25, 0.5, 0.25], 0.25f64.ln(), ln_pmf(1));
        test_case(&[0.0, 0.25, 0.5, 0.25], 0.25f64.ln(), ln_pmf(3));
    }

    #[test]
    fn test_ln_pmf_x_too_high() {
        let ln_pmf = |arg: u64| move |x: Categorical| x.ln_pmf(arg);
        test_case(&[4.0, 2.5, 2.5, 1.0], f64::NEG_INFINITY, ln_pmf(4));
    }

    #[test]
    fn test_cdf() {
        let cdf = |arg: u64| move |x: Categorical| x.cdf(arg);
        test_case(&[0.0, 3.0, 1.0, 1.0], 3.0 / 5.0, cdf(1));
        test_case(&[1.0, 1.0, 1.0, 1.0], 0.25, cdf(0));
        test_case(&[4.0, 2.5, 2.5, 1.0], 0.4, cdf(0));
        test_case(&[4.0, 2.5, 2.5, 1.0], 1.0, cdf(3));
        test_case(&[4.0, 2.5, 2.5, 1.0], 1.0, cdf(4));
    }

    #[test]
    fn test_sf() {
        let sf = |arg: u64| move |x: Categorical| x.sf(arg);
        test_case(&[0.0, 3.0, 1.0, 1.0], 2.0 / 5.0, sf(1));
        test_case(&[1.0, 1.0, 1.0, 1.0], 0.75, sf(0));
        test_case(&[4.0, 2.5, 2.5, 1.0], 0.6, sf(0));
        test_case(&[4.0, 2.5, 2.5, 1.0], 0.0, sf(3));
        test_case(&[4.0, 2.5, 2.5, 1.0], 0.0, sf(4));
    }

    #[test]
    fn test_cdf_input_high() {
        let cdf = |arg: u64| move |x: Categorical| x.cdf(arg);
        test_case(&[4.0, 2.5, 2.5, 1.0], 1.0, cdf(4));
    }

    #[test]
    fn test_sf_input_high() {
        let sf = |arg: u64| move |x: Categorical| x.sf(arg);
        test_case(&[4.0, 2.5, 2.5, 1.0], 0.0, sf(4));
    }

    #[test]
    fn test_cdf_sf_mirror() {
        let mass = [4.0, 2.5, 2.5, 1.0];
        let cat = Categorical::new(&mass).unwrap();
        assert_eq!(cat.cdf(0), 1.-cat.sf(0));
        assert_eq!(cat.cdf(1), 1.-cat.sf(1));
        assert_eq!(cat.cdf(2), 1.-cat.sf(2));
        assert_eq!(cat.cdf(3), 1.-cat.sf(3));
    }

    #[test]
    fn test_inverse_cdf() {
        let inverse_cdf = |arg: f64| move |x: Categorical| x.inverse_cdf(arg);
        test_case(&[0.0, 3.0, 1.0, 1.0], 1, inverse_cdf(0.2));
        test_case(&[0.0, 3.0, 1.0, 1.0], 1, inverse_cdf(0.5));
        test_case(&[0.0, 3.0, 1.0, 1.0], 3, inverse_cdf(0.95));
        test_case(&[4.0, 2.5, 2.5, 1.0], 0, inverse_cdf(0.2));
        test_case(&[4.0, 2.5, 2.5, 1.0], 1, inverse_cdf(0.5));
        test_case(&[4.0, 2.5, 2.5, 1.0], 3, inverse_cdf(0.95));
    }

    #[test]
    #[should_panic]
    fn test_inverse_cdf_input_low() {
        let inverse_cdf = |arg: f64| move |x: Categorical| x.inverse_cdf(arg);
        get_value(&[4.0, 2.5, 2.5, 1.0], inverse_cdf(0.0));
    }

    #[test]
    #[should_panic]
    fn test_inverse_cdf_input_high() {
        let inverse_cdf = |arg: f64| move |x: Categorical| x.inverse_cdf(arg);
        get_value(&[4.0, 2.5, 2.5, 1.0], inverse_cdf(1.0));
    }

    #[test]
    fn test_discrete() {
        test::check_discrete_distribution(&try_create(&[1.0, 2.0, 3.0, 4.0]), 4);
        test::check_discrete_distribution(&try_create(&[0.0, 1.0, 2.0, 3.0, 4.0]), 5);
    }
}
