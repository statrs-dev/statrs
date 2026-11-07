use crate::distribution::{Discrete, DiscreteCDF};
use crate::statistics::*;
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

/// Implements the
/// [Categorical](https://en.wikipedia.org/wiki/Categorical_distribution)
/// distribution, also known as the generalized Bernoulli or discrete
/// distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Categorical, Discrete};
/// use statrs::statistics::Distribution;
/// use statrs::prec;
/// use approx::assert_abs_diff_eq;
///
/// let n = Categorical::new(&[0.0, 1.0, 2.0]).unwrap();
/// assert_abs_diff_eq!(n.mean().unwrap(), 5.0 / 3.0, epsilon = 1e-15);
/// assert_eq!(n.pmf(1), 1.0 / 3.0);
/// ```
#[derive(Clone, PartialEq, Debug)]
pub struct Categorical {
    norm_pmf: Vec<f64>,
    norm_cdf: Vec<f64>,
}

/// Represents the errors that can occur when creating a [`Categorical`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum CategoricalError {
    /// The probability mass is empty.
    ProbMassEmpty,

    /// The probabilities sums up to zero.
    ProbMassSumZero,

    /// The probability mass contains at least one element which is NaN or less than zero.
    ProbMassHasInvalidElements,
}

impl core::fmt::Display for CategoricalError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match self {
            CategoricalError::ProbMassEmpty => write!(f, "Probability mass is empty"),
            CategoricalError::ProbMassSumZero => write!(f, "Probabilities sum up to zero"),
            CategoricalError::ProbMassHasInvalidElements => write!(
                f,
                "Probability mass contains at least one element which is NaN or less than zero"
            ),
        }
    }
}

impl core::error::Error for CategoricalError {}

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
    pub fn new(prob_mass: &[f64]) -> Result<Categorical, CategoricalError> {
        if prob_mass.is_empty() {
            return Err(CategoricalError::ProbMassEmpty);
        }

        let mut prob_sum = 0.0;
        for &p in prob_mass {
            if p.is_nan() || p < 0.0 {
                return Err(CategoricalError::ProbMassHasInvalidElements);
            }

            prob_sum += p;
        }

        if prob_sum == 0.0 {
            return Err(CategoricalError::ProbMassSumZero);
        }

        let mut cdf_sum = 0.0;

        let mut norm_cdf = Vec::with_capacity(prob_mass.len());
        let mut norm_pmf = Vec::with_capacity(prob_mass.len());

        for &prob in prob_mass {
            cdf_sum += prob;

            norm_cdf.push(cdf_sum / prob_sum);
            norm_pmf.push(prob / prob_sum);
        }

        Ok(Categorical { norm_pmf, norm_cdf })
    }

    /// Returns the first index `i` such that `norm_cdf[i] >= x`.
    ///
    /// This is the lower bound needed by `inverse_cdf`, including when
    /// zero-probability categories repeat a cumulative probability.
    fn locate(&self, x: f64) -> u64 {
        self.norm_cdf
            .partition_point(|v| v.partial_cmp(&x).unwrap().is_lt()) as u64
    }
}

impl core::fmt::Display for Categorical {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Cat({:#?})", self.norm_pmf)
    }
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl ::rand::distr::Distribution<usize> for Categorical {
    fn sample<R: ::rand::Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let draw = ::rand::RngExt::random::<f64>(rng);
        // Sampling uses F(i) > draw, unlike the quantile's F(i) >= x.
        // Skip repeated boundaries so zero-mass categories cannot be drawn.
        self.norm_cdf
            .partition_point(|v| v.partial_cmp(&draw).unwrap().is_le())
    }
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl ::rand::distr::Distribution<u64> for Categorical {
    fn sample<R: ::rand::Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        <Self as ::rand::distr::Distribution<usize>>::sample(self, rng) as u64
    }
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl ::rand::distr::Distribution<f64> for Categorical {
    fn sample<R: ::rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        <Self as ::rand::distr::Distribution<usize>>::sample(self, rng) as f64
    }
}

impl DiscreteCDF<u64, f64> for Categorical {
    /// Calculates the cumulative distribution function for the categorical
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// sum(p_j) from 0..x
    /// ```
    ///
    /// where `p_j` is the probability mass for the `j`th category
    fn cdf(&self, x: u64) -> f64 {
        *self.norm_cdf.get(x as usize).unwrap_or(&1.0)
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
    /// ```text
    /// i
    /// ```
    ///
    /// where `i` is the first index such that `x <= f(i)`
    /// and `f(x)` is defined as `p_x + f(x - 1)` and `f(0) = p_0` where
    /// `p_x` is the `x`th probability mass
    fn inverse_cdf(&self, x: f64) -> u64 {
        if x >= 1.0 || x <= 0.0 {
            panic!("x must be in [0, 1]")
        }

        self.locate(x)
    }
}

impl Min<u64> for Categorical {
    /// Returns the minimum value in the domain of the
    /// categorical distribution representable by a 64-bit
    /// integer
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

impl Max<u64> for Categorical {
    /// Returns the maximum value in the domain of the
    /// categorical distribution representable by a 64-bit
    /// integer
    ///
    /// # Formula
    ///
    /// ```text
    /// n
    /// ```
    fn max(&self) -> u64 {
        self.norm_cdf.len() as u64 - 1
    }
}

impl Distribution<f64> for Categorical {
    /// Returns the mean of the categorical distribution
    ///
    /// # Formula
    ///
    /// ```text
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
    /// ```text
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
    /// ```text
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
    /// ```text
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
    /// ```text
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

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::internal::density_util;
    use crate::distribution::internal::testing_boiler;

    testing_boiler!(prob_mass: &[f64]; Categorical; CategoricalError);

    #[test]
    fn test_create() {
        create_ok(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_bad_create() {
        let invalid: &[(&[f64], CategoricalError)] = &[
            (&[], CategoricalError::ProbMassEmpty),
            (&[-1.0, 1.0], CategoricalError::ProbMassHasInvalidElements),
            (&[0.0, 0.0, 0.0], CategoricalError::ProbMassSumZero),
        ];

        for &(prob_mass, err) in invalid {
            test_create_err(prob_mass, err);
        }
    }

    #[test]
    fn test_mean() {
        let mean = |x: Categorical| x.mean().unwrap();
        test_exact(&[0.0, 0.25, 0.5, 0.25], 2.0, mean);
        test_exact(&[0.0, 1.0, 2.0, 1.0], 2.0, mean);
        test_exact(&[0.0, 0.5, 0.5], 1.5, mean);
        test_exact(&[0.75, 0.25], 0.25, mean);
        test_exact(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 5.0, mean);
    }

    #[test]
    fn test_variance() {
        let variance = |x: Categorical| x.variance().unwrap();
        test_exact(&[0.0, 0.25, 0.5, 0.25], 0.5, variance);
        test_exact(&[0.0, 1.0, 2.0, 1.0], 0.5, variance);
        test_exact(&[0.0, 0.5, 0.5], 0.25, variance);
        test_exact(&[0.75, 0.25], 0.1875, variance);
        test_exact(&[1.0, 0.0, 1.0], 1.0, variance);
    }

    #[test]
    fn test_entropy() {
        let entropy = |x: Categorical| x.entropy().unwrap();
        test_exact(&[0.0, 1.0], 0.0, entropy);
        test_absolute(&[0.0, 1.0, 1.0], 2f64.ln(), 1e-15, entropy);
        test_absolute(&[1.0, 1.0, 1.0], 3f64.ln(), 1e-15, entropy);
        test_absolute(&[1.0; 100], 100f64.ln(), 1e-14, entropy);
        test_absolute(&[0.0, 0.25, 0.5, 0.25], 1.0397207708399179, 1e-15, entropy);
    }

    #[test]
    fn test_median() {
        let median = |x: Categorical| x.median();
        test_exact(&[0.0, 3.0, 1.0, 1.0], 1.0, median);
        test_exact(&[4.0, 2.5, 2.5, 1.0], 1.0, median);
    }

    #[test]
    fn test_min_max() {
        let min = |x: Categorical| x.min();
        let max = |x: Categorical| x.max();
        test_exact(&[4.0, 2.5, 2.5, 1.0], 0, min);
        test_exact(&[4.0, 2.5, 2.5, 1.0], 3, max);
    }

    #[test]
    fn test_pmf() {
        let pmf = |arg: u64| move |x: Categorical| x.pmf(arg);
        test_exact(&[0.0, 0.25, 0.5, 0.25], 0.0, pmf(0));
        test_exact(&[0.0, 0.25, 0.5, 0.25], 0.25, pmf(1));
        test_exact(&[0.0, 0.25, 0.5, 0.25], 0.25, pmf(3));
    }

    #[test]
    fn test_pmf_x_too_high() {
        let pmf = |arg: u64| move |x: Categorical| x.pmf(arg);
        test_exact(&[4.0, 2.5, 2.5, 1.0], 0.0, pmf(4));
    }

    #[test]
    fn test_ln_pmf() {
        let ln_pmf = |arg: u64| move |x: Categorical| x.ln_pmf(arg);
        test_exact(&[0.0, 0.25, 0.5, 0.25], 0f64.ln(), ln_pmf(0));
        test_exact(&[0.0, 0.25, 0.5, 0.25], 0.25f64.ln(), ln_pmf(1));
        test_exact(&[0.0, 0.25, 0.5, 0.25], 0.25f64.ln(), ln_pmf(3));
    }

    #[test]
    fn test_ln_pmf_x_too_high() {
        let ln_pmf = |arg: u64| move |x: Categorical| x.ln_pmf(arg);
        test_exact(&[4.0, 2.5, 2.5, 1.0], f64::NEG_INFINITY, ln_pmf(4));
    }

    #[test]
    fn test_cdf() {
        let cdf = |arg: u64| move |x: Categorical| x.cdf(arg);
        test_exact(&[0.0, 3.0, 1.0, 1.0], 3.0 / 5.0, cdf(1));
        test_exact(&[1.0, 1.0, 1.0, 1.0], 0.25, cdf(0));
        test_exact(&[4.0, 2.5, 2.5, 1.0], 0.4, cdf(0));
        test_exact(&[4.0, 2.5, 2.5, 1.0], 1.0, cdf(3));
        test_exact(&[4.0, 2.5, 2.5, 1.0], 1.0, cdf(4));
    }

    #[test]
    fn test_sf() {
        let sf = |arg: u64| move |x: Categorical| x.sf(arg);
        test_exact(&[0.0, 3.0, 1.0, 1.0], 2.0 / 5.0, sf(1));
        test_exact(&[1.0, 1.0, 1.0, 1.0], 0.75, sf(0));
        test_exact(&[4.0, 2.5, 2.5, 1.0], 0.6, sf(0));
        test_exact(&[4.0, 2.5, 2.5, 1.0], 0.0, sf(3));
        test_exact(&[4.0, 2.5, 2.5, 1.0], 0.0, sf(4));
    }

    #[test]
    fn test_cdf_input_high() {
        let cdf = |arg: u64| move |x: Categorical| x.cdf(arg);
        test_exact(&[4.0, 2.5, 2.5, 1.0], 1.0, cdf(4));
    }

    #[test]
    fn test_sf_input_high() {
        let sf = |arg: u64| move |x: Categorical| x.sf(arg);
        test_exact(&[4.0, 2.5, 2.5, 1.0], 0.0, sf(4));
    }

    #[test]
    fn test_cdf_sf_mirror() {
        let mass = [4.0, 2.5, 2.5, 1.0];
        let cat = Categorical::new(&mass).unwrap();
        assert_eq!(cat.cdf(0), 1. - cat.sf(0));
        assert_eq!(cat.cdf(1), 1. - cat.sf(1));
        assert_eq!(cat.cdf(2), 1. - cat.sf(2));
        assert_eq!(cat.cdf(3), 1. - cat.sf(3));
    }

    #[test]
    fn test_cdf_sf_sum_to_one() {
        let masses: &[&[f64]] = &[
            &[4.0, 2.5, 2.5, 1.0],
            &[0.0, 3.0, 1.0, 1.0],
            &[1.0, 1.0, 1.0, 1.0],
            &[1.0; 20],
        ];

        for &mass in masses {
            let cat = Categorical::new(mass).unwrap();
            for x in 0..mass.len() as u64 + 1 {
                crate::prec::assert_abs_diff_eq!(cat.cdf(x) + cat.sf(x), 1.0, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_locate_treats_zero_as_first_index() {
        let cat = create_ok(&[4.0, 2.5, 2.5, 1.0]);
        assert_eq!(cat.locate(0.0), 0);
    }

    #[test]
    fn test_locate_matches_inverse_cdf() {
        let cat = create_ok(&[4.0, 2.5, 2.5, 1.0]);
        for &x in &[0.1, 0.2, 0.4, 0.5, 0.8, 0.95, 0.999] {
            assert_eq!(cat.locate(x), cat.inverse_cdf(x));
        }
    }

    #[test]
    fn test_inverse_cdf() {
        let inverse_cdf = |arg: f64| move |x: Categorical| x.inverse_cdf(arg);
        test_exact(&[0.0, 3.0, 1.0, 1.0], 1, inverse_cdf(0.2));
        test_exact(&[0.0, 3.0, 1.0, 1.0], 1, inverse_cdf(0.5));
        test_exact(&[0.0, 3.0, 1.0, 1.0], 3, inverse_cdf(0.95));
        test_exact(&[4.0, 2.5, 2.5, 1.0], 0, inverse_cdf(0.2));
        test_exact(&[4.0, 2.5, 2.5, 1.0], 1, inverse_cdf(0.5));
        test_exact(&[4.0, 2.5, 2.5, 1.0], 3, inverse_cdf(0.95));
    }

    #[test]
    fn test_inverse_cdf_zero_mass_boundaries() {
        let cat = Categorical::new(&[0.25, 0.25, 0.0, 0.0, 0.5]).unwrap();
        let below = f64::from_bits(0.5f64.to_bits() - 1);
        let above = f64::from_bits(0.5f64.to_bits() + 1);
        assert_eq!(cat.inverse_cdf(below), 1);
        assert_eq!(cat.inverse_cdf(0.5), 1);
        assert_eq!(cat.inverse_cdf(above), 4);
        assert_eq!(cat.median(), 1.0);

        let leading = Categorical::new(&[0.0, 0.0, 0.5, 0.0, 0.5, 0.0]).unwrap();
        assert_eq!(leading.inverse_cdf(f64::from_bits(1)), 2);
        assert_eq!(leading.inverse_cdf(0.5), 2);
        assert_eq!(leading.inverse_cdf(above), 4);
        let trailing = Categorical::new(&[1.0, 0.0, 0.0]).unwrap();
        assert_eq!(trailing.inverse_cdf(0.5), 0);
        assert_eq!(trailing.inverse_cdf(f64::from_bits(1.0f64.to_bits() - 1)), 0);
    }

    #[test]
    fn test_inverse_cdf_integer_mass_oracle() {
        // Every nonzero four-category mass vector with entries in {0, 1, 2}.
        for encoded in 1..81 {
            let mut code = encoded;
            let mut mass = [0.0; 4];
            for value in &mut mass {
                *value = (code % 3) as f64;
                code /= 3;
            }
            let total: f64 = mass.iter().sum();
            let cat = Categorical::new(&mass).unwrap();
            let mut probabilities = alloc::vec![f64::from_bits(1), 0.25, 0.5, 0.75,
                f64::from_bits(1.0f64.to_bits() - 1)];
            let mut cumulative = 0.0;
            for value in mass {
                cumulative += value;
                let boundary = cumulative / total;
                if boundary > 0.0 && boundary < 1.0 {
                    probabilities.extend_from_slice(&[
                        f64::from_bits(boundary.to_bits() - 1), boundary,
                        f64::from_bits(boundary.to_bits() + 1),
                    ]);
                }
            }
            for p in probabilities {
                if p <= 0.0 || p >= 1.0 { continue; }
                let mut prefix = 0.0;
                let expected = mass.iter().position(|value| {
                    prefix += value;
                    prefix / total >= p
                }).unwrap();
                let actual = cat.inverse_cdf(p);
                assert_eq!(actual, expected as u64, "mass={mass:?}, p={p:?}");
                assert!(cat.pmf(actual) > 0.0);
            }
        }
    }

    #[cfg(feature = "rand")]
    #[test]
    fn test_sample_zero_mass_boundaries() {
        // Fixed words drive exact StandardUniform f64 draws, not frequencies.
        struct FixedWord(u64);
        impl ::rand::TryRng for FixedWord {
            type Error = core::convert::Infallible;
            fn try_next_u32(&mut self) -> Result<u32, Self::Error> { Ok(self.0 as u32) }
            fn try_next_u64(&mut self) -> Result<u64, Self::Error> { Ok(self.0) }
            fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
                for chunk in dst.chunks_mut(8) {
                    chunk.copy_from_slice(&self.0.to_le_bytes()[..chunk.len()]);
                }
                Ok(())
            }
        }
        let draws = [
            (0, 0.0),
            (1 << 62, 0.25),
            ((1 << 63) - 2048, f64::from_bits(0.5f64.to_bits() - 2)),
            (1 << 63, 0.5),
            ((1 << 63) + 2048, f64::from_bits(0.5f64.to_bits() + 1)),
            (u64::MAX, f64::from_bits(1.0f64.to_bits() - 1)),
        ];
        let masses: &[&[f64]] = &[
            &[0.0, 0.0, 1.0, 0.0, 0.0],
            &[0.25, 0.25, 0.0, 0.0, 0.5],
            &[1.0, 0.0, 0.0],
        ];
        for mass in masses {
            let cat = Categorical::new(mass).unwrap();
            let total: f64 = mass.iter().sum();
            for (word, draw) in draws {
                assert_eq!(::rand::RngExt::random::<f64>(&mut FixedWord(word)), draw);
                let mut prefix = 0.0;
                let expected = mass.iter().position(|value| {
                    prefix += value;
                    prefix / total > draw
                }).unwrap();
                let usize_sample = <Categorical as ::rand::distr::Distribution<usize>>::sample(
                    &cat, &mut FixedWord(word));
                let u64_sample = <Categorical as ::rand::distr::Distribution<u64>>::sample(
                    &cat, &mut FixedWord(word));
                let f64_sample = <Categorical as ::rand::distr::Distribution<f64>>::sample(
                    &cat, &mut FixedWord(word));
                assert_eq!(usize_sample, expected, "mass={mass:?}, draw={draw:?}");
                assert_eq!(u64_sample, expected as u64);
                assert_eq!(f64_sample, expected as f64);
                assert!(cat.pmf(u64_sample) > 0.0);
            }
        }
    }

    #[cfg(feature = "rand")]
    #[test]
    fn test_sample_seeded_away_from_boundaries() {
        use ::rand::{RngExt, SeedableRng};
        let cat = Categorical::new(&[0.25, 0.25, 0.0, 0.0, 0.5]).unwrap();
        let mut draws = ::rand::rngs::StdRng::seed_from_u64(459);
        let mut samples = ::rand::rngs::StdRng::seed_from_u64(459);
        let mut checked = 0;
        for _ in 0..128 {
            let draw = draws.random::<f64>();
            let actual = <Categorical as ::rand::distr::Distribution<usize>>::sample(
                &cat, &mut samples);
            if !cat.norm_cdf.contains(&draw) {
                let previous = cat.norm_cdf.binary_search_by(|v| v.partial_cmp(&draw).unwrap())
                    .unwrap_or_else(|i| i);
                assert_eq!(actual, previous);
                checked += 1;
            }
        }
        assert!(checked > 0);
    }

    #[test]
    #[should_panic]
    fn test_inverse_cdf_input_nan() {
        Categorical::new(&[1.0, 1.0]).unwrap().inverse_cdf(f64::NAN);
    }

    #[test]
    #[should_panic]
    fn test_inverse_cdf_input_below_zero() {
        Categorical::new(&[1.0, 1.0]).unwrap().inverse_cdf(-0.1);
    }

    #[test]
    #[should_panic]
    fn test_inverse_cdf_input_above_one() {
        Categorical::new(&[1.0, 1.0]).unwrap().inverse_cdf(1.1);
    }

    #[test]
    #[should_panic]
    fn test_inverse_cdf_input_low() {
        let dist = create_ok(&[4.0, 2.5, 2.5, 1.0]);
        dist.inverse_cdf(0.0);
    }

    #[test]
    #[should_panic]
    fn test_inverse_cdf_input_high() {
        let dist = create_ok(&[4.0, 2.5, 2.5, 1.0]);
        dist.inverse_cdf(1.0);
    }

    #[test]
    fn test_discrete() {
        density_util::check_discrete_distribution(&create_ok(&[1.0, 2.0, 3.0, 4.0]), 4);
        density_util::check_discrete_distribution(&create_ok(&[0.0, 1.0, 2.0, 3.0, 4.0]), 5);
    }
}
