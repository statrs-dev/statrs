use crate::distribution::{CheckedContinuous, Continuous};
use crate::function::gamma;
use crate::statistics::*;
use crate::{prec, Result, StatsError};
use rand::distributions::Distribution;
use rand::Rng;
use std::f64;

/// Implements the
/// [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Dirichlet, Continuous};
/// use statrs::statistics::Mean;
///
/// let n = Dirichlet::new(&[1.0, 2.0, 3.0]).unwrap();
/// assert_eq!(n.mean(), [1.0 / 6.0, 1.0 / 3.0, 0.5]);
/// assert_eq!(n.pdf(&[0.33333, 0.33333, 0.33333]), 2.222155556222205);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Dirichlet {
    alpha: Vec<f64>,
}

impl Dirichlet {
    /// Constructs a new dirichlet distribution with the given
    /// concenctration parameters (alpha)
    ///
    /// # Errors
    ///
    /// Returns an error if any element `x` in alpha exist
    /// such that `x < = 0.0` or `x` is `NaN`, or if the length of alpha is
    /// less than 2
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Dirichlet;
    ///
    /// let alpha_ok = [1.0, 2.0, 3.0];
    /// let mut result = Dirichlet::new(&alpha_ok);
    /// assert!(result.is_ok());
    ///
    /// let alpha_err = [0.0];
    /// result = Dirichlet::new(&alpha_err);
    /// assert!(result.is_err());
    /// ```
    pub fn new(alpha: &[f64]) -> Result<Dirichlet> {
        if !is_valid_alpha(alpha) {
            Err(StatsError::BadParams)
        } else {
            Ok(Dirichlet {
                alpha: alpha.to_vec(),
            })
        }
    }

    /// Constructs a new dirichlet distribution with the given
    /// concentration parameter (alpha) repeated `n` times
    ///
    /// # Errors
    ///
    /// Returns an error if `alpha < = 0.0` or `alpha` is `NaN`,
    /// or if `n < 2`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Dirichlet;
    ///
    /// let mut result = Dirichlet::new_with_param(1.0, 3);
    /// assert!(result.is_ok());
    ///
    /// result = Dirichlet::new_with_param(0.0, 1);
    /// assert!(result.is_err());
    /// ```
    pub fn new_with_param(alpha: f64, n: usize) -> Result<Dirichlet> {
        Self::new(&vec![alpha; n])
    }

    /// Returns the concentration parameters of
    /// the dirichlet distribution as a slice
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Dirichlet;
    ///
    /// let n = Dirichlet::new(&[1.0, 2.0, 3.0]).unwrap();
    /// assert_eq!(n.alpha(), [1.0, 2.0, 3.0]);
    /// ```
    pub fn alpha(&self) -> &[f64] {
        &self.alpha
    }

    fn alpha_sum(&self) -> f64 {
        self.alpha.iter().fold(0.0, |acc, x| acc + x)
    }
}

impl Distribution<Vec<f64>> for Dirichlet {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<f64> {
        let mut sum = 0.0;
        let mut samples: Vec<f64> = self
            .alpha
            .iter()
            .map(|&a| {
                let sample = super::gamma::sample_unchecked(rng, a, 1.0);
                sum += sample;
                sample
            })
            .collect();
        for _ in samples.iter_mut().map(|x| *x /= sum) {}
        samples
    }
}

impl Mean<Vec<f64>> for Dirichlet {
    /// Returns the means of the dirichlet distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// α_i / α_0
    /// ```
    ///
    /// for the `i`th element where `α_i` is the `i`th concentration parameter
    /// and `α_0` is the sum of all concentration parameters
    fn mean(&self) -> Vec<f64> {
        let sum = self.alpha_sum();
        self.alpha.iter().map(|x| x / sum).collect()
    }
}

impl Variance<Vec<f64>> for Dirichlet {
    /// Returns the variances of the dirichlet distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (α_i * (α_0 - α_i)) / (α_0^2 * (α_0 + 1))
    /// ```
    ///
    /// for the `i`th element where `α_i` is the `i`th concentration parameter
    /// and `α_0` is the sum of all concentration parameters
    fn variance(&self) -> Vec<f64> {
        let sum = self.alpha_sum();
        self.alpha
            .iter()
            .map(|x| x * (sum - x) / (sum * sum * (sum + 1.0)))
            .collect()
    }

    /// Returns the standard deviation of the dirichlet distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// sqrt((α_i * (α_0 - α_i)) / (α_0^2 * (α_0 + 1)))
    /// ```
    ///
    /// for the `i`th element where `α_i` is the `i`th concentration parameter
    /// and `α_0` is the sum of all concentration parameters
    fn std_dev(&self) -> Vec<f64> {
        self.variance().iter().map(|x| x.sqrt()).collect()
    }
}

impl Entropy<f64> for Dirichlet {
    /// Returns the entropy of the dirichlet distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln(B(α)) - (K - α_0)ψ(α_0) - Σ((α_i - 1)ψ(α_i))
    /// ```
    ///
    /// where
    ///
    /// ```ignore
    /// B(α) = Π(Γ(α_i)) / Γ(Σ(α_i))
    /// ```
    ///
    /// `α_0` is the sum of all concentration parameters,
    /// `K` is the number of concentration parameters, `ψ` is the digamma
    /// function, `α_i`
    /// is the `i`th concentration parameter, and `Σ` is the sum from `1` to `K`
    fn entropy(&self) -> f64 {
        let sum = self.alpha_sum();
        let num = self
            .alpha
            .iter()
            .fold(0.0, |acc, &x| acc + (x - 1.0) * gamma::digamma(x));
        gamma::ln_gamma(sum) + (sum - self.alpha.len() as f64) * gamma::digamma(sum) - num
    }
}

impl<'a> Continuous<&'a [f64], f64> for Dirichlet {
    /// Calculates the probabiliy density function for the dirichlet
    /// distribution
    /// with given `x`'s corresponding to the concentration parameters for this
    /// distribution
    ///
    /// # Panics
    ///
    /// If any element in `x` is not in `(0, 1)`, the elements in `x` do not
    /// sum to
    /// `1` with a tolerance of `1e-4`,  or if `x` is not the same length as
    /// the vector of
    /// concentration parameters for this distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (1 / B(α)) * Π(x_i^(α_i - 1))
    /// ```
    ///
    /// where
    ///
    /// ```ignore
    /// B(α) = Π(Γ(α_i)) / Γ(Σ(α_i))
    /// ```
    ///
    /// `α` is the vector of concentration parameters, `α_i` is the `i`th
    /// concentration parameter, `x_i` is the `i`th argument corresponding to
    /// the `i`th concentration parameter, `Γ` is the gamma function,
    /// `Π` is the product from `1` to `K`, `Σ` is the sum from `1` to `K`,
    /// and `K` is the number of concentration parameters
    fn pdf(&self, x: &[f64]) -> f64 {
        self.checked_pdf(x).unwrap()
    }

    /// Calculates the log probabiliy density function for the dirichlet
    /// distribution
    /// with given `x`'s corresponding to the concentration parameters for this
    /// distribution
    ///
    /// # Panics
    ///
    /// If any element in `x` is not in `(0, 1)`, the elements in `x` do not
    /// sum to
    /// `1` with a tolerance of `1e-4`,  or if `x` is not the same length as
    /// the vector of
    /// concentration parameters for this distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln((1 / B(α)) * Π(x_i^(α_i - 1)))
    /// ```
    ///
    /// where
    ///
    /// ```ignore
    /// B(α) = Π(Γ(α_i)) / Γ(Σ(α_i))
    /// ```
    ///
    /// `α` is the vector of concentration parameters, `α_i` is the `i`th
    /// concentration parameter, `x_i` is the `i`th argument corresponding to
    /// the `i`th concentration parameter, `Γ` is the gamma function,
    /// `Π` is the product from `1` to `K`, `Σ` is the sum from `1` to `K`,
    /// and `K` is the number of concentration parameters
    fn ln_pdf(&self, x: &[f64]) -> f64 {
        self.checked_ln_pdf(x).unwrap()
    }
}

impl<'a> CheckedContinuous<&'a [f64], f64> for Dirichlet {
    /// Calculates the probabiliy density function for the dirichlet
    /// distribution
    /// with given `x`'s corresponding to the concentration parameters for this
    /// distribution
    ///
    /// # Errors
    ///
    /// If any element in `x` is not in `(0, 1)`, the elements in `x` do not
    /// sum to
    /// `1` with a tolerance of `1e-4`,  or if `x` is not the same length as
    /// the vector of
    /// concentration parameters for this distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (1 / B(α)) * Π(x_i^(α_i - 1))
    /// ```
    ///
    /// where
    ///
    /// ```ignore
    /// B(α) = Π(Γ(α_i)) / Γ(Σ(α_i))
    /// ```
    ///
    /// `α` is the vector of concentration parameters, `α_i` is the `i`th
    /// concentration parameter, `x_i` is the `i`th argument corresponding to
    /// the `i`th concentration parameter, `Γ` is the gamma function,
    /// `Π` is the product from `1` to `K`, `Σ` is the sum from `1` to `K`,
    /// and `K` is the number of concentration parameters
    fn checked_pdf(&self, x: &[f64]) -> Result<f64> {
        self.checked_ln_pdf(x).map(|x| x.exp())
    }

    /// Calculates the log probabiliy density function for the dirichlet
    /// distribution
    /// with given `x`'s corresponding to the concentration parameters for this
    /// distribution
    ///
    /// # Errors
    ///
    /// If any element in `x` is not in `(0, 1)`, the elements in `x` do not
    /// sum to
    /// `1` with a tolerance of `1e-4`,  or if `x` is not the same length as
    /// the vector of
    /// concentration parameters for this distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln((1 / B(α)) * Π(x_i^(α_i - 1)))
    /// ```
    ///
    /// where
    ///
    /// ```ignore
    /// B(α) = Π(Γ(α_i)) / Γ(Σ(α_i))
    /// ```
    ///
    /// `α` is the vector of concentration parameters, `α_i` is the `i`th
    /// concentration parameter, `x_i` is the `i`th argument corresponding to
    /// the `i`th concentration parameter, `Γ` is the gamma function,
    /// `Π` is the product from `1` to `K`, `Σ` is the sum from `1` to `K`,
    /// and `K` is the number of concentration parameters
    fn checked_ln_pdf(&self, x: &[f64]) -> Result<f64> {
        // TODO: would it be clearer here to just do a for loop instead
        // of using iterators?
        if self.alpha.len() != x.len() {
            return Err(StatsError::ContainersMustBeSameLength);
        }
        if x.iter().any(|&x| x <= 0.0 || x >= 1.0) {
            return Err(StatsError::ArgIntervalExcl("x", 0.0, 1.0));
        }
        let (term, sum_xi, sum_alpha) = x
            .iter()
            .enumerate()
            .map(|pair| (pair.1, self.alpha[pair.0]))
            .fold((0.0, 0.0, 0.0), |acc, pair| {
                (
                    acc.0 + (pair.1 - 1.0) * pair.0.ln() - gamma::ln_gamma(pair.1),
                    acc.1 + pair.0,
                    acc.2 + pair.1,
                )
            });

        if !prec::almost_eq(sum_xi, 1.0, 1e-4) {
            Err(StatsError::ContainerExpectedSum("x", 1.0))
        } else {
            Ok(term + gamma::ln_gamma(sum_alpha))
        }
    }
}

// determines if `a` is a valid alpha array
// for the Dirichlet distribution
fn is_valid_alpha(a: &[f64]) -> bool {
    a.len() >= 2 && super::internal::is_valid_multinomial(a, false)
}

#[test]
fn test_is_valid_alpha() {
    let invalid = [1.0];
    assert!(!is_valid_alpha(&invalid));
}

#[rustfmt::skip]
#[cfg(test)]
mod test {
    use std::f64;
    use crate::function::gamma;
    use crate::statistics::*;
    use crate::distribution::{CheckedContinuous, Continuous, Dirichlet};

    fn try_create(alpha: &[f64]) -> Dirichlet {
        let n = Dirichlet::new(alpha);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(alpha: &[f64]) {
        let n = try_create(alpha);
        let a2 = n.alpha();
        for i in 0..alpha.len() {
            assert_eq!(alpha[i], a2[i]);
        }
    }

    fn bad_create_case(alpha: &[f64]) {
        let n = Dirichlet::new(alpha);
        assert!(n.is_err());
    }

    #[test]
    fn test_create() {
        create_case(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        create_case(&[0.001, f64::INFINITY, 3756.0]);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(&[1.0]);
        bad_create_case(&[1.0, 2.0, 0.0, 4.0, 5.0]);
        bad_create_case(&[1.0, f64::NAN, 3.0, 4.0, 5.0]);
        bad_create_case(&[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_mean() {
        let n = Dirichlet::new_with_param(0.3, 5).unwrap();
        let res = n.mean();
        for x in res {
            assert_eq!(x, 0.3 / 1.5);
        }
    }

    #[test]
    fn test_variance() {
        let alpha = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let sum = alpha.iter().fold(0.0, |acc, x| acc + x);
        let n = Dirichlet::new(&alpha).unwrap();
        let res = n.variance();
        for i in 1..11 {
            let f = i as f64;
            assert_almost_eq!(res[i-1], f * (sum - f) / (sum * sum * (sum + 1.0)), 1e-15);
        }
    }

    #[test]
    fn test_std_dev() {
        let alpha = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let sum = alpha.iter().fold(0.0, |acc, x| acc + x);
        let n = Dirichlet::new(&alpha).unwrap();
        let res = n.std_dev();
        for i in 1..11 {
            let f = i as f64;
            assert_almost_eq!(res[i-1], (f * (sum - f) / (sum * sum * (sum + 1.0))).sqrt(), 1e-15);
        }
    }

    #[test]
    fn test_entropy() {
        let mut alpha = [0.1, 0.3, 0.5, 0.8];
        let mut n = try_create(&[0.1, 0.3, 0.5, 0.8]);
        let mut sum = alpha.iter().fold(0.0, |acc, x| acc + x);
        let mut num = alpha.iter().fold(0.0, |acc, x| acc + (x - 1.0) * gamma::digamma(*x));
        assert_eq!(n.entropy(), gamma::ln_gamma(sum) + (sum - 4.0) * gamma::digamma(sum) - num);

        alpha = [0.1, 0.2, 0.3, 0.4];
        n = try_create(&alpha);
        sum = alpha.iter().fold(0.0, |acc, x| acc + x);
        num = alpha.iter().fold(0.0, |acc, x| acc + (x - 1.0) * gamma::digamma(*x));
        assert_eq!(n.entropy(), gamma::ln_gamma(sum) + (sum - 4.0) * gamma::digamma(sum) - num);
    }

    #[test]
    fn test_pdf() {
        let n = try_create(&[0.1, 0.3, 0.5, 0.8]);
        assert_almost_eq!(n.pdf(&[0.01, 0.03, 0.5, 0.46]), 18.77225681167061, 1e-12);
        assert_almost_eq!(n.pdf(&[0.1,0.2,0.3,0.4]), 0.8314656481199253, 1e-14);
    }

    #[test]
    fn test_ln_pdf() {
        let n = try_create(&[0.1, 0.3, 0.5, 0.8]);
        assert_almost_eq!(n.ln_pdf(&[0.01, 0.03, 0.5, 0.46]), 18.77225681167061f64.ln(), 1e-12);
        assert_almost_eq!(n.ln_pdf(&[0.1,0.2,0.3,0.4]), 0.8314656481199253f64.ln(), 1e-14);
    }

    #[test]
    #[should_panic]
    fn test_pdf_bad_input_length() {
        let n = try_create(&[0.1, 0.3, 0.5, 0.8]);
        n.pdf(&[0.5]);
    }

    #[test]
    #[should_panic]
    fn test_pdf_bad_input_range() {
        let n = try_create(&[0.1, 0.3, 0.5, 0.8]);
        n.pdf(&[1.5, 0.0, 0.0, 0.0]);
    }

    #[test]
    #[should_panic]
    fn test_pdf_bad_input_sum() {
        let n = try_create(&[0.1, 0.3, 0.5, 0.8]);
        n.pdf(&[0.5, 0.25, 0.8, 0.9]);
    }

    #[test]
    #[should_panic]
    fn test_ln_pdf_bad_input_length() {
        let n = try_create(&[0.1, 0.3, 0.5, 0.8]);
        n.ln_pdf(&[0.5]);
    }

    #[test]
    #[should_panic]
    fn test_ln_pdf_bad_input_range() {
        let n = try_create(&[0.1, 0.3, 0.5, 0.8]);
        n.ln_pdf(&[1.5, 0.0, 0.0, 0.0]);
    }

    #[test]
    #[should_panic]
    fn test_ln_pdf_bad_input_sum() {
        let n = try_create(&[0.1, 0.3, 0.5, 0.8]);
        n.ln_pdf(&[0.5, 0.25, 0.8, 0.9]);
    }

    #[test]
    fn test_checked_pdf_bad_input_length() {
        let n = try_create(&[0.1, 0.3, 0.5, 0.8]);
        assert!(n.checked_pdf(&[0.5]).is_err());
    }

    #[test]
    fn test_checked_pdf_bad_input_range() {
        let n = try_create(&[0.1, 0.3, 0.5, 0.8]);
        assert!(n.checked_pdf(&[1.5, 0.0, 0.0, 0.0]).is_err());
    }

    #[test]
    fn test_checked_pdf_bad_input_sum() {
        let n = try_create(&[0.1, 0.3, 0.5, 0.8]);
        assert!(n.checked_pdf(&[0.5, 0.25, 0.8, 0.9]).is_err());
    }

    #[test]
    fn test_checked_ln_pdf_bad_input_length() {
        let n = try_create(&[0.1, 0.3, 0.5, 0.8]);
        assert!(n.checked_ln_pdf(&[0.5]).is_err());
    }

    #[test]
    fn test_checked_ln_pdf_bad_input_range() {
        let n = try_create(&[0.1, 0.3, 0.5, 0.8]);
        assert!(n.checked_ln_pdf(&[1.5, 0.0, 0.0, 0.0]).is_err());
    }

    #[test]
    fn test_checked_ln_pdf_bad_input_sum() {
        let n = try_create(&[0.1, 0.3, 0.5, 0.8]);
        assert!(n.checked_ln_pdf(&[0.5, 0.25, 0.8, 0.9]).is_err());
    }
}
