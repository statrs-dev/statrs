use crate::distribution::Continuous;
use crate::function::gamma;
use crate::statistics::*;
use crate::{prec, Result, StatsError};
use nalgebra::{Const, Dim, Dyn, OMatrix, OVector};
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
/// use statrs::statistics::Distribution;
/// use nalgebra::DVector;
/// use statrs::statistics::MeanN;
///
/// let n = Dirichlet::new(vec![1.0, 2.0, 3.0]).unwrap();
/// assert_eq!(n.mean().unwrap(), DVector::from_vec(vec![1.0 / 6.0, 1.0 / 3.0, 0.5]));
/// assert_eq!(n.pdf(&DVector::from_vec(vec![0.33333, 0.33333, 0.33333])), 2.222155556222205);
/// ```
#[derive(Clone, PartialEq, Debug)]
pub struct Dirichlet<D>
where
    D: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, D>,
{
    alpha: OVector<f64, D>,
}

impl Dirichlet<Dyn> {
    /// Constructs a new dirichlet distribution with the given
    /// concentration parameters (alpha)
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
    /// use nalgebra::DVector;
    ///
    /// let alpha_ok = vec![1.0, 2.0, 3.0];
    /// let mut result = Dirichlet::new(alpha_ok);
    /// assert!(result.is_ok());
    ///
    /// let alpha_err = vec![0.0];
    /// result = Dirichlet::new(alpha_err);
    /// assert!(result.is_err());
    /// ```
    pub fn new(alpha: Vec<f64>) -> Result<Self> {
        Self::new_from_nalgebra(alpha.into())
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
    pub fn new_with_param(alpha: f64, n: usize) -> Result<Self> {
        Self::new(vec![alpha; n])
    }
}

impl<D> Dirichlet<D>
where
    D: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, D>,
{
    /// Constructs a new distribution with the given vector for `alpha`
    /// Does not clone the vector it takes ownership of
    ///
    /// # Error
    ///
    /// Returns an error if vector has length less than 2 or if any element
    /// of alpha is NOT finite positive
    pub fn new_from_nalgebra(alpha: OVector<f64, D>) -> Result<Self> {
        if !is_valid_alpha(alpha.as_slice()) {
            Err(StatsError::BadParams)
        } else {
            Ok(Self { alpha })
        }
    }

    /// Returns the concentration parameters of
    /// the dirichlet distribution as a slice
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Dirichlet;
    /// use nalgebra::DVector;
    ///
    /// let n = Dirichlet::new(vec![1.0, 2.0, 3.0]).unwrap();
    /// assert_eq!(n.alpha(), &DVector::from_vec(vec![1.0, 2.0, 3.0]));
    /// ```
    pub fn alpha(&self) -> &nalgebra::OVector<f64, D> {
        &self.alpha
    }

    fn alpha_sum(&self) -> f64 {
        self.alpha.sum()
    }

    /// Returns the entropy of the dirichlet distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// ln(B(α)) - (K - α_0)ψ(α_0) - Σ((α_i - 1)ψ(α_i))
    /// ```
    ///
    /// where
    ///
    /// ```text
    /// B(α) = Π(Γ(α_i)) / Γ(Σ(α_i))
    /// ```
    ///
    /// `α_0` is the sum of all concentration parameters,
    /// `K` is the number of concentration parameters, `ψ` is the digamma
    /// function, `α_i`
    /// is the `i`th concentration parameter, and `Σ` is the sum from `1` to `K`
    pub fn entropy(&self) -> Option<f64> {
        let sum = self.alpha_sum();
        let num = self.alpha.iter().fold(0.0, |acc, &x| {
            acc + gamma::ln_gamma(x) + (x - 1.0) * gamma::digamma(x)
        });
        let entr =
            -gamma::ln_gamma(sum) + (sum - self.alpha.len() as f64) * gamma::digamma(sum) - num;
        Some(entr)
    }
}

impl<D> std::fmt::Display for Dirichlet<D>
where
    D: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, D>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Dir({}, {})", self.alpha.len(), &self.alpha)
    }
}

impl<D> ::rand::distributions::Distribution<OVector<f64, D>> for Dirichlet<D>
where
    D: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, D>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> OVector<f64, D> {
        let mut sum = 0.0;
        OVector::from_iterator_generic(
            self.alpha.shape_generic().0,
            Const::<1>,
            self.alpha.iter().map(|&a| {
                let sample = super::gamma::sample_unchecked(rng, a, 1.0);
                sum += sample;
                sample
            }),
        )
    }
}

impl<D> MeanN<OVector<f64, D>> for Dirichlet<D>
where
    D: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, D>,
{
    /// Returns the means of the dirichlet distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// α_i / α_0
    /// ```
    ///
    /// for the `i`th element where `α_i` is the `i`th concentration parameter
    /// and `α_0` is the sum of all concentration parameters
    fn mean(&self) -> Option<OVector<f64, D>> {
        let sum = self.alpha_sum();
        Some(self.alpha.map(|x| x / sum))
    }
}

impl<D> VarianceN<OMatrix<f64, D, D>> for Dirichlet<D>
where
    D: Dim,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<f64, D> + nalgebra::allocator::Allocator<f64, D, D>,
{
    /// Returns the variances of the dirichlet distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// (α_i * (α_0 - α_i)) / (α_0^2 * (α_0 + 1))
    /// ```
    ///
    /// for the `i`th element where `α_i` is the `i`th concentration parameter
    /// and `α_0` is the sum of all concentration parameters
    fn variance(&self) -> Option<OMatrix<f64, D, D>> {
        let sum = self.alpha_sum();
        let normalizing = sum * sum * (sum + 1.0);
        let mut cov = OMatrix::from_diagonal(&self.alpha.map(|x| x * (sum - x) / normalizing));
        let mut offdiag = |x: usize, y: usize| {
            let elt = -self.alpha[x] * self.alpha[y] / normalizing;
            cov[(x, y)] = elt;
            cov[(y, x)] = elt;
        };
        for i in 0..self.alpha.len() {
            for j in 0..i {
                offdiag(i, j);
            }
        }
        Some(cov)
    }
}

impl<'a, D> Continuous<&'a OVector<f64, D>, f64> for Dirichlet<D>
where
    D: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, D>
        + nalgebra::allocator::Allocator<f64, D, D>
        + nalgebra::allocator::Allocator<f64, nalgebra::Const<1>, D>,
{
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
    /// ```text
    /// (1 / B(α)) * Π(x_i^(α_i - 1))
    /// ```
    ///
    /// where
    ///
    /// ```text
    /// B(α) = Π(Γ(α_i)) / Γ(Σ(α_i))
    /// ```
    ///
    /// `α` is the vector of concentration parameters, `α_i` is the `i`th
    /// concentration parameter, `x_i` is the `i`th argument corresponding to
    /// the `i`th concentration parameter, `Γ` is the gamma function,
    /// `Π` is the product from `1` to `K`, `Σ` is the sum from `1` to `K`,
    /// and `K` is the number of concentration parameters
    fn pdf(&self, x: &OVector<f64, D>) -> f64 {
        self.ln_pdf(x).exp()
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
    /// ```text
    /// ln((1 / B(α)) * Π(x_i^(α_i - 1)))
    /// ```
    ///
    /// where
    ///
    /// ```text
    /// B(α) = Π(Γ(α_i)) / Γ(Σ(α_i))
    /// ```
    ///
    /// `α` is the vector of concentration parameters, `α_i` is the `i`th
    /// concentration parameter, `x_i` is the `i`th argument corresponding to
    /// the `i`th concentration parameter, `Γ` is the gamma function,
    /// `Π` is the product from `1` to `K`, `Σ` is the sum from `1` to `K`,
    /// and `K` is the number of concentration parameters
    fn ln_pdf(&self, x: &OVector<f64, D>) -> f64 {
        // TODO: would it be clearer here to just do a for loop instead
        // of using iterators?
        if self.alpha.len() != x.len() {
            panic!("Arguments must have correct dimensions.");
        }
        if x.iter().any(|&x| x <= 0.0 || x >= 1.0) {
            panic!("Arguments must be in (0, 1)");
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
            panic!();
        } else {
            term + gamma::ln_gamma(sum_alpha)
        }
    }
}

// determines if `a` is a valid alpha array
// for the Dirichlet distribution
fn is_valid_alpha(a: &[f64]) -> bool {
    a.len() >= 2 && a.iter().all(|&a_i| a_i.is_finite() && a_i > 0.0)
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use nalgebra::{dvector, vector, DimMin, OVector};

    use super::*;
    use crate::distribution::Continuous;

    fn try_create<D>(alpha: OVector<f64, D>) -> Dirichlet<D>
    where
        D: DimMin<D, Output = D>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, D>,
    {
        let mvn = Dirichlet::new_from_nalgebra(alpha);
        assert!(mvn.is_ok());
        mvn.unwrap()
    }

    fn bad_create_case<D>(alpha: OVector<f64, D>)
    where
        D: DimMin<D, Output = D>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, D>,
    {
        let dd = Dirichlet::new_from_nalgebra(alpha);
        assert!(dd.is_err());
    }

    fn test_almost<F, D>(alpha: OVector<f64, D>, expected: f64, acc: f64, eval: F)
    where
        F: FnOnce(Dirichlet<D>) -> f64,
        D: DimMin<D, Output = D>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, D>,
    {
        let dd = try_create(alpha);
        let x = eval(dd);
        assert_almost_eq!(expected, x, acc);
    }

    #[test]
    fn test_is_valid_alpha() {
        assert!(!is_valid_alpha(&[1.0]));
        assert!(!is_valid_alpha(&[1.0, f64::NAN]));
        assert!(is_valid_alpha(&[1.0, 2.0]));
        assert!(!is_valid_alpha(&[1.0, 0.0]));
        assert!(!is_valid_alpha(&[1.0, f64::INFINITY]));
        assert!(!is_valid_alpha(&[-1.0, 2.0]));
    }

    #[test]
    fn test_create() {
        try_create(vector![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(Dirichlet::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]).is_ok());
        // try_create(vector![0.001, f64::INFINITY, 3756.0]); // moved to bad case as this is degenerate
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(vector![1.0]);
        bad_create_case(vector![1.0, 2.0, 0.0, 4.0, 5.0]);
        bad_create_case(vector![1.0, f64::NAN, 3.0, 4.0, 5.0]);
        bad_create_case(vector![0.0, 0.0, 0.0]);
        bad_create_case(vector![0.001, f64::INFINITY, 3756.0]); // moved to bad case as this is degenerate
    }

    // #[test]
    // fn test_mean() {
    //     let n = Dirichlet::new_with_param(0.3, 5).unwrap();
    //     let res = n.mean();
    //     for x in res {
    //         assert_eq!(x, 0.3 / 1.5);
    //     }
    // }

    // #[test]
    // fn test_variance() {
    //     let alpha = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    //     let sum = alpha.iter().fold(0.0, |acc, x| acc + x);
    //     let n = Dirichlet::new(&alpha).unwrap();
    //     let res = n.variance();
    //     for i in 1..11 {
    //         let f = i as f64;
    //         assert_almost_eq!(res[i-1], f * (sum - f) / (sum * sum * (sum + 1.0)), 1e-15);
    //     }
    // }

    // #[test]
    // fn test_std_dev() {
    //     let alpha = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    //     let sum = alpha.iter().fold(0.0, |acc, x| acc + x);
    //     let n = Dirichlet::new(&alpha).unwrap();
    //     let res = n.std_dev();
    //     for i in 1..11 {
    //         let f = i as f64;
    //         assert_almost_eq!(res[i-1], (f * (sum - f) / (sum * sum * (sum + 1.0))).sqrt(), 1e-15);
    //     }
    // }

    #[test]
    fn test_entropy() {
        let entropy = |x: Dirichlet<_>| x.entropy().unwrap();
        test_almost(
            vector![0.1, 0.3, 0.5, 0.8],
            -17.46469081094079,
            1e-30,
            entropy,
        );
        test_almost(
            vector![0.1, 0.2, 0.3, 0.4],
            -21.53881433791513,
            1e-30,
            entropy,
        );
    }

    #[test]
    fn test_pdf() {
        let pdf = |arg| move |x: Dirichlet<_>| x.pdf(&arg);
        test_almost(
            vector![0.1, 0.3, 0.5, 0.8],
            18.77225681167061,
            1e-12,
            pdf([0.01, 0.03, 0.5, 0.46].into()),
        );
        test_almost(
            vector![0.1, 0.3, 0.5, 0.8],
            0.8314656481199253,
            1e-14,
            pdf([0.1, 0.2, 0.3, 0.4].into()),
        );
    }

    #[test]
    fn test_ln_pdf() {
        let ln_pdf = |arg| move |x: Dirichlet<_>| x.ln_pdf(&arg);
        test_almost(
            vector![0.1, 0.3, 0.5, 0.8],
            18.77225681167061_f64.ln(),
            1e-12,
            ln_pdf([0.01, 0.03, 0.5, 0.46].into()),
        );
        test_almost(
            vector![0.1, 0.3, 0.5, 0.8],
            0.8314656481199253_f64.ln(),
            1e-14,
            ln_pdf([0.1, 0.2, 0.3, 0.4].into()),
        );
    }

    #[test]
    #[should_panic]
    fn test_pdf_bad_input_length() {
        let n = try_create(dvector![0.1, 0.3, 0.5, 0.8]);
        n.pdf(&dvector![0.5]);
    }

    #[test]
    #[should_panic]
    fn test_pdf_bad_input_range() {
        let n = try_create(vector![0.1, 0.3, 0.5, 0.8]);
        n.pdf(&vector![1.5, 0.0, 0.0, 0.0]);
    }

    #[test]
    #[should_panic]
    fn test_pdf_bad_input_sum() {
        let n = try_create(vector![0.1, 0.3, 0.5, 0.8]);
        n.pdf(&vector![0.5, 0.25, 0.8, 0.9]);
    }

    #[test]
    #[should_panic]
    fn test_ln_pdf_bad_input_length() {
        let n = try_create(dvector![0.1, 0.3, 0.5, 0.8]);
        n.ln_pdf(&dvector![0.5]);
    }

    #[test]
    #[should_panic]
    fn test_ln_pdf_bad_input_range() {
        let n = try_create(vector![0.1, 0.3, 0.5, 0.8]);
        n.ln_pdf(&vector![1.5, 0.0, 0.0, 0.0]);
    }

    #[test]
    #[should_panic]
    fn test_ln_pdf_bad_input_sum() {
        let n = try_create(vector![0.1, 0.3, 0.5, 0.8]);
        n.ln_pdf(&vector![0.5, 0.25, 0.8, 0.9]);
    }
}
