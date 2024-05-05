use crate::distribution::{
    Continuous, ContinuousCDF, ContinuousMultivariateCDF, MultivariateUniform, Normal,
};
use crate::statistics::{Max, MeanN, Min, Mode, VarianceN};
use crate::{Result, StatsError};
use nalgebra::{
    base::allocator::Allocator, base::dimension::DimName, Cholesky, DefaultAllocator, Dim, DimMin,
    LU, U1,
};
use nalgebra::{DMatrix, DVector};
use primes::{PrimeSet, Sieve};
use rand::distributions::Distribution;
use rand::Rng;
use std::f64;
use std::f64::consts::{E, PI};

/// Implements the [Multivariate Normal](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
/// distribution using the "nalgebra" crate for matrix operations
///
/// # Examples
///
/// ```
/// use statrs::distribution::{MultivariateNormal, Continuous};
/// use nalgebra::{DVector, DMatrix};
/// use statrs::statistics::{MeanN, VarianceN};
///
/// let mvn = MultivariateNormal::new(vec![0., 0.], vec![1., 0., 0., 1.]).unwrap();
/// assert_eq!(mvn.mean().unwrap(), DVector::from_vec(vec![0., 0.]));
/// assert_eq!(mvn.variance().unwrap(), DMatrix::from_vec(2, 2, vec![1., 0., 0., 1.]));
/// assert_eq!(mvn.pdf(&DVector::from_vec(vec![1.,  1.])), 0.05854983152431917);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct MultivariateNormal {
    dim: usize,
    cov_chol_decomp: DMatrix<f64>,
    mu: DVector<f64>,
    cov: DMatrix<f64>,
    precision: DMatrix<f64>,
    pdf_const: f64,
}

impl MultivariateNormal {
    /// Constructs a new multivariate normal distribution with a mean of `mean`
    /// and covariance matrix `cov`
    ///
    /// # Errors
    ///
    /// Returns an error if the given covariance matrix is not
    /// symmetric or positive-definite
    pub fn new(mean: Vec<f64>, cov: Vec<f64>) -> Result<Self> {
        let mean = DVector::from_vec(mean);
        let cov = DMatrix::from_vec(mean.len(), mean.len(), cov);
        MultivariateNormal::new_from_nalgebra(mean, cov)
    }

    /// Constructs a new multivariate normal distribution with a mean of `mean`
    /// and covariance matrix `cov`, but with explicitly using nalgebras
    /// DVector and DMatrix instead of Vec<f64>
    ///
    /// # Errors
    ///
    /// Returns an error if the given covariance matrix is not
    /// symmetric or positive-definite
    pub fn new_from_nalgebra(mean: DVector<f64>, cov: DMatrix<f64>) -> Result<Self> {
        let dim = mean.len();
        // Check that the provided covariance matrix is symmetric
        if cov.lower_triangle() != cov.upper_triangle().transpose()
        // Check that mean and covariance do not contain NaN
            || mean.iter().any(|f| f.is_nan())
            || cov.iter().any(|f| f.is_nan())
        // Check that the dimensions match
            || mean.nrows() != cov.nrows() || cov.nrows() != cov.ncols()
        {
            return Err(StatsError::BadParams);
        }
        let cov_det = cov.determinant();
        let pdf_const = ((2. * PI).powi(mean.nrows() as i32) * cov_det.abs())
            .recip()
            .sqrt();
        // Store the Cholesky decomposition of the covariance matrix
        // for sampling
        match Cholesky::new(cov.clone()) {
            None => Err(StatsError::BadParams),
            Some(cholesky_decomp) => {
                let precision = cholesky_decomp.inverse();
                Ok(MultivariateNormal {
                    dim,
                    cov_chol_decomp: cholesky_decomp.unpack(),
                    mu: mean,
                    cov,
                    precision,
                    pdf_const,
                })
            }
        }
    }

    /// Returns the entropy of the multivariate normal distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// (1 / 2) * ln(det(2 * π * e * Σ))
    /// ```
    ///
    /// where `Σ` is the covariance matrix and `det` is the determinant
    pub fn entropy(&self) -> Option<f64> {
        Some(
            0.5 * self
                .variance()
                .unwrap()
                .scale(2. * PI * E)
                .determinant()
                .ln(),
        )
    }

    /// Returns the lower triangular cholesky decomposition of
    /// self.cov wrt. switching of rows and columns of the matrix as well as mutating
    /// the input `a` and `b` with the row switches
    ///
    /// Algorithm explained in 4.1.3 in ´Computation of Multivariate
    /// Normal and t Probabilities´, Alan Genz.
    fn chol_chrows(&self, a: &mut DVector<f64>, b: &mut DVector<f64>) -> DMatrix<f64> {
        let mut cov = self.cov.clone();
        let mut chol_lower: DMatrix<f64> = DMatrix::zeros(self.dim, self.dim);
        let mut y: DVector<f64> = DVector::zeros(self.dim);

        let std_normal = Normal::new(0., 1.).unwrap();
        for i in 0..self.dim {
            let mut cdf_diff = f64::INFINITY;
            let mut new_i = i;
            let mut a_tilde = a[i];
            let mut b_tilde = b[i];
            // Find the index of which to switch rows with
            for j in i..self.dim {
                let mut num = 0.;
                let mut den = cov[(j, j)].sqrt();
                if i > 0 {
                    // Numerator:
                    // -Σ lᵢₘyₘ, sum from m=1 to i-1
                    num = (chol_lower.index((j, ..i)) * y.index((..i, 0)))[0];
                    // Denominator:
                    // √(σᵢᵢ - Σ lᵢₘ²), sum from m=1 to i-1
                    den = (cov[(j, j)]
                        - (chol_lower.index((j, ..i)).transpose() * chol_lower.index((j, ..i)))[0])
                        .sqrt();
                }
                let pot_a_tilde = (a[j] - num) / den;
                let pot_b_tilde = (b[j] - num) / den;
                let cdf_a = std_normal.cdf(pot_a_tilde);
                let cdf_b = std_normal.cdf(pot_b_tilde);

                let pot_cdf_diff = cdf_b - cdf_a; // Potential minimum
                if pot_cdf_diff < cdf_diff {
                    new_i = j;
                    cdf_diff = pot_cdf_diff;
                    a_tilde = pot_a_tilde;
                    b_tilde = pot_b_tilde;
                }
            }
            if i != new_i {
                cov.swap_rows(i, new_i);
                cov.swap_columns(i, new_i);
                a.swap_rows(i, new_i);
                b.swap_rows(i, new_i);
                chol_lower.swap_rows(i, new_i);
                chol_lower.swap_columns(i, new_i);
            }

            // Get the expected values:
            // yᵢ = 1 / (Ψ(bᵢ) - Ψ(𝑎ᵢ)) * ∫_𝑎ᵢ^𝑏ᵢ sψ(s) ds
            y[i] = ((-a_tilde.powi(2) / 2.).exp() - (-b_tilde.powi(2) / 2.).exp())
                / ((2. * PI).sqrt() * cdf_diff);

            // Cholesky decomposition algorithm with the new changed row
            let mut ids = chol_lower.index_mut((.., ..i + 1)); // Get only the relevant indices
            ids[(i, i)] =
                (cov[(i, i)] - (ids.index((i, ..i)) * ids.index((i, ..i)).transpose())[0]).sqrt();
            for j in i + 1..self.dim {
                ids[(j, i)] = (cov[(j, i)]
                    - (ids.index((i, ..i)) * ids.index((j, ..i)).transpose())[0])
                    / ids[(i, i)];
            }
        }
        chol_lower
    }

    /// Uses the algorithm as explained in
    /// 'Computation of Multivariate Normal and t Probabilites', Section 4.2.2,
    /// by Alan Genz.
    fn integrate_pdf(&self, a: &mut DVector<f64>, b: &mut DVector<f64>) -> (f64, f64) {
        let chol_lower = self.chol_chrows(a, b);

        // Generate first `dim` primes, Ricthmyer generators
        // Could write function in crate for this instead if we
        // want less imports. Efficiency here does not matter much
        let mut sqrt_primes = DVector::zeros(self.dim);
        let mut pset = Sieve::new();
        for (i, n) in pset.iter().enumerate().take(self.dim) {
            sqrt_primes[i] = (n as f64).sqrt();
        }

        let n_samples = 15;
        let n_points = 1000 * self.dim;
        let mvu = MultivariateUniform::standard(self.dim).unwrap();
        let std_normal = Normal::new(0., 1.).unwrap();
        let mut rng = rand::thread_rng();

        let one = DVector::from_vec(vec![1.; self.dim]);
        let mut y: DVector<f64> = DVector::zeros(self.dim - 1);

        let alpha = 3.;
        let mut err = 0.;
        let mut err_help = 0.;
        let mut p = 0.; // The cdf probability
        for i in 0..n_samples {
            let rnd_points = mvu.sample(&mut rng).unwrap();
            let mut sum_i = 0.;
            for j in 0..n_points {
                let w =
                    (2. * DVector::from_vec(
                        ((j as f64) * &sqrt_primes + &rnd_points)
                            .iter()
                            .map(|x| x % 1.)
                            .collect::<Vec<f64>>(),
                    ) - &one)
                        .abs();
                let mut di = std_normal.cdf(a[0] / chol_lower[(0, 0)]);
                let mut ei = std_normal.cdf(b[0] / chol_lower[(0, 0)]);
                let mut fi = ei - di;
                for m in 1..self.dim {
                    y[m - 1] = std_normal.inverse_cdf(di + w[m - 1] * (ei - di));
                    let mut num = (chol_lower.index((m, ..m)) * y.index((..m, 0)))[0];
                    let den = chol_lower[(m, m)];
                    if num.is_nan() {
                        // Either -inf, 0 or inf, comes when yᵢ = -inf and chol_lowerₘᵢ = 0
                        num = 0.;
                    }
                    di = std_normal.cdf((a[m] - num) / den);
                    ei = std_normal.cdf((b[m] - num) / den);
                    fi *= ei - di;
                }
                sum_i += (fi - sum_i) / ((j + 1) as f64);
            }
            let delta = (sum_i - p) / ((i + 1) as f64);
            p += delta;
            err_help = (((i - 1) as f64) * err_help / ((i + 1) as f64)) + delta.powi(2);
            err = alpha * err_help.sqrt()
        }
        return (p, err);
    }
}

impl ::rand::distributions::Distribution<DVector<f64>> for MultivariateNormal {
    /// Samples from the multivariate normal distribution
    ///
    /// # Formula
    /// ```text
    /// L * Z + μ
    /// ```
    ///
    /// where `L` is the Cholesky decomposition of the covariance matrix,
    /// `Z` is a vector of normally distributed random variables, and
    /// `μ` is the mean vector

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> DVector<f64> {
        let d = Normal::new(0., 1.).unwrap();
        let z = DVector::<f64>::from_distribution(self.dim, &d, rng);
        (&self.cov_chol_decomp * z) + &self.mu
    }
}

impl ContinuousMultivariateCDF<f64, f64> for MultivariateNormal {
    /// Returns the cumulative distribution function at `x` for the
    /// multivariate normal distribution
    fn cdf(&self, mut x: DVector<f64>) -> f64 {
        // Shift integration limit wrt. mean
        x -= &self.mu;
        let (p, _) = self.integrate_pdf(
            &mut DVector::from_vec(vec![f64::NEG_INFINITY; self.dim]),
            &mut x,
        );
        p
    }

    /// Returns the survival function at `x` for the
    /// multivariate normal distribution, using approximation with
    /// `N` points.
    fn sf(&self, x: DVector<f64>) -> f64 {
        // Shift integration limit wrt. mean
        1. - self.cdf(x)
    }
}

impl Min<DVector<f64>> for MultivariateNormal {
    /// Returns the minimum value in the domain of the
    /// multivariate normal distribution represented by a real vector
    fn min(&self) -> DVector<f64> {
        DVector::from_vec(vec![f64::NEG_INFINITY; self.dim])
    }
}

impl Max<DVector<f64>> for MultivariateNormal {
    /// Returns the maximum value in the domain of the
    /// multivariate normal distribution represented by a real vector
    fn max(&self) -> DVector<f64> {
        DVector::from_vec(vec![f64::INFINITY; self.dim])
    }
}

impl MeanN<DVector<f64>> for MultivariateNormal {
    /// Returns the mean of the normal distribution
    ///
    /// # Remarks
    ///
    /// This is the same mean used to construct the distribution
    fn mean(&self) -> Option<DVector<f64>> {
        let mut vec = vec![];
        for elt in self.mu.clone().into_iter() {
            vec.push(*elt);
        }
        Some(DVector::from_vec(vec))
    }
}

impl VarianceN<DMatrix<f64>> for MultivariateNormal {
    /// Returns the covariance matrix of the multivariate normal distribution
    fn variance(&self) -> Option<DMatrix<f64>> {
        Some(self.cov.clone())
    }
}

impl Mode<DVector<f64>> for MultivariateNormal {
    /// Returns the mode of the multivariate normal distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// μ
    /// ```
    ///
    /// where `μ` is the mean
    fn mode(&self) -> DVector<f64> {
        self.mu.clone()
    }
}

impl<'a> Continuous<&'a DVector<f64>, f64> for MultivariateNormal {
    /// Calculates the probability density function for the multivariate
    /// normal distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// (2 * π) ^ (-k / 2) * det(Σ) ^ (1 / 2) * e ^ ( -(1 / 2) * transpose(x - μ) * inv(Σ) * (x - μ))
    /// ```
    ///
    /// where `μ` is the mean, `inv(Σ)` is the precision matrix, `det(Σ)` is the determinant
    /// of the covariance matrix, and `k` is the dimension of the distribution
    fn pdf(&self, x: &'a DVector<f64>) -> f64 {
        let dv = x - &self.mu;
        let exp_term = -0.5
            * *(&dv.transpose() * &self.precision * &dv)
                .get((0, 0))
                .unwrap();
        self.pdf_const * exp_term.exp()
    }
    /// Calculates the log probability density function for the multivariate
    /// normal distribution at `x`. Equivalent to pdf(x).ln().
    fn ln_pdf(&self, x: &'a DVector<f64>) -> f64 {
        let dv = x - &self.mu;
        let exp_term = -0.5
            * *(&dv.transpose() * &self.precision * &dv)
                .get((0, 0))
                .unwrap();
        self.pdf_const.ln() + exp_term
    }
}

impl Continuous<Vec<f64>, f64> for MultivariateNormal {
    /// Calculates the probability density function for the multivariate
    /// normal distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// (2 * π) ^ (-k / 2) * det(Σ) ^ (1 / 2) * e ^ ( -(1 / 2) * transpose(x - μ) * inv(Σ) * (x - μ))
    /// ```
    ///
    /// where `μ` is the mean, `inv(Σ)` is the precision matrix, `det(Σ)` is the determinant
    /// of the covariance matrix, and `k` is the dimension of the distribution
    fn pdf(&self, x: Vec<f64>) -> f64 {
        self.pdf(&DVector::from(x))
    }
    /// Calculates the log probability density function for the multivariate
    /// normal distribution at `x`. Equivalent to pdf(x).ln().
    fn ln_pdf(&self, x: Vec<f64>) -> f64 {
        self.pdf(&DVector::from(x))
    }
}

#[rustfmt::skip]
#[cfg(all(test, feature = "nightly"))]
mod tests  {
    use crate::distribution::{Continuous, MultivariateNormal};
    use crate::statistics::*;
    use crate::consts::ACC;
    use core::fmt::Debug;
    use nalgebra::base::allocator::Allocator;
    use nalgebra::{
        DefaultAllocator, Dim, DimMin, DimName, Matrix2, Matrix3, Vector2, Vector3,
        U1, U2,
    };

    fn try_create(mean: Vec<f64>, covariance: Vec<f64>) -> MultivariateNormal
    {
        let mvn = MultivariateNormal::new(mean, covariance);
        assert!(mvn.is_ok());
        mvn.unwrap()
    }

    fn create_case(mean: Vec<f64>, covariance: Vec<f64>)
    {
        let mvn = try_create(mean.clone(), covariance.clone());
        assert_eq!(DVector::from_vec(mean.clone()), mvn.mean().unwrap());
        assert_eq!(DMatrix::from_vec(mean.len(), mean.len(), covariance), mvn.variance().unwrap());
    }

    fn bad_create_case(mean: Vec<f64>, covariance: Vec<f64>)
    {
        let mvn = MultivariateNormal::new(mean, covariance);
        assert!(mvn.is_err());
    }

    fn test_case<T, F>(mean: Vec<f64>, covariance: Vec<f64>, expected: T, eval: F)
    where
        T: Debug + PartialEq,
        F: FnOnce(MultivariateNormal) -> T,
    {
        let mvn = try_create(mean, covariance);
        let x = eval(mvn);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(
        mean: Vec<f64>,
        covariance: Vec<f64>,
        expected: f64,
        acc: f64,
        eval: F,
    ) where
        F: FnOnce(MultivariateNormal) -> f64,
    {
        let mvn = try_create(mean, covariance);
        let x = eval(mvn);
        assert_almost_eq!(expected, x, acc);
    }

    fn identity_vec(dim: usize) -> Vec<f64> {
        let id: DMatrix<f64> = DMatrix::identity(dim,dim);
        return id.data.into();
    }

    fn cov_matrix(dim: usize, diag_factor: f64, off_diag_factor: f64) -> Vec<f64> {
        let id = DMatrix::from_diagonal_element(dim,dim,diag_factor);
        let rho = DMatrix::repeat(dim, dim, off_diag_factor);
        return (id - DMatrix::identity(dim,dim)*off_diag_factor + rho).data.into()
    }

    use super::*;

    macro_rules! dvec {
        ($($x:expr),*) => {
            DVector::from_vec(vec![$($x),*])
        };
        ($($x:expr)?;$($y:expr)?) => {
            DVector::from_vec(vec![$($x)?;$($y)?])
        };
    }

    macro_rules! mat2 {
        ($x11:expr, $x12:expr, $x21:expr, $x22:expr) => (DMatrix::from_vec(2,2,vec![$x11, $x12, $x21, $x22]));
    }

    // macro_rules! mat3 {
    //     ($x11:expr, $x12:expr, $x13:expr, $x21:expr, $x22:expr, $x23:expr, $x31:expr, $x32:expr, $x33:expr) => (DMatrix::from_vec(3,3,vec![$x11, $x12, $x13, $x21, $x22, $x23, $x31, $x32, $x33]));
    // }

    #[test]
    fn test_create() {
        create_case(vec![0., 0.], vec![1., 0., 0., 1.]);
        create_case(vec![10.,  5.], vec![2., 1., 1., 2.]);
        create_case(vec![4., 5., 6.], vec![2., 1., 0., 1., 2., 1., 0., 1., 2.]);
        create_case(vec![0., f64::INFINITY], vec![1., 0., 0., 1.]);
        create_case(vec![0., 0.], vec![f64::INFINITY, 0., 0., f64::INFINITY]);
    }

    #[test]
    fn test_bad_create() {
        // Covariance not symmetric
        bad_create_case(vec![0., 0.], vec![1., 1., 0., 1.]);
        // Covariance not positive-definite
        bad_create_case(vec![0., 0.], vec![1., 2., 2., 1.]);
        // NaN in mean
        bad_create_case(vec![0., f64::NAN], vec![1., 0., 0., 1.]);
        // NaN in Covariance Matrix
        bad_create_case(vec![0., 0.], vec![1., 0., 0., f64::NAN]);
    }

    #[test]
    fn test_variance() {
        let variance = |x: MultivariateNormal| x.variance().unwrap();
        test_case(vec![0., 0.], vec![1., 0., 0., 1.], mat2![1., 0., 0., 1.], variance);
        test_case(vec![0., 0.], vec![f64::INFINITY, 0., 0., f64::INFINITY], mat2![f64::INFINITY, 0., 0., f64::INFINITY], variance);
    }

    #[test]
    fn test_entropy() {
        let entropy = |x: MultivariateNormal| x.entropy().unwrap();
        test_case(vec![0., 0.], vec![1., 0., 0., 1.], 2.8378770664093453, entropy);
        test_case(vec![0., 0.], vec![1., 0.5, 0.5, 1.], 2.694036030183455, entropy);
        test_case(vec![0., 0.], vec![f64::INFINITY, 0., 0., f64::INFINITY], f64::INFINITY, entropy);
    }

    #[test]
    fn test_mode() {
        let mode = |x: MultivariateNormal| x.mode();
        test_case(vec![0., 0.], vec![1., 0., 0., 1.], dvec![0.,  0.], mode);
        test_case(vec![f64::INFINITY, f64::INFINITY], vec![1., 0., 0., 1.], dvec![f64::INFINITY,  f64::INFINITY], mode);
    }

    #[test]
    fn test_min_max() {
        let min = |x: MultivariateNormal| x.min();
        let max = |x: MultivariateNormal| x.max();
        test_case(vec![0., 0.], vec![1., 0., 0., 1.], dvec![f64::NEG_INFINITY, f64::NEG_INFINITY], min);
        test_case(vec![0., 0.], vec![1., 0., 0., 1.], dvec![f64::INFINITY, f64::INFINITY], max);
        test_case(vec![10., 1.], vec![1., 0., 0., 1.], dvec![f64::NEG_INFINITY, f64::NEG_INFINITY], min);
        test_case(vec![-3., 5.], vec![1., 0., 0., 1.], dvec![f64::INFINITY, f64::INFINITY], max);
    }

    #[test]
    fn test_pdf() {
        let pdf = |arg: DVector<f64>| move |x: MultivariateNormal| x.pdf(&arg);
        test_case(vec![0., 0.], vec![1., 0., 0., 1.], 0.05854983152431917, pdf(dvec![1., 1.]));
        test_almost(vec![0., 0.], vec![1., 0., 0., 1.], 0.013064233284684921, 1e-15, pdf(dvec![1., 2.]));
        test_almost(vec![0., 0.], vec![1., 0., 0., 1.], 1.8618676045881531e-23, 1e-35, pdf(dvec![1., 10.]));
        test_almost(vec![0., 0.], vec![1., 0., 0., 1.], 5.920684802611216e-45, 1e-58, pdf(dvec![10., 10.]));
        test_almost(vec![0., 0.], vec![1., 0.9, 0.9, 1.], 1.6576716577547003e-05, 1e-18, pdf(dvec![1., -1.]));
        test_almost(vec![0., 0.], vec![1., 0.99, 0.99, 1.], 4.1970621773477824e-44, 1e-54, pdf(dvec![1., -1.]));
        test_almost(vec![0.5, -0.2], vec![2.0, 0.3, 0.3,  0.5], 0.0013075203140666656, 1e-15, pdf(dvec![2., 2.]));
        test_case(vec![0., 0.], vec![f64::INFINITY, 0., 0., f64::INFINITY], 0.0, pdf(dvec![10., 10.]));
        test_case(vec![0., 0.], vec![f64::INFINITY, 0., 0., f64::INFINITY], 0.0, pdf(dvec![100., 100.]));
    }

    #[test]
    fn test_ln_pdf() {
        let ln_pdf = |arg: DVector<_>| move |x: MultivariateNormal| x.ln_pdf(&arg);
        test_case(vec![0., 0.], vec![1., 0., 0., 1.], (0.05854983152431917f64).ln(), ln_pdf(dvec![1., 1.]));
        test_almost(vec![0., 0.], vec![1., 0., 0., 1.], (0.013064233284684921f64).ln(), 1e-15, ln_pdf(dvec![1., 2.]));
        test_almost(vec![0., 0.], vec![1., 0., 0., 1.], (1.8618676045881531e-23f64).ln(), 1e-15, ln_pdf(dvec![1., 10.]));
        test_almost(vec![0., 0.], vec![1., 0., 0., 1.], (5.920684802611216e-45f64).ln(), 1e-15, ln_pdf(dvec![10., 10.]));
        test_almost(vec![0., 0.], vec![1., 0.9, 0.9, 1.], (1.6576716577547003e-05f64).ln(), 1e-14, ln_pdf(dvec![1., -1.]));
        test_almost(vec![0., 0.], vec![1., 0.99, 0.99, 1.], (4.1970621773477824e-44f64).ln(), 1e-12, ln_pdf(dvec![1., -1.]));
        test_almost(vec![0.5, -0.2], vec![2.0, 0.3, 0.3, 0.5],  (0.0013075203140666656f64).ln(), 1e-15, ln_pdf(dvec![2., 2.]));
        test_case(vec![0., 0.], vec![f64::INFINITY, 0., 0., f64::INFINITY], f64::NEG_INFINITY, ln_pdf(dvec![10., 10.]));
        test_case(vec![0., 0.], vec![f64::INFINITY, 0., 0., f64::INFINITY], f64::NEG_INFINITY, ln_pdf(dvec![100., 100.]));
    }

    #[test]
    fn test_cdf() {
        let cdf = |arg: DVector<_>| move |x: MultivariateNormal| x.cdf(arg);
        test_case(vec![0., 0., 0.], identity_vec(3), 0., cdf(dvec![f64::NEG_INFINITY; 3]));
        test_case(vec![0., 0., 0.], identity_vec(3), 1., cdf(dvec![f64::INFINITY; 3]));
        test_case(vec![1., -1., 10.], identity_vec(3), 1., cdf(dvec![f64::INFINITY; 3]));
        test_almost(vec![0., 0., 0.], cov_matrix(3, 1., 0.1), 0.119415222, 1e-5, cdf(dvec![-0.1; 3]));
        test_almost(vec![-5., 0., 5.], cov_matrix(3, 1., 0.5), 0.23397186, 1e-5, cdf(dvec![-2., 1., 4.3]));
        test_almost(vec![1., 0., 2.], cov_matrix(3, 1., 0.9), 0.0663303, 1e-5, cdf(dvec![0.5; 3]));
        test_almost(vec![-0.5, 1.1], cov_matrix(2, 1., 0.2), 0.700540224, 1e-5, cdf(dvec![0.5, 2.]));
        test_almost(vec![10., 10., 10.], cov_matrix(3, 5., 1.5), 0.5945585970, 1e-5, cdf(dvec![12.; 3]));
        test_almost(vec![1.; 4], cov_matrix(4, 2., 0.5), 0.1264796225, 1e-5, cdf(dvec![1.; 4]));
        test_almost(vec![1.; 15], cov_matrix(15, 2., 0.5), 0.011545573, 1e-5, cdf(dvec![1.; 15]));
        test_almost(vec![-100., -150., 150.], vec![1., 0.2, 0.9, 0.2, 1., 0.3, 0.9, 0.3, 1.], 0.999999713, 1e-5, cdf(dvec![-90., -140., 155.]));
        test_almost(vec![0.5,0.2,1.1], vec![1., 0.2, 0.9, 0.2, 1., 0.3, 0.9, 0.3, 1.], 0.07532228836, 1e-5, cdf(dvec![-0.9, 1.3, 2.2]));
    }
}
