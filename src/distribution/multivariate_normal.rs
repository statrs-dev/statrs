use crate::distribution::Continuous;
use crate::statistics::{Max, MeanN, Min, Mode, VarianceN};
use crate::StatsError;
use nalgebra::{Cholesky, Const, DMatrix, DVector, Dim, DimMin, Dyn, OMatrix, OVector};
use std::f64;
use std::f64::consts::{E, PI};

/// computes both the normalization and exponential argument in the normal distribution
/// # Errors
/// will error on dimension mismatch
pub(super) fn density_normalization_and_exponential<D>(
    mu: &OVector<f64, D>,
    cov: &OMatrix<f64, D, D>,
    precision: &OMatrix<f64, D, D>,
    x: &OVector<f64, D>,
) -> std::result::Result<(f64, f64), StatsError>
where
    D: DimMin<D, Output = D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, D>
        + nalgebra::allocator::Allocator<f64, D, D>
        + nalgebra::allocator::Allocator<(usize, usize), D>,
{
    Ok((
        density_distribution_pdf_const(mu, cov)?,
        density_distribution_exponential(mu, precision, x)?,
    ))
}

/// computes the argument of the exponential term in the normal distribution
/// ```text
/// ```
/// # Errors
/// will error on dimension mismatch
#[inline]
pub(super) fn density_distribution_exponential<D>(
    mu: &OVector<f64, D>,
    precision: &OMatrix<f64, D, D>,
    x: &OVector<f64, D>,
) -> std::result::Result<f64, StatsError>
where
    D: Dim,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<f64, D> + nalgebra::allocator::Allocator<f64, D, D>,
{
    if x.shape_generic().0 != precision.shape_generic().0
        || x.shape_generic().0 != mu.shape_generic().0
        || !precision.is_square()
    {
        return Err(StatsError::ContainersMustBeSameLength);
    }
    let dv = x - mu;
    let exp_term: f64 = -0.5 * (precision * &dv).dot(&dv);
    Ok(exp_term)
    // TODO update to dimension mismatch error
}

/// computes the argument of the normalization term in the normal distribution
/// # Errors
/// will error on dimension mismatch
#[inline]
pub(super) fn density_distribution_pdf_const<D>(
    mu: &OVector<f64, D>,
    cov: &OMatrix<f64, D, D>,
) -> std::result::Result<f64, StatsError>
where
    D: DimMin<D, Output = D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, D>
        + nalgebra::allocator::Allocator<f64, D, D>
        + nalgebra::allocator::Allocator<(usize, usize), D>,
{
    if cov.shape_generic().0 != mu.shape_generic().0 || !cov.is_square() {
        return Err(StatsError::ContainersMustBeSameLength);
    }
    let cov_det = cov.determinant();
    Ok(((2. * PI).powi(mu.nrows() as i32) * cov_det.abs())
        .recip()
        .sqrt())
}

/// Implements the [Multivariate Normal](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
/// distribution using the "nalgebra" crate for matrix operations
///
/// # Examples
///
/// ```
/// use statrs::distribution::{MultivariateNormal, Continuous};
/// use nalgebra::{matrix, vector};
/// use statrs::statistics::{MeanN, VarianceN};
///
/// let mvn = MultivariateNormal::new_from_nalgebra(vector![0., 0.], matrix![1., 0.; 0., 1.]).unwrap();
/// assert_eq!(mvn.mean().unwrap(), vector![0., 0.]);
/// assert_eq!(mvn.variance().unwrap(), matrix![1., 0.; 0., 1.]);
/// assert_eq!(mvn.pdf(&vector![1.,  1.]), 0.05854983152431917);
/// ```
#[derive(Clone, PartialEq, Debug)]
pub struct MultivariateNormal<D>
where
    D: Dim,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<f64, D> + nalgebra::allocator::Allocator<f64, D, D>,
{
    cov_chol_decomp: OMatrix<f64, D, D>,
    mu: OVector<f64, D>,
    cov: OMatrix<f64, D, D>,
    precision: OMatrix<f64, D, D>,
    pdf_const: f64,
}

/// Represents the errors that can occur when creating a [`MultivariateNormal`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum MultivariateNormalError {
    /// The covariance matrix is asymmetric or contains a NaN.
    CovInvalid,

    /// The mean vector contains a NaN.
    MeanInvalid,

    /// The amount of rows in the vector of means is not equal to the amount
    /// of rows in the covariance matrix.
    DimensionMismatch,

    /// After all other validation, computing the Cholesky decomposition failed.
    CholeskyFailed,
}

impl std::fmt::Display for MultivariateNormalError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            MultivariateNormalError::CovInvalid => {
                write!(f, "Covariance matrix is asymmetric or contains a NaN")
            }
            MultivariateNormalError::MeanInvalid => write!(f, "Mean vector contains a NaN"),
            MultivariateNormalError::DimensionMismatch => write!(
                f,
                "Mean vector and covariance matrix do not have the same number of rows"
            ),
            MultivariateNormalError::CholeskyFailed => {
                write!(f, "Computing the Cholesky decomposition failed")
            }
        }
    }
}

impl std::error::Error for MultivariateNormalError {}

impl MultivariateNormal<Dyn> {
    /// Constructs a new multivariate normal distribution with a mean of `mean`
    /// and covariance matrix `cov`
    ///
    /// # Errors
    ///
    /// Returns an error if the given covariance matrix is not
    /// symmetric or positive-definite
    pub fn new(mean: Vec<f64>, cov: Vec<f64>) -> Result<Self, MultivariateNormalError> {
        let mean = DVector::from_vec(mean);
        let cov = DMatrix::from_vec(mean.len(), mean.len(), cov);
        MultivariateNormal::new_from_nalgebra(mean, cov)
    }
}

impl<D> MultivariateNormal<D>
where
    D: DimMin<D, Output = D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, D>
        + nalgebra::allocator::Allocator<f64, D, D>
        + nalgebra::allocator::Allocator<(usize, usize), D>,
{
    /// Constructs a new multivariate normal distribution with a mean of `mean`
    /// and covariance matrix `cov` using `nalgebra` `OVector` and `OMatrix`
    /// instead of `Vec<f64>`
    ///
    /// # Errors
    ///
    /// Returns an error if the given covariance matrix is not
    /// symmetric or positive-definite
    pub fn new_from_nalgebra(
        mean: OVector<f64, D>,
        cov: OMatrix<f64, D, D>,
    ) -> Result<Self, MultivariateNormalError> {
        if mean.iter().any(|f| f.is_nan()) {
            return Err(MultivariateNormalError::MeanInvalid);
        }

        if !cov.is_square()
            || cov.lower_triangle() != cov.upper_triangle().transpose()
            || cov.iter().any(|f| f.is_nan())
        {
            return Err(MultivariateNormalError::CovInvalid);
        }

        // Compare number of rows
        if mean.shape_generic().0 != cov.shape_generic().0 {
            return Err(MultivariateNormalError::DimensionMismatch);
        }

        // Store the Cholesky decomposition of the covariance matrix
        // for sampling
        match Cholesky::new(cov.clone()) {
            None => Err(MultivariateNormalError::CholeskyFailed),
            Some(cholesky_decomp) => {
                let precision = cholesky_decomp.inverse();
                Ok(MultivariateNormal {
                    // .unwrap() because prerequisites are already checked above
                    pdf_const: density_distribution_pdf_const(&mean, &cov).unwrap(),
                    cov_chol_decomp: cholesky_decomp.unpack(),
                    mu: mean,
                    cov,
                    precision,
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
}

impl<D> std::fmt::Display for MultivariateNormal<D>
where
    D: Dim,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<f64, D> + nalgebra::allocator::Allocator<f64, D, D>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "N({}, {})", &self.mu, &self.cov)
    }
}

#[cfg(feature = "rand")]
impl<D> ::rand::distributions::Distribution<OVector<f64, D>> for MultivariateNormal<D>
where
    D: Dim,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<f64, D> + nalgebra::allocator::Allocator<f64, D, D>,
{
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

    fn sample<R: ::rand::Rng + ?Sized>(&self, rng: &mut R) -> OVector<f64, D> {
        let d = crate::distribution::Normal::new(0., 1.).unwrap();
        let z = OVector::from_distribution_generic(self.mu.shape_generic().0, Const::<1>, &d, rng);
        (&self.cov_chol_decomp * z) + &self.mu
    }
}

impl<D> Min<OVector<f64, D>> for MultivariateNormal<D>
where
    D: Dim,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<f64, D> + nalgebra::allocator::Allocator<f64, D, D>,
{
    /// Returns the minimum value in the domain of the
    /// multivariate normal distribution represented by a real vector
    fn min(&self) -> OVector<f64, D> {
        OMatrix::repeat_generic(self.mu.shape_generic().0, Const::<1>, f64::NEG_INFINITY)
    }
}

impl<D> Max<OVector<f64, D>> for MultivariateNormal<D>
where
    D: Dim,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<f64, D> + nalgebra::allocator::Allocator<f64, D, D>,
{
    /// Returns the maximum value in the domain of the
    /// multivariate normal distribution represented by a real vector
    fn max(&self) -> OVector<f64, D> {
        OMatrix::repeat_generic(self.mu.shape_generic().0, Const::<1>, f64::INFINITY)
    }
}

impl<D> MeanN<OVector<f64, D>> for MultivariateNormal<D>
where
    D: Dim,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<f64, D> + nalgebra::allocator::Allocator<f64, D, D>,
{
    /// Returns the mean of the normal distribution
    ///
    /// # Remarks
    ///
    /// This is the same mean used to construct the distribution
    fn mean(&self) -> Option<OVector<f64, D>> {
        Some(self.mu.clone())
    }
}

impl<D> VarianceN<OMatrix<f64, D, D>> for MultivariateNormal<D>
where
    D: Dim,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<f64, D> + nalgebra::allocator::Allocator<f64, D, D>,
{
    /// Returns the covariance matrix of the multivariate normal distribution
    fn variance(&self) -> Option<OMatrix<f64, D, D>> {
        Some(self.cov.clone())
    }
}

impl<D> Mode<OVector<f64, D>> for MultivariateNormal<D>
where
    D: Dim,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<f64, D> + nalgebra::allocator::Allocator<f64, D, D>,
{
    /// Returns the mode of the multivariate normal distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// μ
    /// ```
    ///
    /// where `μ` is the mean
    fn mode(&self) -> OVector<f64, D> {
        self.mu.clone()
    }
}

impl<'a, D> Continuous<&'a OVector<f64, D>, f64> for MultivariateNormal<D>
where
    D: Dim,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<f64, D> + nalgebra::allocator::Allocator<f64, D, D>,
{
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
    fn pdf(&self, x: &OVector<f64, D>) -> f64 {
        self.pdf_const
            * density_distribution_exponential(&self.mu, &self.precision, x)
                .unwrap()
                .exp()
    }

    /// Calculates the log probability density function for the multivariate
    /// normal distribution at `x`. Equivalent to pdf(x).ln().
    fn ln_pdf(&self, x: &OVector<f64, D>) -> f64 {
        self.pdf_const.ln()
            + density_distribution_exponential(&self.mu, &self.precision, x).unwrap()
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod tests  {
    use core::fmt::Debug;

    use nalgebra::{dmatrix, dvector, matrix, vector, DimMin, OMatrix, OVector};

    use crate::{
        distribution::{Continuous, MultivariateNormal},
        statistics::{Max, MeanN, Min, Mode, VarianceN},
    };

    use super::MultivariateNormalError;

    fn try_create<D>(mean: OVector<f64, D>, covariance: OMatrix<f64, D, D>) -> MultivariateNormal<D>
    where
        D: DimMin<D, Output = D>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, D>
            + nalgebra::allocator::Allocator<f64, D, D>
            + nalgebra::allocator::Allocator<(usize, usize), D>,
    {
        let mvn = MultivariateNormal::new_from_nalgebra(mean, covariance);
        assert!(mvn.is_ok());
        mvn.unwrap()
    }

    fn create_case<D>(mean: OVector<f64, D>, covariance: OMatrix<f64, D, D>)
    where
        D: DimMin<D, Output = D>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, D>
            + nalgebra::allocator::Allocator<f64, D, D>
            + nalgebra::allocator::Allocator<(usize, usize), D>,
    {
        let mvn = try_create(mean.clone(), covariance.clone());
        assert_eq!(mean, mvn.mean().unwrap());
        assert_eq!(covariance, mvn.variance().unwrap());
    }

    fn bad_create_case<D>(mean: OVector<f64, D>, covariance: OMatrix<f64, D, D>)
    where
        D: DimMin<D, Output = D>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, D>
            + nalgebra::allocator::Allocator<f64, D, D>
            + nalgebra::allocator::Allocator<(usize, usize), D>,
    {
        let mvn = MultivariateNormal::new_from_nalgebra(mean, covariance);
        assert!(mvn.is_err());
    }

    fn test_case<T, F, D>(
        mean: OVector<f64, D>, covariance: OMatrix<f64, D, D>, expected: T, eval: F,
    ) where
        T: Debug + PartialEq,
        F: FnOnce(MultivariateNormal<D>) -> T,
        D: DimMin<D, Output = D>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, D>
            + nalgebra::allocator::Allocator<f64, D, D>
            + nalgebra::allocator::Allocator<(usize, usize), D>,
    {
        let mvn = try_create(mean, covariance);
        let x = eval(mvn);
        assert_eq!(expected, x);
    }

    fn test_almost<F, D>(
        mean: OVector<f64, D>, covariance: OMatrix<f64, D, D>, expected: f64, acc: f64, eval: F,
    ) where
        F: FnOnce(MultivariateNormal<D>) -> f64,
        D: DimMin<D, Output = D>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, D>
            + nalgebra::allocator::Allocator<f64, D, D>
            + nalgebra::allocator::Allocator<(usize, usize), D>,
    {
        let mvn = try_create(mean, covariance);
        let x = eval(mvn);
        assert_almost_eq!(expected, x, acc);
    }

    #[test]
    fn test_create() {
        create_case(vector![0., 0.], matrix![1., 0.; 0., 1.]);
        create_case(vector![10., 5.], matrix![2., 1.; 1., 2.]);
        create_case(
            vector![4., 5., 6.],
            matrix![2., 1., 0.; 1., 2., 1.; 0., 1., 2.],
        );
        create_case(dvector![0., f64::INFINITY], dmatrix![1., 0.; 0., 1.]);
        create_case(
            dvector![0., 0.],
            dmatrix![f64::INFINITY, 0.; 0., f64::INFINITY],
        );
    }

    #[test]
    fn test_bad_create() {
        // Covariance not symmetric
        bad_create_case(vector![0., 0.], matrix![1., 1.; 0., 1.]);
        // Covariance not positive-definite
        bad_create_case(vector![0., 0.], matrix![1., 2.; 2., 1.]);
        // NaN in mean
        bad_create_case(dvector![0., f64::NAN], dmatrix![1., 0.; 0., 1.]);
        // NaN in Covariance Matrix
        bad_create_case(dvector![0., 0.], dmatrix![1., 0.; 0., f64::NAN]);
    }

    #[test]
    fn test_variance() {
        let variance = |x: MultivariateNormal<_>| x.variance().unwrap();
        test_case(
            vector![0., 0.],
            matrix![1., 0.; 0., 1.],
            matrix![1., 0.; 0., 1.],
            variance,
        );
        test_case(
            vector![0., 0.],
            matrix![f64::INFINITY, 0.; 0., f64::INFINITY],
            matrix![f64::INFINITY, 0.; 0., f64::INFINITY],
            variance,
        );
    }

    #[test]
    fn test_entropy() {
        let entropy = |x: MultivariateNormal<_>| x.entropy().unwrap();
        test_case(
            dvector![0., 0.],
            dmatrix![1., 0.; 0., 1.],
            2.8378770664093453,
            entropy,
        );
        test_case(
            dvector![0., 0.],
            dmatrix![1., 0.5; 0.5, 1.],
            2.694036030183455,
            entropy,
        );
        test_case(
            dvector![0., 0.],
            dmatrix![f64::INFINITY, 0.; 0., f64::INFINITY],
            f64::INFINITY,
            entropy,
        );
    }

    #[test]
    fn test_mode() {
        let mode = |x: MultivariateNormal<_>| x.mode();
        test_case(
            vector![0., 0.],
            matrix![1., 0.; 0., 1.],
            vector![0., 0.],
            mode,
        );
        test_case(
            vector![f64::INFINITY, f64::INFINITY],
            matrix![1., 0.; 0., 1.],
            vector![f64::INFINITY, f64::INFINITY],
            mode,
        );
    }

    #[test]
    fn test_min_max() {
        let min = |x: MultivariateNormal<_>| x.min();
        let max = |x: MultivariateNormal<_>| x.max();
        test_case(
            dvector![0., 0.],
            dmatrix![1., 0.; 0., 1.],
            dvector![f64::NEG_INFINITY, f64::NEG_INFINITY],
            min,
        );
        test_case(
            dvector![0., 0.],
            dmatrix![1., 0.; 0., 1.],
            dvector![f64::INFINITY, f64::INFINITY],
            max,
        );
        test_case(
            dvector![10., 1.],
            dmatrix![1., 0.; 0., 1.],
            dvector![f64::NEG_INFINITY, f64::NEG_INFINITY],
            min,
        );
        test_case(
            dvector![-3., 5.],
            dmatrix![1., 0.; 0., 1.],
            dvector![f64::INFINITY, f64::INFINITY],
            max,
        );
    }

    #[test]
    fn test_pdf() {
        let pdf = |arg| move |x: MultivariateNormal<_>| x.pdf(&arg);
        test_case(
            vector![0., 0.],
            matrix![1., 0.; 0., 1.],
            0.05854983152431917,
            pdf(vector![1., 1.]),
        );
        test_almost(
            vector![0., 0.],
            matrix![1., 0.; 0., 1.],
            0.013064233284684921,
            1e-15,
            pdf(vector![1., 2.]),
        );
        test_almost(
            vector![0., 0.],
            matrix![1., 0.; 0., 1.],
            1.8618676045881531e-23,
            1e-35,
            pdf(vector![1., 10.]),
        );
        test_almost(
            vector![0., 0.],
            matrix![1., 0.; 0., 1.],
            5.920684802611216e-45,
            1e-58,
            pdf(vector![10., 10.]),
        );
        test_almost(
            vector![0., 0.],
            matrix![1., 0.9; 0.9, 1.],
            1.6576716577547003e-05,
            1e-18,
            pdf(vector![1., -1.]),
        );
        test_almost(
            vector![0., 0.],
            matrix![1., 0.99; 0.99, 1.],
            4.1970621773477824e-44,
            1e-54,
            pdf(vector![1., -1.]),
        );
        test_almost(
            vector![0.5, -0.2],
            matrix![2.0, 0.3; 0.3, 0.5],
            0.0013075203140666656,
            1e-15,
            pdf(vector![2., 2.]),
        );
        test_case(
            vector![0., 0.],
            matrix![f64::INFINITY, 0.; 0., f64::INFINITY],
            0.0,
            pdf(vector![10., 10.]),
        );
        test_case(
            vector![0., 0.],
            matrix![f64::INFINITY, 0.; 0., f64::INFINITY],
            0.0,
            pdf(vector![100., 100.]),
        );
    }

    #[test]
    fn test_ln_pdf() {
        let ln_pdf = |arg| move |x: MultivariateNormal<_>| x.ln_pdf(&arg);
        test_case(
            dvector![0., 0.],
            dmatrix![1., 0.; 0., 1.],
            (0.05854983152431917f64).ln(),
            ln_pdf(dvector![1., 1.]),
        );
        test_almost(
            dvector![0., 0.],
            dmatrix![1., 0.; 0., 1.],
            (0.013064233284684921f64).ln(),
            1e-15,
            ln_pdf(dvector![1., 2.]),
        );
        test_almost(
            dvector![0., 0.],
            dmatrix![1., 0.; 0., 1.],
            (1.8618676045881531e-23f64).ln(),
            1e-15,
            ln_pdf(dvector![1., 10.]),
        );
        test_almost(
            dvector![0., 0.],
            dmatrix![1., 0.; 0., 1.],
            (5.920684802611216e-45f64).ln(),
            1e-15,
            ln_pdf(dvector![10., 10.]),
        );
        test_almost(
            dvector![0., 0.],
            dmatrix![1., 0.9; 0.9, 1.],
            (1.6576716577547003e-05f64).ln(),
            1e-14,
            ln_pdf(dvector![1., -1.]),
        );
        test_almost(
            dvector![0., 0.],
            dmatrix![1., 0.99; 0.99, 1.],
            (4.1970621773477824e-44f64).ln(),
            1e-12,
            ln_pdf(dvector![1., -1.]),
        );
        test_almost(
            dvector![0.5, -0.2],
            dmatrix![2.0, 0.3; 0.3, 0.5],
            (0.0013075203140666656f64).ln(),
            1e-15,
            ln_pdf(dvector![2., 2.]),
        );
        test_case(
            dvector![0., 0.],
            dmatrix![f64::INFINITY, 0.; 0., f64::INFINITY],
            f64::NEG_INFINITY,
            ln_pdf(dvector![10., 10.]),
        );
        test_case(
            dvector![0., 0.],
            dmatrix![f64::INFINITY, 0.; 0., f64::INFINITY],
            f64::NEG_INFINITY,
            ln_pdf(dvector![100., 100.]),
        );
    }

    #[test]
    #[should_panic]
    fn test_pdf_mismatched_arg_size() {
        let mvn = MultivariateNormal::new(vec![0., 0.], vec![1., 0., 0., 1.,]).unwrap();
        mvn.pdf(&vec![1.].into()); // x.size != mu.size
    }

    #[test]
    fn test_error_is_sync_send() {
        fn assert_sync_send<T: Sync + Send>() {}
        assert_sync_send::<MultivariateNormalError>();
    }
}
