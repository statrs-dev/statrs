use crate::distribution::Continuous;
use crate::function::gamma;
use crate::statistics::{Max, MeanN, Min, Mode, VarianceN};
use nalgebra::{Cholesky, Const, DMatrix, Dim, DimMin, Dyn, OMatrix, OVector};
use std::f64::consts::PI;

/// Implements the [Multivariate Student's t-distribution](https://en.wikipedia.org/wiki/Multivariate_t-distribution)
/// distribution using the "nalgebra" crate for matrix operations.
///
/// Assumes all the marginal distributions have the same degree of freedom, ν.
///
/// # Examples
///
/// ```
/// use statrs::distribution::{MultivariateStudent, Continuous};
/// use nalgebra::{DVector, DMatrix};
/// use statrs::statistics::{MeanN, VarianceN};
///
/// let mvs = MultivariateStudent::new(vec![0., 0.], vec![1., 0., 0., 1.], 4.).unwrap();
/// assert_eq!(mvs.mean().unwrap(), DVector::from_vec(vec![0., 0.]));
/// assert_eq!(mvs.variance().unwrap(), DMatrix::from_vec(2, 2, vec![2., 0., 0., 2.]));
/// assert_eq!(mvs.pdf(&DVector::from_vec(vec![1.,  1.])), 0.04715702017537655);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct MultivariateStudent<D>
where
    D: Dim,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D> + nalgebra::allocator::Allocator<D, D>,
{
    scale_chol_decomp: OMatrix<f64, D, D>,
    location: OVector<f64, D>,
    scale: OMatrix<f64, D, D>,
    freedom: f64,
    precision: OMatrix<f64, D, D>,
    ln_pdf_const: f64,
}

/// Represents the errors that can occur when creating a [`MultivariateStudent`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum MultivariateStudentError {
    /// The scale matrix is asymmetric or contains a NaN.
    ScaleInvalid,

    /// The location vector contains a NaN.
    LocationInvalid,

    /// The degrees of freedom are NaN, zero or less than zero.
    FreedomInvalid,

    /// The amount of rows in the location vector is not equal to the amount
    /// of rows in the scale matrix.
    DimensionMismatch,

    /// After all other validation, computing the Cholesky decomposition failed.
    /// This means that the scale matrix is not definite-positive.
    CholeskyFailed,
}

impl std::fmt::Display for MultivariateStudentError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            MultivariateStudentError::ScaleInvalid => {
                write!(f, "Scale matrix is asymmetric or contains a NaN")
            }
            MultivariateStudentError::LocationInvalid => {
                write!(f, "Location vector contains a NaN")
            }
            MultivariateStudentError::FreedomInvalid => {
                write!(f, "Degrees of freedom are NaN, zero or less than zero")
            }
            MultivariateStudentError::DimensionMismatch => write!(
                f,
                "Location vector and scale matrix do not have the same number of rows"
            ),
            MultivariateStudentError::CholeskyFailed => {
                write!(f, "Computing the Cholesky decomposition failed")
            }
        }
    }
}

impl std::error::Error for MultivariateStudentError {}

impl MultivariateStudent<Dyn> {
    /// Constructs a new multivariate students t distribution with a location of `location`,
    /// scale matrix `scale` and `freedom` degrees of freedom.
    ///
    /// # Errors
    ///
    /// Returns `StatsError::BadParams` if the scale matrix is not symmetric-positive
    /// definite and `StatsError::ArgMustBePositive` if freedom is non-positive.
    pub fn new(
        location: Vec<f64>,
        scale: Vec<f64>,
        freedom: f64,
    ) -> Result<Self, MultivariateStudentError> {
        let dim = location.len();
        Self::new_from_nalgebra(location.into(), DMatrix::from_vec(dim, dim, scale), freedom)
    }

    /// Returns the dimension of the distribution.
    pub fn dim(&self) -> usize {
        self.location.len()
    }
}

impl<D> MultivariateStudent<D>
where
    D: DimMin<D, Output = D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>
        + nalgebra::allocator::Allocator<D, D>
        + nalgebra::allocator::Allocator<D>,
{
    pub fn new_from_nalgebra(
        location: OVector<f64, D>,
        scale: OMatrix<f64, D, D>,
        freedom: f64,
    ) -> Result<Self, MultivariateStudentError> {
        let dim = location.len();

        if location.iter().any(|f| f.is_nan()) {
            return Err(MultivariateStudentError::LocationInvalid);
        }

        if !scale.is_square()
            || scale.lower_triangle() != scale.upper_triangle().transpose()
            || scale.iter().any(|f| f.is_nan())
        {
            return Err(MultivariateStudentError::ScaleInvalid);
        }

        if freedom.is_nan() || freedom <= 0.0 {
            return Err(MultivariateStudentError::FreedomInvalid);
        }

        if location.nrows() != scale.nrows() {
            return Err(MultivariateStudentError::DimensionMismatch);
        }

        let scale_det = scale.determinant();
        let ln_pdf_const = gamma::ln_gamma(0.5 * (freedom + dim as f64))
            - gamma::ln_gamma(0.5 * freedom)
            - 0.5 * (dim as f64) * (freedom * PI).ln()
            - 0.5 * scale_det.ln();

        match Cholesky::new(scale.clone()) {
            None => Err(MultivariateStudentError::CholeskyFailed),
            Some(cholesky_decomp) => {
                let precision = cholesky_decomp.inverse();
                Ok(MultivariateStudent {
                    scale_chol_decomp: cholesky_decomp.unpack(),
                    location,
                    scale,
                    freedom,
                    precision,
                    ln_pdf_const,
                })
            }
        }
    }

    /// Returns the cholesky decomposiiton matrix of the scale matrix.
    ///
    /// Returns A where Σ = AAᵀ.
    pub fn scale_chol_decomp(&self) -> &OMatrix<f64, D, D> {
        &self.scale_chol_decomp
    }

    /// Returns the location of the distribution.
    pub fn location(&self) -> &OVector<f64, D> {
        &self.location
    }

    /// Returns the scale matrix of the distribution.
    pub fn scale(&self) -> &OMatrix<f64, D, D> {
        &self.scale
    }

    /// Returns the degrees of freedom of the distribution.
    pub fn freedom(&self) -> f64 {
        self.freedom
    }

    /// Returns the inverse of the cholesky decomposition matrix.
    pub fn precision(&self) -> &OMatrix<f64, D, D> {
        &self.precision
    }

    /// Returns the logarithmed constant part of the probability
    /// distribution function.
    pub fn ln_pdf_const(&self) -> f64 {
        self.ln_pdf_const
    }
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl<D> ::rand::distributions::Distribution<OVector<f64, D>> for MultivariateStudent<D>
where
    D: Dim,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D> + nalgebra::allocator::Allocator<D, D>,
{
    /// Samples from the multivariate student distribution
    ///
    /// # Formula
    ///
    /// ```math
    /// W ⋅ L ⋅ Z + μ
    /// ```
    ///
    /// where `W` has √(ν/Sν) distribution, Sν has Chi-squared
    /// distribution with ν degrees of freedom,
    /// `L` is the Cholesky decomposition of the scale matrix,
    /// `Z` is a vector of normally distributed random variables, and
    /// `μ` is the location vector
    fn sample<R: ::rand::Rng + ?Sized>(&self, rng: &mut R) -> OVector<f64, D> {
        use crate::distribution::{ChiSquared, Normal};

        let d = Normal::new(0., 1.).unwrap();
        let s = ChiSquared::new(self.freedom).unwrap();
        let w = (self.freedom / s.sample(rng)).sqrt();
        let (r, c) = self.location.shape_generic();
        let z = OVector::<f64, D>::from_distribution_generic(r, c, &d, rng);
        (w * &self.scale_chol_decomp * z) + &self.location
    }
}

impl<D> Min<OVector<f64, D>> for MultivariateStudent<D>
where
    D: Dim,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D> + nalgebra::allocator::Allocator<D, D>,
{
    /// Returns the minimum value in the domain of the
    /// multivariate normal distribution represented by a real vector
    fn min(&self) -> OVector<f64, D> {
        OMatrix::repeat_generic(
            self.location.shape_generic().0,
            Const::<1>,
            f64::NEG_INFINITY,
        )
    }
}

impl<D> Max<OVector<f64, D>> for MultivariateStudent<D>
where
    D: Dim,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D> + nalgebra::allocator::Allocator<D, D>,
{
    /// Returns the minimum value in the domain of the
    /// multivariate normal distribution represented by a real vector
    fn max(&self) -> OVector<f64, D> {
        OMatrix::repeat_generic(self.location.shape_generic().0, Const::<1>, f64::INFINITY)
    }
}

impl<D> MeanN<OVector<f64, D>> for MultivariateStudent<D>
where
    D: Dim,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D> + nalgebra::allocator::Allocator<D, D>,
{
    /// Returns the mean of the student distribution.
    ///
    /// # Remarks
    ///
    /// This is the same mean used to construct the distribution if
    /// the degrees of freedom is larger than 1.
    fn mean(&self) -> Option<OVector<f64, D>> {
        if self.freedom > 1. {
            Some(self.location.clone())
        } else {
            None
        }
    }
}

impl<D> VarianceN<OMatrix<f64, D, D>> for MultivariateStudent<D>
where
    D: Dim,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D> + nalgebra::allocator::Allocator<D, D>,
{
    /// Returns the covariance matrix of the multivariate student distribution.
    ///
    /// # Formula
    ///
    /// ```math
    /// Σ ⋅ ν / (ν - 2)
    /// ```
    ///
    /// where `Σ` is the scale matrix and `ν` is the degrees of freedom.
    /// Only defined if freedom is larger than 2.
    fn variance(&self) -> Option<OMatrix<f64, D, D>> {
        if self.freedom > 2. {
            Some(self.scale.clone() * self.freedom / (self.freedom - 2.))
        } else {
            None
        }
    }
}

impl<D> Mode<OVector<f64, D>> for MultivariateStudent<D>
where
    D: Dim,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D> + nalgebra::allocator::Allocator<D, D>,
{
    /// Returns the mode of the multivariate student distribution.
    ///
    /// # Formula
    ///
    /// ```math
    /// μ
    /// ```
    ///
    /// where `μ` is the location.
    fn mode(&self) -> OVector<f64, D> {
        self.location.clone()
    }
}

impl<D> Continuous<&OVector<f64, D>, f64> for MultivariateStudent<D>
where
    D: Dim + DimMin<D, Output = D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>
        + nalgebra::allocator::Allocator<D, D>
        + nalgebra::allocator::Allocator<D>,
{
    /// Calculates the probability density function for the multivariate.
    /// student distribution at `x`.
    ///
    /// # Formula
    ///
    /// ```math
    /// [Γ(ν+p)/2] / [Γ(ν/2) ((ν * π)^p det(Σ))^(1 / 2)] * [1 + 1/ν (x - μ)ᵀ inv(Σ) (x - μ)]^(-(ν+p)/2)
    /// ```
    ///
    /// where
    /// - `ν` is the degrees of freedom,
    /// - `μ` is the mean,
    /// - `Γ` is the Gamma function,
    /// - `inv(Σ)` is the precision matrix,
    /// - `det(Σ)` is the determinant of the scale matrix, and
    /// - `k` is the dimension of the distribution.
    fn pdf(&self, x: &OVector<f64, D>) -> f64 {
        if self.freedom.is_infinite() {
            use super::multivariate_normal::density_normalization_and_exponential;
            let (pdf_const, exp_arg) = density_normalization_and_exponential(
                &self.location,
                &self.scale,
                &self.precision,
                x,
            )
            .unwrap();
            return pdf_const * exp_arg.exp();
        }

        let dv = x - &self.location;
        let exp_arg: f64 = (&self.precision * &dv).dot(&dv);
        let base_term = 1. + exp_arg / self.freedom;
        self.ln_pdf_const.exp() * base_term.powf(-(self.freedom + self.location.len() as f64) / 2.)
    }

    /// Calculates the log probability density function for the multivariate
    /// student distribution at `x`. Equivalent to pdf(x).ln().
    fn ln_pdf(&self, x: &OVector<f64, D>) -> f64 {
        if self.freedom.is_infinite() {
            use super::multivariate_normal::density_normalization_and_exponential;
            let (pdf_const, exp_arg) = density_normalization_and_exponential(
                &self.location,
                &self.scale,
                &self.precision,
                x,
            )
            .unwrap();
            return pdf_const.ln() + exp_arg;
        }

        let dv = x - &self.location;
        let exp_arg: f64 = (&self.precision * &dv).dot(&dv);
        let base_term = 1. + exp_arg / self.freedom;
        self.ln_pdf_const - (self.freedom + self.location.len() as f64) / 2. * base_term.ln()
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod tests  {
    use core::fmt::Debug;

    use approx::RelativeEq;
    use nalgebra::{DMatrix, DVector, Dyn, OMatrix, OVector, U1, U2};

    use crate::{
        distribution::{Continuous, MultivariateStudent, MultivariateNormal},
        statistics::{Max, MeanN, Min, Mode, VarianceN},
    };

    use super::MultivariateStudentError;

    fn try_create(location: Vec<f64>, scale: Vec<f64>, freedom: f64) -> MultivariateStudent<Dyn>
    {
        let mvs = MultivariateStudent::new(location, scale, freedom);
        assert!(mvs.is_ok());
        mvs.unwrap()
    }

    fn create_case(location: Vec<f64>, scale: Vec<f64>, freedom: f64)
    {
        let mvs = try_create(location.clone(), scale.clone(), freedom);
        assert_eq!(DMatrix::from_vec(location.len(), location.len(), scale), mvs.scale);
        assert_eq!(DVector::from_vec(location), mvs.location);
    }

    fn bad_create_case(location: Vec<f64>, scale: Vec<f64>, freedom: f64)
    {
        let mvs = MultivariateStudent::new(location, scale, freedom);
        assert!(mvs.is_err());
    }

    fn test_case<T, F>(location: Vec<f64>, scale: Vec<f64>, freedom: f64, expected: T, eval: F)
    where
        T: Debug + PartialEq,
        F: FnOnce(MultivariateStudent<Dyn>) -> T,
    {
        let mvs = try_create(location, scale, freedom);
        let x = eval(mvs);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(
        location: Vec<f64>,
        scale: Vec<f64>,
        freedom: f64,
        expected: f64,
        acc: f64,
        eval: F,
        ) where
        F: FnOnce(MultivariateStudent<Dyn>) -> f64,
    {
        let mvs = try_create(location, scale, freedom);
        let x = eval(mvs);
        assert_almost_eq!(expected, x, acc);
    }

    fn test_almost_multivariate_normal<F1, F2>(
        location: Vec<f64>,
        scale: Vec<f64>,
        freedom: f64,
        acc: f64,
        x: DVector<f64>,
        eval_mvs: F1,
        eval_mvn: F2,
        ) where
            F1: FnOnce(MultivariateStudent<Dyn>, DVector<f64>) -> f64,
            F2: FnOnce(MultivariateNormal<Dyn>, DVector<f64>) -> f64,
        {
        let mvs = try_create(location.clone(), scale.clone(), freedom);
        let mvn0 = MultivariateNormal::new(location, scale);
        assert!(mvn0.is_ok());
        let mvn = mvn0.unwrap();
        let mvs_x = eval_mvs(mvs, x.clone());
        let mvn_x = eval_mvn(mvn, x.clone());
        assert!(mvs_x.relative_eq(&mvn_x, acc, acc), "mvn: {mvn_x} =/=\nmvs: {mvs_x}");
        // assert_relative_eq!(mvs_x, mvn_x, acc);
    }


    macro_rules! dvec {
        ($($x:expr),*) => (DVector::from_vec(vec![$($x),*]));
    }

    macro_rules! mat2 {
        ($x11:expr, $x12:expr, $x21:expr, $x22:expr) => (DMatrix::from_vec(2,2,vec![$x11, $x12, $x21, $x22]));
    }

    // macro_rules! mat3 {
    //     ($x11:expr, $x12:expr, $x13:expr, $x21:expr, $x22:expr, $x23:expr, $x31:expr, $x32:expr, $x33:expr) => (DMatrix::from_vec(3,3,vec![$x11, $x12, $x13, $x21, $x22, $x23, $x31, $x32, $x33]));
    // }

    #[test]
    fn test_create() {
        create_case(vec![0., 0.], vec![1., 0., 0., 1.], 1.);
        create_case(vec![10.,  5.], vec![2., 1., 1., 2.], 3.);
        create_case(vec![4., 5., 6.], vec![2., 1., 0., 1., 2., 1., 0., 1., 2.], 14.);
        create_case(vec![0., f64::INFINITY], vec![1., 0., 0., 1.], f64::INFINITY);
        create_case(vec![0., 0.], vec![f64::INFINITY, 0., 0., f64::INFINITY], 0.1);
    }

    #[test]
    fn test_bad_create() {
        // scale not symmetric.
        bad_create_case(vec![0., 0.], vec![1., 1., 0., 1.], 1.);
        // scale not positive-definite.
        bad_create_case(vec![0., 0.], vec![1., 2., 2., 1.], 1.);
        // NaN in location.
        bad_create_case(vec![0., f64::NAN], vec![1., 0., 0., 1.], 1.);
        // NaN in scale Matrix.
        bad_create_case(vec![0., 0.], vec![1., 0., 0., f64::NAN], 1.);
        // NaN in freedom.
        bad_create_case(vec![0., 0.], vec![1., 0., 0., 1.], f64::NAN);
        // Non-positive freedom.
        bad_create_case(vec![0., 0.], vec![1., 0., 0., 1.], 0.);
        bad_create_case(vec![0., 0.], vec![1., 0., 0., 1.], -1.);
    }

    #[test]
    fn test_variance() {
        let variance = |x: MultivariateStudent<Dyn>| x.variance().unwrap();
        test_case(vec![0., 0.], vec![1., 0., 0., 1.], 3., 3. * mat2![1., 0., 0., 1.], variance);
        test_case(vec![0., 0.], vec![f64::INFINITY, 0., 0., f64::INFINITY], 3., mat2![f64::INFINITY, 0., 0., f64::INFINITY], variance);
    }

    // Variance is only defined for freedom > 2.
    #[test]
    fn test_bad_variance() {
        let variance = |x: MultivariateStudent<Dyn>| x.variance();
        test_case(vec![0., 0.], vec![1., 0., 0., 1.], 2., None, variance);
    }

    #[test]
    fn test_mode() {
        let mode = |x: MultivariateStudent<Dyn>| x.mode();
        test_case(vec![0., 0.], vec![1., 0., 0., 1.], 1., dvec![0.,  0.], mode);
        test_case(vec![f64::INFINITY, f64::INFINITY], vec![1., 0., 0., 1.], 1., dvec![f64::INFINITY,  f64::INFINITY], mode);
    }

    #[test]
    fn test_mean() {
        let mean = |x: MultivariateStudent<Dyn>| x.mean().unwrap();
        test_case(vec![0., 0.], vec![1., 0., 0., 1.], 2., dvec![0., 0.], mean);
        test_case(vec![-1., 1., 3.], vec![1., 0., 0.5, 0., 2.0, 0., 0.5, 0., 3.0], 2., dvec![-1., 1., 3.], mean);
    }

    // Mean is only defined if freedom > 1.
    #[test]
    fn test_bad_mean() {
        let mean = |x: MultivariateStudent<Dyn>| x.mean();
        test_case(vec![0., 0.], vec![1., 0., 0., 1.], 1., None, mean);
    }

    #[test]
    fn test_min_max() {
        let min = |x: MultivariateStudent<Dyn>| x.min();
        let max = |x: MultivariateStudent<Dyn>| x.max();
        test_case(vec![0., 0.], vec![1., 0., 0., 1.], 1., dvec![f64::NEG_INFINITY, f64::NEG_INFINITY], min);
        test_case(vec![0., 0.], vec![1., 0., 0., 1.], 1., dvec![f64::INFINITY, f64::INFINITY], max);
        test_case(vec![10., 1.], vec![1., 0., 0., 1.], 1., dvec![f64::NEG_INFINITY, f64::NEG_INFINITY], min);
        test_case(vec![-3., 5.], vec![1., 0., 0., 1.], 1., dvec![f64::INFINITY, f64::INFINITY], max);
    }

    #[test]
    fn test_pdf() {
        let pdf = |arg: DVector<f64>| move |x: MultivariateStudent<Dyn>| x.pdf(&arg);
        test_almost(vec![0., 0.], vec![1., 0., 0., 1.], 4., 0.047157020175376416, 1e-15, pdf(dvec![1., 1.]));
        test_almost(vec![0., 0.], vec![1., 0., 0., 1.], 4., 0.013972450422333741737457302178882, 1e-15, pdf(dvec![1., 2.]));
        test_almost(vec![0., 0.], vec![1., 0., 0., 1.], 2., 0.012992240252399619, 1e-17, pdf(dvec![1., 2.]));
        test_almost(vec![2., 1.], vec![5., 0., 0., 1.], 2.5, 2.639780816598878e-5, 1e-19, pdf(dvec![1., 10.]));
        test_almost(vec![-1., 0.], vec![2., 1., 1., 6.], 1.5, 6.438051574348526e-5, 1e-19, pdf(dvec![10., 10.]));
        // These three are crossed checked against both python's scipy.multivariate_t.pdf and octave's mvtpdf.
        test_almost(vec![-1., 1., 50.], vec![1., 0.5, 0.25, 0.5, 1., -0.1, 0.25, -0.1, 1.], 8., 6.960998836915657e-16, 1e-30, pdf(dvec![0.9718, 0.1298, 0.8134]));
        test_almost(vec![-1., 1., 50.], vec![1., 0.5, 0.25, 0.5, 1., -0.1, 0.25, -0.1, 1.], 8., 7.369987979187023e-16, 1e-30, pdf(dvec![0.4922, 0.5522, 0.7185]));
        test_almost(vec![-1., 1., 50.], vec![1., 0.5, 0.25, 0.5, 1., -0.1, 0.25, -0.1, 1.], 8.,6.951631724511314e-16, 1e-30, pdf(dvec![0.3020, 0.1491, 0.5008]));
        test_case(vec![-1., 0.], vec![f64::INFINITY, 0., 0., f64::INFINITY], 10., 0., pdf(dvec![10., 10.]));
    }

    #[test]
    fn test_ln_pdf() {
        let ln_pdf = |arg: DVector<f64>| move |x: MultivariateStudent<Dyn>| x.ln_pdf(&arg);
        test_almost(vec![0., 0.], vec![1., 0., 0., 1.], 4., -3.0542723907338383, 1e-14, ln_pdf(dvec![1., 1.]));
        test_almost(vec![0., 0.], vec![1., 0., 0., 1.], 2., -4.3434030034000815, 1e-14, ln_pdf(dvec![1., 2.]));
        test_almost(vec![2., 1.], vec![5., 0., 0., 1.], 2.5, -10.542229575274265, 1e-14, ln_pdf(dvec![1., 10.]));
        test_almost(vec![-1., 0.], vec![2., 1., 1., 6.], 1.5, -9.650699521198622, 1e-14, ln_pdf(dvec![10., 10.]));
        // test_case(vec![-1., 0.], vec![f64::INFINITY, 0., 0., f64::INFINITY], 10., f64::NEG_INFINITY, ln_pdf(dvec![10., 10.]));
    }

    #[test]
    fn test_pdf_freedom_large() {
        let pdf_mvs = |mv: MultivariateStudent<Dyn>, arg: DVector<f64>| mv.pdf(&arg);
        let pdf_mvn = |mv: MultivariateNormal<Dyn>, arg: DVector<f64>| mv.pdf(&arg);
        test_almost_multivariate_normal(vec![0., 0.,], vec![1., 0., 0., 1.], 1e5, 1e-6, dvec![1., 1.], pdf_mvs, pdf_mvn);
        test_almost_multivariate_normal(vec![0., 0.,], vec![1., 0., 0., 1.], 1e10, 1e-7, dvec![1., 1.], pdf_mvs, pdf_mvn);
        test_almost_multivariate_normal(vec![0., 0.,], vec![1., 0., 0., 1.], f64::INFINITY, 1e-300, dvec![1., 1.], pdf_mvs, pdf_mvn);
        test_almost_multivariate_normal(vec![5., -1.,], vec![1., 0.99, 0.99, 1.], f64::INFINITY, 1e-300, dvec![5., 1.], pdf_mvs, pdf_mvn);
    }
    #[test]
    fn test_ln_pdf_freedom_large() {
        let pdf_mvs = |mv: MultivariateStudent<Dyn>, arg: DVector<f64>| mv.ln_pdf(&arg);
        let pdf_mvn = |mv: MultivariateNormal<Dyn>, arg: DVector<f64>| mv.ln_pdf(&arg);
        test_almost_multivariate_normal(vec![0., 0.,], vec![1., 0., 0., 1.], 1e5, 1e-5, dvec![1., 1.], pdf_mvs, pdf_mvn);
        test_almost_multivariate_normal(vec![0., 0.,], vec![1., 0., 0., 1.], 1e10, 5e-6, dvec![1., 1.], pdf_mvs, pdf_mvn);
        test_almost_multivariate_normal(vec![0., 0.,], vec![1., 0., 0., 1.], f64::INFINITY, 1e-300, dvec![1., 1.], pdf_mvs, pdf_mvn);
        test_almost_multivariate_normal(vec![0., 0.,], vec![1., 0.99, 0.99, 1.], f64::INFINITY, 1e-300, dvec![1., 1.], pdf_mvs, pdf_mvn);
    }

    #[test]
    fn test_immut_field_access() {
        // init as Dyn
        let mvs = MultivariateStudent::new(vec![1., 1.], vec![1., 0., 0., 1.], 2.)
            .expect("hard coded valid construction");
        assert_eq!(mvs.freedom(), 2.);
        assert_relative_eq!(mvs.ln_pdf_const(), std::f64::consts::TAU.recip().ln(), epsilon = 1e-15);

        // compare to static
        assert_eq!(mvs.dim(), 2); 
        assert!(mvs.location().eq(&OVector::<f64, U2>::new(1., 1.)));
        assert!(mvs.scale().eq(&OMatrix::<f64, U2, U2>::identity()));
        assert!(mvs.precision().eq(&OMatrix::<f64, U2, U2>::identity()));
        assert!(mvs.scale_chol_decomp().eq(&OMatrix::<f64, U2, U2>::identity()));

        // compare to Dyn
        assert_eq!(mvs.location(),&OVector::<f64, Dyn>::from_element_generic(Dyn(2), U1, 1.));
        assert_eq!(mvs.scale(), &OMatrix::<f64, Dyn, Dyn>::identity(2, 2));
        assert_eq!(mvs.precision(), &OMatrix::<f64, Dyn, Dyn>::identity(2, 2));
        assert_eq!(mvs.scale_chol_decomp(), &OMatrix::<f64, Dyn, Dyn>::identity(2, 2));
    }
        
    #[test]
    fn test_error_is_sync_send() {
        fn assert_sync_send<T: Sync + Send>() {}
        assert_sync_send::<MultivariateStudentError>();
    }
}
