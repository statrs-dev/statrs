use crate::distribution::Continuous;
use crate::function::gamma;
use crate::prec;
use crate::statistics::*;
use alloc::{vec, vec::Vec};
use nalgebra::{Const, Dim, Dyn, OMatrix, OVector};
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

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
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    alpha: OVector<f64, D>,
}

/// Represents the errors that can occur when creating a [`Dirichlet`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum DirichletError {
    /// Alpha contains less than two elements.
    AlphaTooShort,

    /// Alpha contains an element that is NaN, infinite, zero or less than zero.
    AlphaHasInvalidElements,
}

impl core::fmt::Display for DirichletError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match self {
            DirichletError::AlphaTooShort => write!(f, "Alpha contains less than two elements"),
            DirichletError::AlphaHasInvalidElements => write!(
                f,
                "Alpha contains an element that is NaN, infinite, zero or less than zero"
            ),
        }
    }
}

impl core::error::Error for DirichletError {}

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
    pub fn new(alpha: Vec<f64>) -> Result<Self, DirichletError> {
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
    pub fn new_with_param(alpha: f64, n: usize) -> Result<Self, DirichletError> {
        Self::new(vec![alpha; n])
    }
}

impl<D> Dirichlet<D>
where
    D: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    /// Constructs a new distribution with the given vector for `alpha`
    /// Does not clone the vector it takes ownership of
    ///
    /// # Error
    ///
    /// Returns an error if vector has length less than 2 or if any element
    /// of alpha is NOT finite positive
    pub fn new_from_nalgebra(alpha: OVector<f64, D>) -> Result<Self, DirichletError> {
        if alpha.len() < 2 {
            return Err(DirichletError::AlphaTooShort);
        }

        if alpha.iter().any(|&a_i| !a_i.is_finite() || a_i <= 0.0) {
            return Err(DirichletError::AlphaHasInvalidElements);
        }

        Ok(Self { alpha })
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
        let ln_b =
            self.alpha.iter().map(|&x| gamma::ln_gamma(x)).sum::<f64>() - gamma::ln_gamma(sum);
        let entr = ln_b + (sum - self.alpha.len() as f64) * gamma::digamma(sum)
            - self
                .alpha
                .iter()
                .map(|&x| (x - 1.0) * gamma::digamma(x))
                .sum::<f64>();
        Some(entr)
    }
}

impl<D> core::fmt::Display for Dirichlet<D>
where
    D: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Dir({}, {})", self.alpha.len(), self.alpha)
    }
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl<D> ::rand::distr::Distribution<OVector<f64, D>> for Dirichlet<D>
where
    D: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    fn sample<R: ::rand::Rng + ?Sized>(&self, rng: &mut R) -> OVector<f64, D> {
        let mut sum = 0.0;
        OVector::from_iterator_generic(
            self.alpha.shape_generic().0,
            nalgebra::Const::<1>,
            self.alpha.iter().map(|&a| {
                let sample = super::gamma::sample_unchecked(rng, a, 1.0);
                sum += sample;
                sample
            }),
        ) / sum
    }
}

impl<D> Min<OVector<f64, D>> for Dirichlet<D>
where
    D: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    /// Returns the componentwise infimum over the support of the Dirichlet
    /// distribution, the zero vector.
    ///
    /// # Remarks
    ///
    /// This matches [`Beta::min`](crate::distribution::Beta::min), of which the
    /// Dirichlet is the multivariate generalization: each coordinate is
    /// supported on `(0, 1)`, so zero is an infimum rather than an attained
    /// value. See [`Self::max`] for why the vector itself is not in the support.
    fn min(&self) -> OVector<f64, D> {
        OMatrix::repeat_generic(self.alpha.shape_generic().0, Const::<1>, 0.0)
    }
}

impl<D> Max<OVector<f64, D>> for Dirichlet<D>
where
    D: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    /// Returns the componentwise supremum over the support of the Dirichlet
    /// distribution, one in every coordinate.
    ///
    /// # Remarks
    ///
    /// As with [`Self::min`], these are bounds on each coordinate separately and
    /// are approached but not attained. The returned vector is not in the
    /// support for `k > 1`: a Dirichlet sample lies on the unit simplex and so
    /// sums to one, whereas this vector sums to `k`. It is the corner of the
    /// smallest axis-aligned box containing the simplex.
    fn max(&self) -> OVector<f64, D> {
        OMatrix::repeat_generic(self.alpha.shape_generic().0, Const::<1>, 1.0)
    }
}

impl<D> Mode<Option<OVector<f64, D>>> for Dirichlet<D>
where
    D: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    /// Returns the mode of the dirichlet distribution, or `None` if any
    /// `α_i <= 1`.
    ///
    /// # Formula
    ///
    /// ```text
    /// (α_i - 1) / (α_0 - K)
    /// ```
    ///
    /// for the `i`th element, where `α_0` is the sum of all concentration
    /// parameters and `K` is their count.
    ///
    /// # Remarks
    ///
    /// The restriction to `α_i > 1` mirrors
    /// [`Beta::mode`](crate::distribution::Beta::mode), of which this is the
    /// generalization: at `K == 2` the formula is `(α - 1) / (α + β - 2)`. Below
    /// that threshold the density is unbounded at the corresponding face of the
    /// simplex, so no interior maximizer exists.
    ///
    /// Unlike [`Self::min`] and [`Self::max`], the mode *is* a point of the
    /// support: its coordinates sum to `(α_0 - K) / (α_0 - K) = 1`. The
    /// denominator is positive whenever the guard passes, since
    /// `α_0 - K = Σ(α_i - 1)`.
    fn mode(&self) -> Option<OVector<f64, D>> {
        if self.alpha.iter().any(|&a| a <= 1.0) {
            return None;
        }
        let sum = self.alpha_sum() - self.alpha.len() as f64;
        Some(self.alpha.map(|a| (a - 1.0) / sum))
    }
}

impl<D> MeanN<OVector<f64, D>> for Dirichlet<D>
where
    D: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
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
        nalgebra::allocator::Allocator<D> + nalgebra::allocator::Allocator<D, D>,
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

impl<D> Continuous<&OVector<f64, D>, f64> for Dirichlet<D>
where
    D: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>
        + nalgebra::allocator::Allocator<D, D>
        + nalgebra::allocator::Allocator<nalgebra::Const<1>, D>,
{
    /// Calculates the probabiliy density function for the dirichlet
    /// distribution
    /// with given `x`'s corresponding to the concentration parameters for this
    /// distribution
    ///
    /// # Remarks
    ///
    /// Returns `0.0` for any `x` outside the support: an element not in
    /// `(0, 1)`, or elements that do not sum to `1` within a tolerance of
    /// `1e-4`. This matches every other distribution in the crate, including
    /// [`Multinomial`](crate::distribution::Multinomial), whose `pmf` likewise
    /// returns zero rather than failing when the coordinates do not sum
    /// correctly.
    ///
    /// # Panics
    ///
    /// If `x` is not the same length as the vector of concentration parameters
    /// for this distribution. Unlike an out-of-support value, that is a
    /// dimension error with no meaningful density to return.
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
    /// # Remarks
    ///
    /// Returns `f64::NEG_INFINITY` for any `x` outside the support, namely an
    /// element outside `(0, 1)`[^*] or elements that do not add to `1f64` within
    /// `1e-4`.
    ///
    /// [^*]: inspected by checking each element of `x` with `(f64::MIN_POSITIVE..1.0).contains(&x_i)`, so a subnormal `x_i` is treated as off the simplex
    ///
    /// # Panics
    ///
    /// If `x` is not the same length as the concentration parameters, `alpha`.
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
        if self.alpha.len() != x.len() {
            panic!("Arguments must have correct dimensions.");
        }

        // Off the simplex the density is zero. Classify before evaluating, so
        // that an out-of-range x_i cannot reach `ln` and produce a misleading
        // finite total.
        if x.iter().any(|x_i| !(f64::MIN_POSITIVE..1.0).contains(x_i))
            || !prec::abs_diff_eq!(x.sum(), 1.0, epsilon = 1e-4)
        {
            return f64::NEG_INFINITY;
        }

        let mut term = 0.0;
        let mut sum_alpha = 0.0;

        for (&x_i, &alpha_i) in x.iter().zip(self.alpha.iter()) {
            term += (alpha_i - 1.0) * x_i.ln() - gamma::ln_gamma(alpha_i);
            sum_alpha += alpha_i;
        }

        term + gamma::ln_gamma(sum_alpha)
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::prec;

    use core::fmt::{Debug, Display};

    use nalgebra::{dmatrix, dvector, vector, DimMin, OVector};

    fn try_create<D>(alpha: OVector<f64, D>) -> Dirichlet<D>
    where
        D: DimMin<D, Output = D>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
    {
        let mvn = Dirichlet::new_from_nalgebra(alpha);
        assert!(mvn.is_ok());
        mvn.unwrap()
    }

    fn bad_create_case<D>(alpha: OVector<f64, D>)
    where
        D: DimMin<D, Output = D>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
    {
        let dd = Dirichlet::new_from_nalgebra(alpha);
        assert!(dd.is_err());
    }

    fn test_almost<F, T, D>(alpha: OVector<f64, D>, expected: T, acc: f64, eval: F)
    where
        T: Debug + Display + approx::RelativeEq<Epsilon = f64>,
        F: FnOnce(Dirichlet<D>) -> T,
        D: DimMin<D, Output = D>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
    {
        let dd = try_create(alpha);
        let x = eval(dd);
        prec::assert_relative_eq!(expected, x, epsilon = acc);
    }

    #[test]
    fn test_create() {
        try_create(vector![1.0, 2.0]);
        try_create(vector![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(Dirichlet::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]).is_ok());
        // try_create(vector![0.001, f64::INFINITY, 3756.0]); // moved to bad case as this is degenerate
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(vector![1.0, f64::NAN]);
        bad_create_case(vector![1.0, 0.0]);
        bad_create_case(vector![1.0, f64::INFINITY]);
        bad_create_case(vector![-1.0, 2.0]);
        bad_create_case(vector![1.0]);
        bad_create_case(vector![1.0, 2.0, 0.0, 4.0, 5.0]);
        bad_create_case(vector![1.0, f64::NAN, 3.0, 4.0, 5.0]);
        bad_create_case(vector![0.0, 0.0, 0.0]);
        bad_create_case(vector![0.001, f64::INFINITY, 3756.0]); // moved to bad case as this is degenerate
    }

    #[cfg(feature = "rand")]
    #[test]
    fn test_sample() {
        use rand::distr::Distribution;
        use rand::SeedableRng;

        test_almost(vector![1., 2.], 1., 1e-15, |dd| {
            dd.sample(&mut rand::rngs::StdRng::seed_from_u64(0)).sum()
        });
    }

    #[test]
    fn test_mode() {
        // (α_i - 1) / (α_0 - K): here α_0 = 6, K = 3, so each is 1/3.
        let d = try_create(dvector![2.0, 2.0, 2.0]);
        prec::assert_relative_eq!(d.mode().unwrap(), dvector![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], epsilon = 1e-15);

        // α_0 = 9, K = 3, denominator 6.
        let d = try_create(dvector![2.0, 3.0, 4.0]);
        prec::assert_relative_eq!(
            d.mode().unwrap(),
            dvector![1.0 / 6.0, 1.0 / 3.0, 0.5],
            epsilon = 1e-15
        );

        // Unbounded density at a face -> no interior mode, as for Beta.
        assert!(try_create(dvector![1.0, 2.0, 3.0]).mode().is_none());
        assert!(try_create(dvector![0.5, 2.0]).mode().is_none());
        assert!(try_create(dvector![2.0, 1.0]).mode().is_none());
    }

    /// The mode is a genuine support point, unlike the componentwise bounds, and
    /// it agrees with `Beta` at `K == 2`.
    #[test]
    fn test_mode_is_in_support_and_generalizes_beta() {
        let d = try_create(dvector![2.0, 3.0, 4.0]);
        let m = d.mode().unwrap();
        prec::assert_relative_eq!(m.sum(), 1.0, epsilon = 1e-15);
        assert!(d.pdf(&m) > 0.0);

        let beta = crate::distribution::Beta::new(3.0, 5.0).unwrap();
        let d2 = try_create(vector![3.0, 5.0]);
        prec::assert_relative_eq!(d2.mode().unwrap()[0], beta.mode().unwrap(), epsilon = 1e-15);
    }

    /// Reference-free check of the formula: the returned point must actually
    /// maximize the density. Perturbing mass from one coordinate to another
    /// keeps the point on the simplex, so the density must not increase.
    #[test]
    fn test_mode_maximizes_the_density() {
        for alpha in [
            dvector![2.0, 3.0, 4.0],
            dvector![5.0, 5.0, 5.0],
            dvector![1.5, 9.0, 2.5, 4.0],
        ] {
            let d = try_create(alpha);
            let m = d.mode().unwrap();
            let at_mode = d.pdf(&m);

            for i in 0..m.len() {
                for j in 0..m.len() {
                    if i == j {
                        continue;
                    }
                    for eps in [1e-4, 1e-3, 1e-2, 0.05] {
                        if m[j] <= eps {
                            continue;
                        }
                        let mut q = m.clone();
                        q[i] += eps;
                        q[j] -= eps;
                        assert!(
                            d.pdf(&q) <= at_mode,
                            "pdf at perturbed point exceeded the claimed mode"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_min_max() {
        let d = try_create(dvector![1.0, 2.0, 3.0]);
        assert_eq!(d.min(), dvector![0.0, 0.0, 0.0]);
        assert_eq!(d.max(), dvector![1.0, 1.0, 1.0]);

        let d = try_create(vector![0.5, 0.5]);
        assert_eq!(d.min(), vector![0.0, 0.0]);
        assert_eq!(d.max(), vector![1.0, 1.0]);
    }

    /// As with `Multinomial` (statrs-dev/statrs#276), the bounds are per
    /// coordinate: they contain the simplex without lying on it, and they are
    /// open rather than attained. That is the `Beta` behaviour generalized.
    #[test]
    fn test_min_max_bound_the_support_componentwise() {
        let d = try_create(dvector![2.0, 2.0, 2.0]);

        // Every coordinate of a support point lies strictly between the bounds.
        let interior = dvector![0.25, 0.25, 0.5];
        assert!(d.pdf(&interior) > 0.0, "premise: interior point is in support");
        for i in 0..3 {
            assert!(d.min()[i] < interior[i] && interior[i] < d.max()[i]);
        }

        // But the bound vectors are not support points: a Dirichlet sample sums
        // to one, whereas these sum to 0 and to k. The density is zero at both.
        prec::assert_relative_eq!(interior.sum(), 1.0, epsilon = 1e-15);
        assert_eq!(d.min().sum(), 0.0);
        assert_eq!(d.max().sum(), 3.0);
        assert_eq!(d.pdf(&d.min()), 0.0, "the zero vector is off the simplex");
        assert_eq!(d.pdf(&d.max()), 0.0, "the ones vector sums to k, not 1");

        // Matches the univariate case it generalizes.
        let beta = crate::distribution::Beta::new(2.0, 2.0).unwrap();
        assert_eq!(beta.min(), d.min()[0]);
        assert_eq!(beta.max(), d.max()[0]);
    }

    #[test]
    fn test_mean() {
        let mean = |dd: Dirichlet<_>| dd.mean().unwrap();

        test_almost(vec![0.5; 5].into(), vec![1.0 / 5.0; 5].into(), 1e-15, mean);

        test_almost(
            dvector![0.1, 0.2, 0.3, 0.4],
            dvector![0.1, 0.2, 0.3, 0.4],
            1e-15,
            mean,
        );

        test_almost(
            dvector![1.0, 2.0, 3.0, 4.0],
            dvector![0.1, 0.2, 0.3, 0.4],
            1e-15,
            mean,
        );
    }

    #[test]
    fn test_variance() {
        let variance = |dd: Dirichlet<_>| dd.variance().unwrap();

        test_almost(
            dvector![1.0, 2.0],
            dmatrix![0.055555555555555, -0.055555555555555;
                    -0.055555555555555,  0.055555555555555;
            ],
            1e-15,
            variance,
        );

        test_almost(
            dvector![0.1, 0.2, 0.3, 0.4],
            dmatrix![0.045, -0.010, -0.015, -0.020;
                    -0.010,  0.080, -0.030, -0.040;
                    -0.015, -0.030,  0.105, -0.060;
                    -0.020, -0.040, -0.060,  0.120;
            ],
            1e-15,
            variance,
        );
    }

    // #[test]
    // fn test_std_dev() {
    //     let alpha = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    //     let sum = alpha.iter().fold(0.0, |acc, x| acc + x);
    //     let n = Dirichlet::new(&alpha).unwrap();
    //     let res = n.std_dev();
    //     for i in 1..11 {
    //         let f = i as f64;
    //         prec::assert_abs_diff_eq!(res[i-1], (f * (sum - f) / (sum * sum * (sum + 1.0))).sqrt(), epsilon = 1e-15);
    //     }
    // }

    #[test]
    fn test_entropy() {
        let entropy = |x: Dirichlet<_>| x.entropy().unwrap();
        // cross-checked against scipy.stats.dirichlet(alpha).entropy()
        test_almost(
            vector![0.1, 0.3, 0.5, 0.8],
            -9.318820275187153,
            1e-12,
            entropy,
        );
        test_almost(
            vector![0.1, 0.2, 0.3, 0.4],
            -10.200309764545434,
            1e-12,
            entropy,
        );
        test_almost(
            vector![1.5, 2.0, 3.5, 4.0],
            -2.7374675446648483,
            1e-12,
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

    /// Off the simplex the density is zero rather than a panic. These four
    /// cases previously asserted `#[should_panic]`; see the note in the PR
    /// description accompanying statrs-dev/statrs#276.
    #[test]
    fn test_pdf_out_of_support_is_zero() {
        let n = try_create(vector![0.1, 0.3, 0.5, 0.8]);
        // an element outside (0, 1)
        assert_eq!(n.pdf(&vector![1.5, 0.0, 0.0, 0.0]), 0.0);
        assert_eq!(n.pdf(&vector![-0.5, 0.5, 0.5, 0.5]), 0.0);
        // elements that do not sum to 1
        assert_eq!(n.pdf(&vector![0.5, 0.25, 0.8, 0.9]), 0.0);
        assert_eq!(n.pdf(&vector![0.1, 0.1, 0.1, 0.1]), 0.0);
        // the in-support case still evaluates
        assert!(n.pdf(&vector![0.25, 0.25, 0.25, 0.25]) > 0.0);
    }

    #[test]
    fn test_ln_pdf_out_of_support_is_neg_infinity() {
        let n = try_create(vector![0.1, 0.3, 0.5, 0.8]);
        assert_eq!(n.ln_pdf(&vector![1.5, 0.0, 0.0, 0.0]), f64::NEG_INFINITY);
        assert_eq!(n.ln_pdf(&vector![0.5, 0.25, 0.8, 0.9]), f64::NEG_INFINITY);
        // consistent with pdf, which is its exponential
        assert_eq!(n.pdf(&vector![0.5, 0.25, 0.8, 0.9]), 0.0);
        assert!(n.ln_pdf(&vector![0.25, 0.25, 0.25, 0.25]).is_finite());
    }

    #[test]
    #[should_panic]
    fn test_ln_pdf_bad_input_length() {
        let n = try_create(dvector![0.1, 0.3, 0.5, 0.8]);
        n.ln_pdf(&dvector![0.5]);
    }

    #[test]
    fn test_error_is_sync_send() {
        fn assert_sync_send<T: Sync + Send>() {}
        assert_sync_send::<DirichletError>();
    }
}
