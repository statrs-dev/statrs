use num_traits::float::Float;

/// The `Min` trait specifies than an object has a minimum value
pub trait Min<T> {
    /// Returns the minimum value in the domain of a given distribution
    /// if it exists, otherwise `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Min;
    /// use statrs::distribution::{Uniform, UniformError};
    ///
    /// let n = Uniform::new(0.0, 1.0)?;
    /// assert_eq!(0.0, n.min());
    /// # Ok::<(), UniformError>(())
    /// ```
    fn min(&self) -> T;
}

/// The `Max` trait specifies that an object has a maximum value
pub trait Max<T> {
    /// Returns the maximum value in the domain of a given distribution
    /// if it exists, otherwise `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Max;
    /// use statrs::distribution::{Uniform, UniformError};
    ///
    /// let n = Uniform::new(0.0, 1.0)?;
    /// assert_eq!(1.0, n.max());
    /// # Ok::<(), UniformError>(())
    /// ```
    fn max(&self) -> T;
}

/// Exposes an entropy method, uses base e, (not Shannon, which base 2).
pub trait Entropy<T> {
    fn entropy(&self) -> T;
}

/// Trait to express covariance operations as if the implementing type were a matrix,
///
/// For scalars this is variance and scalar
///
/// ```text
/// Sigma_ij = Cov[X_i, X_j]
/// ```
pub trait CovarianceMatrix<T> {
    /// Type that acts as matrix-like operator
    type M;
    /// Type that acts as vector-like state
    type V;

    /// returns a covariance matrix, M_ij = Sigma_ij
    fn dense(&self) -> Self::M;

    /// returns a vector scaled by covariance, (Sigma)^1/2 * vec{v}
    fn forward(&self, other: Self::V) -> Self::V;

    /// returns a vector unscaled by covariance, (Sigma)^-1/2 * vec{v}
    fn inverse(&self, other: Self::V) -> Self::V;

    /// returns the determinant of the covariance matrix, det(Sigma)
    fn determinant(&self) -> T;
}

#[cfg(feature = "nalgebra")]
mod multivariate {
    use nalgebra::{Cholesky, Dim, OMatrix, OVector};

    impl<D> super::CovarianceMatrix<f64> for OVector<f64, D>
    where
        D: Dim,
        nalgebra::DefaultAllocator:
            nalgebra::allocator::Allocator<f64, D> + nalgebra::allocator::Allocator<f64, D, D>,
    {
        type M = OMatrix<f64, D, D>;
        type V = Self;

        fn dense(&self) -> Self::M {
            OMatrix::from_diagonal(self)
        }

        fn forward(&self, other: Self::V) -> Self::V {
            self.clone().map(|x| x.sqrt()).component_mul(&other)
        }

        fn inverse(&self, other: Self::V) -> Self::V {
            other.component_div(&self.clone().map(|x| x.sqrt()))
        }

        fn determinant(&self) -> f64 {
            self.product()
        }
    }

    impl<D> super::CovarianceMatrix<f64> for Cholesky<f64, D>
    where
        D: Dim,
        nalgebra::DefaultAllocator:
            nalgebra::allocator::Allocator<f64, D> + nalgebra::allocator::Allocator<f64, D, D>,
    {
        type M = OMatrix<f64, D, D>;
        type V = OVector<f64, D>;

        fn dense(&self) -> Self::M {
            self.l() * self.l().transpose()
        }

        fn forward(&self, other: Self::V) -> Self::V {
            self.l() * other
        }

        fn inverse(&self, other: Self::V) -> Self::V {
            self.l_dirty().solve_lower_triangular_unchecked(&other)
        }

        fn determinant(&self) -> f64 {
            self.determinant()
        }
    }
}

impl<T: Float> CovarianceMatrix<T> for T {
    type M = T;
    type V = T;
    fn dense(&self) -> Self::M {
        *self
    }
    fn forward(&self, other: Self::V) -> Self::V {
        self.sqrt() * other
    }
    fn inverse(&self, other: Self::V) -> Self::V {
        other / self.sqrt()
    }
    fn determinant(&self) -> T {
        *self
    }
}

/// Trait to express the mean, i.e. the lowest non-trivial [standardized moment](https://en.wikipedia.org/wiki/Moment_(mathematics)#Standardized_moments)
/// of a distribution's realization implemented on its distribution's type
///
/// # Note for implementors
/// Associated types should capture semantics of the distribution, e.g.
/// [`Option`] should be used where moments is defined or undefined based on parameter values.
/// [`()`] is used for moments that are never defined
pub trait Mean {
    type Mu;
    fn mean(&self) -> Self::Mu;
}

/// Trait to express the variance, the second [standardized moment](https://en.wikipedia.org/wiki/Moment_(mathematics)#Standardized_moments)
/// of a distribution's realization implemented on its distribution's type.
///
/// Multivariates generalize this to covariance.
/// # Note for implementors
/// Associated types should capture semantics of the distribution, e.g.
/// [`Option`] should be used where moments is defined or undefined based on parameter values.
/// [`()`] is used for moments that are never defined
pub trait Variance: Mean {
    type Var;
    fn variance(&self) -> Self::Var;
}

/// Trait to express the skewness, the third [standardized moment](https://en.wikipedia.org/wiki/Moment_(mathematics)#Standardized_moments)
/// of a distribution's realization implemented on its distribution's type.
///
/// This is the third central moment normalized by the variance to 3/2 power.
/// # Note for implementors
/// Associated types should capture semantics of the distribution, e.g.
/// [`Option`] should be used where moments is defined or undefined based on parameter values.
/// [`()`] is used for moments that are never defined
pub trait Skewness: Variance {
    type Skew;
    fn skewness(&self) -> Self::Skew;
}

/// Trait to express the excess kurtosis, the fourth [standardized moment](https://en.wikipedia.org/wiki/Moment_(mathematics)#Standardized_moments)
/// of a distribution's realization implemented on its distribution's type.
///
/// This is the third central moment normalized by the variance's second power.
/// # Note for implementors
/// Associated types should capture semantics of the distribution, e.g.
/// [`Option`] should be used where moments is defined or undefined based on parameter values.
/// [`()`] is used for moments that are never defined
pub trait ExcessKurtosis: Variance {
    type Kurt;
    fn excess_kurtosis(&self) -> Self::Kurt;
}

/// The `Median` trait returns the median of the distribution.
pub trait Median<T> {
    /// Returns the median.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Median;
    /// use statrs::distribution::{Uniform, UniformError};
    ///
    /// let n = Uniform::new(0.0, 1.0)?;
    /// assert_eq!(0.5, n.median());
    /// # Ok::<(), UniformError>(())
    /// ```
    fn median(&self) -> T;
}

/// The `Mode` trait specifies that an object has a closed form solution
/// for its mode(s)
pub trait Mode<T> {
    /// Returns the mode, if one exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Mode;
    /// use statrs::distribution::{Uniform, UniformError};
    ///
    /// let n = Uniform::new(0.0, 1.0)?;
    /// assert_eq!(Some(0.5), n.mode());
    /// # Ok::<(), UniformError>(())
    /// ```
    fn mode(&self) -> T;
}
