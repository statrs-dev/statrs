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

/// Trait to express covariance operations as if the implementing type were an operator.
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

    /// transforms vector of uncorrelated samples to have correlations from self, (Sigma)^1/2 * vec{v}
    #[doc(alias = "matvec")]
    fn colorize(&self, other: Self::V) -> Self::V;

    /// transforms vector sampled with correlations from self to uncorrelated, (Sigma)^1/2 \ vec{v}
    #[doc(alias = "solve")]
    fn whiten(&self, other: Self::V) -> Self::V;

    /// returns the determinant of the covariance matrix, det(Sigma)
    fn determinant(&self) -> T;
}

#[cfg(feature = "nalgebra")]
mod multivariate {
    use nalgebra::{Cholesky, Dim, OMatrix, OVector, Scalar};

    impl<T, D> super::CovarianceMatrix<T> for OVector<T, D>
    where
        T: Scalar + nalgebra::RealField,
        D: Dim,
        nalgebra::DefaultAllocator:
            nalgebra::allocator::Allocator<T, D> + nalgebra::allocator::Allocator<T, D, D>,
    {
        type M = OMatrix<T, D, D>;
        type V = Self;

        fn dense(&self) -> Self::M {
            OMatrix::from_diagonal(self)
        }

        fn colorize(&self, other: Self::V) -> Self::V {
            self.clone().map(|x| x.sqrt()).component_mul(&other)
        }

        fn whiten(&self, other: Self::V) -> Self::V {
            other.component_div(&self.clone().map(|x| x.sqrt()))
        }

        fn determinant(&self) -> T {
            self.product()
        }
    }

    impl<T, D> super::CovarianceMatrix<T> for Cholesky<T, D>
    where
        T: Scalar + nalgebra::RealField,
        D: Dim,
        nalgebra::DefaultAllocator:
            nalgebra::allocator::Allocator<T, D> + nalgebra::allocator::Allocator<T, D, D>,
    {
        type M = OMatrix<T, D, D>;
        type V = OVector<T, D>;

        fn dense(&self) -> Self::M {
            self.l() * self.l().transpose()
        }

        fn colorize(&self, other: Self::V) -> Self::V {
            self.l().transpose() * other
        }

        fn whiten(&self, other: Self::V) -> Self::V {
            self.l_dirty().solve_lower_triangular_unchecked(&other)
        }

        fn determinant(&self) -> T {
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
    fn colorize(&self, other: Self::V) -> Self::V {
        self.sqrt() * other
    }
    fn whiten(&self, other: Self::V) -> Self::V {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covariance_scalar() {
        let x = 4.0_f64;
        assert_eq!(x.colorize(1.0), 2.0);
        assert_eq!(x.whiten(2.0), 1.0);
        assert_eq!(x.determinant(), 4.0);
    }

    #[cfg(feature = "nalgebra")]
    #[test]
    fn covariance_vector() {
        use approx::assert_relative_eq;
        use nalgebra::{vector, OMatrix};
        let v = vector![1.0_f64, 4.0, 9.0];
        assert_relative_eq!(v.dense(), OMatrix::from_diagonal(&v));
        assert_relative_eq!(v.colorize(vector![1.0, 1.0, 1.0]), vector![1.0, 2.0, 3.0]);
        assert_relative_eq!(v.whiten(vector![1.0, 2.0, 3.0]), vector![1.0, 1.0, 1.0]);
    }

    #[cfg(feature = "nalgebra")]
    #[test]
    fn covariance_matrix() {
        use approx::assert_relative_eq;
        use nalgebra::{matrix, vector, Cholesky};
        let m = matrix![5.0_f64, 8.0; 8.0, 13.0];
        let c = Cholesky::new(m).unwrap();
        assert_relative_eq!(c.dense(), m);

        for v in [vector![1.0, 0.0], vector![0.0, 1.0], vector![1.0, 1.0]] {
            assert_relative_eq!(
                c.colorize(v).norm_squared(),
                (v.transpose() * m * v)[(0, 0)]
            );
        }

        for v in [vector![1.0, 0.0], vector![0.0, 1.0], vector![1.0, 1.0]] {
            assert_relative_eq!(
                c.whiten(v).norm_squared(),
                (v.transpose() * (m.solve_lower_triangular_unchecked(&v)))[(0, 0)]
            );
        }
    }
}
