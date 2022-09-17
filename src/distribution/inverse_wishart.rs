use std::f64::consts::LN_2;
use nalgebra::{Cholesky, DMatrix, Dynamic};
use rand::Rng;
use crate::{Result, StatsError};
use crate::distribution::Continuous;
use crate::distribution::wishart::Wishart;
use crate::function::gamma::{mvgamma, mvlgamma};
use crate::statistics::{MeanN, Mode, VarianceN};

/// Implements the [Inverse Wishart distribution](http://en.wikipedia.org/wiki/Inverse-Wishart_distribution)
///
/// # Example
/// ```
/// use nalgebra::DMatrix;
/// use statrs::distribution::{InverseWishart, Continuous};
/// use statrs::statistics::Distribution;
/// use statrs::prec;
///
/// let result = InverseWishart::new(2.0, DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]));
/// assert!(result.is_ok());
///
/// let result = InverseWishart::new(1.0, DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]));
/// assert!(result.is_err());
/// ```
#[derive(Debug, Clone)]
pub struct InverseWishart {
    freedom: f64,
    scale: DMatrix<f64>,
    chol: Cholesky<f64, Dynamic>,
}

impl InverseWishart {
    /// Constructs a new Inverse Wishart distribution with a degrees of freedom (ν) of `freedom`
    /// and a scale matrix (ψ) of `scale`
    ///
    /// # Errors
    ///
    /// Returns an error if `scale` matrix is not square.
    /// Returns an error if `freedom` is `NaN`.
    /// Returns an error if `freedom <= rows(scale) - 1`
    /// Returns an error if `scale` is not positive definite.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::DMatrix;
    /// use statrs::distribution::InverseWishart;
    ///
    /// let result = InverseWishart::new(2.0, DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]));
    /// assert!(result.is_ok());
    ///
    /// let result = InverseWishart::new(1.0, DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]));
    /// assert!(result.is_err());
    /// ```
    pub fn new(freedom: f64, scale: DMatrix<f64>) -> Result<Self> {
        if scale.nrows() != scale.ncols() {
            return Err(StatsError::BadParams);
        }
        if freedom <= 0.0 || freedom.is_nan() {
            return Err(StatsError::ArgMustBePositive("degree of freedom must be positive"));
        }
        if freedom <= scale.nrows() as f64 - 1.0 {
            return Err(StatsError::ArgGt("degree of freedom must be greater than p-1", freedom));
        }

        match Cholesky::new(scale.clone()) {
            None => Err(StatsError::BadParams),
            Some(chol) => {
                Ok(InverseWishart { freedom, scale, chol })
            }
        }
    }

    /// Returns the degrees of freedom of
    /// the Inverse Wishart distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::DMatrix;
    /// use statrs::distribution::InverseWishart;
    ///
    /// let w = InverseWishart::new(2.0, DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0])).unwrap();
    /// assert_eq!(w.freedom(), 2.0);
    /// ```
    pub fn freedom(&self) -> f64 {
        self.freedom
    }

    /// Returns the scale of the Inverse Wishart distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::DMatrix;
    /// use statrs::distribution::InverseWishart;
    ///
    /// let w = InverseWishart::new(2.0, DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0])).unwrap();
    /// assert_eq!(w.scale(), &DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]));
    /// ```
    pub fn scale(&self) -> &DMatrix<f64> {
        &self.scale
    }

    /// Returns the dimensionality of the Inverse Wishart distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::DMatrix;
    /// use statrs::distribution::InverseWishart;
    ///
    /// let w = InverseWishart::new(3.0, DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0])).unwrap();
    /// assert_eq!(w.p(), 2);
    /// ```
    pub fn p(&self) -> usize {
        self.scale.nrows()
    }
}

impl MeanN<DMatrix<f64>> for InverseWishart {
    /// Returns the mean of the Inverse Wishart distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ψ/(ν - p - 1)
    /// ```
    ///
    /// where `ν` is the degree of freedom, `ψ` is the scale matrix, and `p` is the dimensionality of the distribution.
    fn mean(&self) -> Option<DMatrix<f64>> {
        if self.freedom > self.p() as f64 + 1.0 {
            Some(&self.scale / (self.freedom - self.p() as f64 - 1.0))
        } else {
            None
        }
    }
}

impl VarianceN<DMatrix<f64>> for InverseWishart {
    /// Returns the variance of the Inverse Wishart distribution
    /// See [formula](https://en.wikipedia.org/wiki/Inverse-Wishart_distribution#Moments)
    fn variance(&self) -> Option<DMatrix<f64>> {
        let p = self.p() as f64;

        Some(self.scale.map_with_location(|i, j, x| {
            let n1 = ((self.freedom - p + 1.0) * x.powi(2)) + ((self.freedom - p - 1.0) * self.scale[(i, i)] * self.scale[(j, j)]);
            let n2 = (self.freedom - p) * (self.freedom - p - 1.0) * (self.freedom - p - 1.0) * (self.freedom - p - 3.0);
            n1 / n2
        }))
    }
}

impl Mode<Option<DMatrix<f64>>> for InverseWishart {
    /// Returns the median of the Inverse Wishart distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ψ/(ν + p + 1)
    /// ```
    ///
    /// where `ν` is the degree of freedom, `ψ` is the scale matrix, and `p` is the dimensionality of the distribution.
    fn mode(&self) -> Option<DMatrix<f64>> {
        if self.freedom > self.p() as f64 + 1.0 {
            Some(&self.scale * (1.0 / (self.freedom + self.p() as f64 + 1.0)))
        } else {
            None
        }
    }
}

impl ::rand::distributions::Distribution<DMatrix<f64>> for InverseWishart {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> DMatrix<f64> {
        let w = Wishart::new(
            self.freedom,
            self.scale.clone().try_inverse().unwrap(), // We already know S is positive definite
        ).unwrap();

        w.sample(rng).pseudo_inverse(1e-4).unwrap().symmetric_part()
    }
}

impl Continuous<DMatrix<f64>, f64> for InverseWishart {
    /// Calculates the probability density function for the Inverse Wishart
    /// distribution at `x`
    fn pdf(&self, x: DMatrix<f64>) -> f64 {
        let p = self.p() as f64;
        let chol = Cholesky::new(x).expect("x is not positive definite");
        let x_det = chol.determinant();
        let sxi = chol.solve(&self.scale);

        x_det.powf(-(self.freedom + p + 1.0) / 2.0)
            * (-0.5 * sxi.trace()).exp()
            * self.chol.determinant().powf(self.freedom / 2.0)
            / (2.0f64).powf(self.freedom * p / 2.0)
            / mvgamma(p as i64, self.freedom / 2.0)
    }

    /// Calculates the log probability density function for the Inverse Wishart
    /// distribution at `x`
    fn ln_pdf(&self, x: DMatrix<f64>) -> f64 {
        let p = self.p() as f64;
        let chol = Cholesky::new(x).expect("x is not positive definite");
        let x_lndet = chol.determinant().ln();
        let sxi = chol.solve(&self.scale);

        x_lndet * -(self.freedom + p + 1.0) / 2.0
            - 0.5 * sxi.trace()
            + self.chol.determinant().ln() * (self.freedom / 2.0)
            - LN_2 * (self.freedom * p / 2.0)
            - mvlgamma(p as i64, self.freedom / 2.0)
    }
}


#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use nalgebra::DMatrix;
    use rand::distributions::Distribution;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use crate::distribution::Continuous;
    use crate::distribution::inverse_wishart::InverseWishart;
    use crate::statistics::{MeanN, Mode, VarianceN};

    fn try_create(freedom: f64, scale: DMatrix<f64>) -> InverseWishart
    {
        let w = InverseWishart::new(freedom, scale);
        assert!(w.is_ok());
        w.unwrap()
    }

    fn test_almost<F>(
        freedom: f64,
        scale: DMatrix<f64>,
        expected: f64,
        acc: f64,
        eval: F,
    ) where
        F: FnOnce(InverseWishart) -> f64,
    {
        let mvn = try_create(freedom, scale);
        let x = eval(mvn);
        assert_almost_eq!(expected, x, acc);
    }

    fn test_almost_mat<F>(
        freedom: f64,
        scale: DMatrix<f64>,
        expected: DMatrix<f64>,
        acc: f64,
        eval: F,
    ) where
        F: FnOnce(InverseWishart) -> DMatrix<f64>,
    {
        let mvn = try_create(freedom, scale);
        let x = eval(mvn);

        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert_almost_eq!(expected[(i, j)], x[(i, j)], acc);
            }
        }
    }

    #[test]
    fn test_mean() {
        test_almost_mat(
            6.0, DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]),
            DMatrix::from_row_slice(2, 2, &[
                0.333333, 0.0,
                0.0, 0.333333
            ]),
            1e-5, |w| w.mean().unwrap(),
        );
        test_almost_mat(
            7.0, DMatrix::from_row_slice(3, 3, &[
                1.0143, 0.0000, 0.0000,
                0.0000, 0.2034, 0.0000,
                0.0000, 0.0000, 0.9495
            ]),
            DMatrix::from_row_slice(3, 3, &[
                0.3381, 0.0, 0.0,
                0.0, 0.0678, 0.0,
                0.0, 0.0, 0.3165,
            ]),
            1e-5, |w| w.mean().unwrap(),
        );
    }

    #[test]
    fn test_mode() {
        test_almost_mat(
            6.0, DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]),
            DMatrix::from_row_slice(2, 2, &[
                0.111111, 0.0,
                0.0, 0.111111
            ]),
            1e-5, |w| w.mode().unwrap(),
        );
        test_almost_mat(
            7.0, DMatrix::from_row_slice(3, 3, &[
                1.0143, 0.0000, 0.0000,
                0.0000, 0.2034, 0.0000,
                0.0000, 0.0000, 0.9495
            ]),
            DMatrix::from_row_slice(3, 3, &[
                0.0922091, 0.0, 0.0,
                0.0, 0.0184909, 0.0,
                0.0, 0.0, 0.0863182,
            ]),
            1e-5, |w| w.mode().unwrap(),
        );
    }

    #[test]
    fn test_variance() {
        test_almost_mat(
            6.0, DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]),
            DMatrix::from_row_slice(2, 2, &[
                0.222222, 0.0833333,
                0.0833333, 0.222222
            ]),
            1e-5, |w| w.variance().unwrap(),
        );
        test_almost_mat(
            7.0, DMatrix::from_row_slice(3, 3, &[
                1.0143, 0.0000, 0.0000,
                0.0000, 0.2034, 0.0000,
                0.0000, 0.0000, 0.9495
            ]),
            DMatrix::from_row_slice(3, 3, &[
                0.228623, 0.0171924, 0.0802565,
                0.0171924, 0.00919368, 0.016094,
                0.0802565, 0.016094, 0.200344,
            ]),
            1e-5, |w| w.variance().unwrap(),
        );
    }

    #[test]
    fn test_pdf() {
        test_almost(
            2.0, DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]),
            0.02927491576215958,
            1e-15, |w| w.pdf(DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0])),
        );
        test_almost(
            2.0, DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]),
            -3.5310242469692907,
            1e-15, |w| w.ln_pdf(DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0])),
        );

        test_almost(
            6.0, DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]),
            0.0012197881567566496,
            1e-15, |w| w.pdf(DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0])),
        );
        test_almost(
            6.0, DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]),
            -6.709078077317236,
            1e-13, |w| w.ln_pdf(DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0])),
        );

        test_almost(
            3.0, DMatrix::from_row_slice(3, 3, &[
                1.0143, 0.0000, 0.0000,
                0.0000, 0.2034, 0.0000,
                0.0000, 0.0000, 0.9495
            ]),
            0.04313330476055326,
            1e-12, |w| w.pdf(DMatrix::from_row_slice(3, 3, &[
                0.7121, 0.0000, 0.0000,
                0.0000, 0.4010, 0.0000,
                0.0000, 0.0000, 0.5627,
            ])),
        );
        test_almost(
            3.0, DMatrix::from_row_slice(3, 3, &[
                1.0143, 0.0000, 0.0000,
                0.0000, 0.2034, 0.0000,
                0.0000, 0.0000, 0.9495
            ]),
            -3.1434598480128795,
            1e-12, |w| w.ln_pdf(DMatrix::from_row_slice(3, 3, &[
                0.7121, 0.0000, 0.0000,
                0.0000, 0.4010, 0.0000,
                0.0000, 0.0000, 0.5627,
            ])),
        );

        test_almost(
            3.0, DMatrix::from_row_slice(3, 3, &[
                1.0143, 0.0000, 0.0000,
                0.0000, 0.2034, 0.0000,
                0.0000, 0.0000, 0.9495
            ]),
            3.3174174014586637e-7,
            1e-12, |w| w.pdf(DMatrix::from_row_slice(3, 3, &[
                1.7121, 0.0000, 0.0000,
                0.0000, 0.4010, 0.0000,
                0.0000, 0.0000, 9.5627,
            ])),
        );
        test_almost(
            3.0, DMatrix::from_row_slice(3, 3, &[
                1.0143, 0.0000, 0.0000,
                0.0000, 0.2034, 0.0000,
                0.0000, 0.0000, 0.9495
            ]),
            -14.91890906187113,
            1e-12, |w| w.ln_pdf(DMatrix::from_row_slice(3, 3, &[
                1.7121, 0.0000, 0.0000,
                0.0000, 0.4010, 0.0000,
                0.0000, 0.0000, 9.5627,
            ])),
        );
    }

    #[test]
    fn test_sample() {
        let w = try_create(4.0, DMatrix::from_row_slice(3, 3, &[
            1.0143, 0.0000, 0.0000,
            0.0000, 0.2034, 0.0000,
            0.0000, 0.0000, 0.9495
        ]));

        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let sample = w.sample(&mut rng);
            let l_prob = w.ln_pdf(sample);
            assert!(l_prob.is_finite());
        }
    }
}
