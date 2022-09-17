use std::f64::consts::LN_2;
use nalgebra::{Cholesky, DMatrix, DVector, Dynamic};
use num_traits::Float;
use num_traits::real::Real;
use rand::Rng;
use crate::{Result, StatsError};
use crate::consts::LN_PI;
use crate::distribution::{ChiSquared, Continuous, Normal, ziggurat};
use crate::function::gamma::{digamma, mvgamma};
use crate::statistics::{MeanN, Mode, VarianceN};
use crate::function::gamma::mvlgamma;

fn mvdigamma(p: i64, a: f64) -> f64 {
    let mut sum = 0.0;
    for i in 0..p {
        sum += digamma(a - (i as f64) / 2.0);
    }
    sum
}


/// Implements the [Wishart distribution](http://en.wikipedia.org/wiki/Wishart_distribution)
///
/// # Example
/// ```
/// use nalgebra::DMatrix;
/// use statrs::distribution::{Wishart, Continuous};
/// use statrs::statistics::Distribution;
/// use statrs::prec;
///
/// let result = Wishart::new(2.0, DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]));
/// assert!(result.is_ok());
///
/// let result = Wishart::new(1.0, DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]));
/// assert!(result.is_err());
/// ```
#[derive(Debug, Clone)]
pub struct Wishart {
    freedom: f64,
    scale: DMatrix<f64>,
    chol: Cholesky<f64, Dynamic>,
}

impl Wishart {
    /// Constructs a new Wishart distribution with a degrees of freedom (ν) of `freedom`
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
    /// use statrs::distribution::Wishart;
    ///
    /// let result = Wishart::new(2.0, DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]));
    /// assert!(result.is_ok());
    ///
    /// let result = Wishart::new(1.0, DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]));
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
                Ok(Wishart { freedom, scale, chol })
            }
        }
    }

    /// Returns the degrees of freedom of
    /// the Wishart distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::DMatrix;
    /// use statrs::distribution::Wishart;
    ///
    /// let w = Wishart::new(2.0, DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0])).unwrap();
    /// assert_eq!(w.freedom(), 2.0);
    /// ```
    pub fn freedom(&self) -> f64 {
        self.freedom
    }

    /// Returns the scale of the Wishart distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::DMatrix;
    /// use statrs::distribution::Wishart;
    ///
    /// let w = Wishart::new(2.0, DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0])).unwrap();
    /// assert_eq!(w.scale(), &DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]));
    /// ```
    pub fn scale(&self) -> &DMatrix<f64> {
        &self.scale
    }

    /// Returns the dimensionality of the Wishart distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::DMatrix;
    /// use statrs::distribution::Wishart;
    ///
    /// let w = Wishart::new(3.0, DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0])).unwrap();
    /// assert_eq!(w.p(), 2);
    /// ```
    pub fn p(&self) -> usize {
        self.scale.nrows()
    }

    /// Returns the entropy of the Wishart distribution.
    /// See [formula](https://en.wikipedia.org/wiki/Wishart_distribution#Entropy)
    pub fn entropy(&self) -> Option<f64> {
        let p = self.p() as f64;
        Some(
            (p + 1.0) / 2.0 * self.chol.determinant().ln()
                + 0.5 * p * (p + 1.0) * LN_2
                + mvlgamma(p as i64, self.freedom / 2.0)
                - (self.freedom - p - 1.0) / 2.0 * mvdigamma(p as i64, self.freedom / 2.0)
                + self.freedom * p / 2.0
        )
    }
}

impl MeanN<DMatrix<f64>> for Wishart {
    /// Returns the mean of the Wishart distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// νψ
    /// ```
    ///
    /// where `ν` is the degree of freedom, `ψ` is the scale matrix
    fn mean(&self) -> Option<DMatrix<f64>> {
        Some(self.freedom * &self.scale)
    }
}

impl VarianceN<DMatrix<f64>> for Wishart {
    /// Returns the variance of the Wishart distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// Var(x_ij) = ν(ψ_ij^2 + ψ_ii ψ_jj)
    /// ```
    ///
    /// where `ν` is the degree of freedom, `ψ` is the scale matrix
    fn variance(&self) -> Option<DMatrix<f64>> {
        Some(self.scale.map_with_location(|i, j, x| {
            self.freedom * (self.scale[(i, i)] * self.scale[(j, j)] + x.powi(2))
        }))
    }
}

impl Mode<Option<DMatrix<f64>>> for Wishart {
    /// Returns the median of the Wishart distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// if k == 1 {
    ///     0
    /// } else {
    ///     λ((k - 1) / k)^(1 / k)
    /// }
    /// ```
    ///
    /// where `ν` is the degree of freedom, `ψ` is the scale matrix
    fn mode(&self) -> Option<DMatrix<f64>> {
        if self.freedom >= self.p() as f64 + 1.0 {
            Some((self.freedom - self.p() as f64 - 1.0) * &self.scale)
        } else {
            None
        }
    }
}

impl ::rand::distributions::Distribution<DMatrix<f64>> for Wishart {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> DMatrix<f64> {
        let p = self.p();
        let mut a = DMatrix::zeros(p, p);

        for i in 0..p {
            a[(i, i)] = ChiSquared::new(self.freedom - i as f64).unwrap().sample(rng).sqrt();
        }

        for i in 1..p {
            for j in 0..i {
                a[(i, j)] = ziggurat::sample_std_normal(rng);
            }
        }

        let l = self.chol.l() * &a;
        &l * &l.transpose()
    }
}

impl Continuous<DMatrix<f64>, f64> for Wishart {
    /// Calculates the probability density function for the Wishart
    /// distribution at `x`
    fn pdf(&self, x: DMatrix<f64>) -> f64 {
        let p = self.p() as f64;
        let x_det = x.determinant();
        let six = self.chol.solve(&x);

        x_det.powf((self.freedom - p - 1.0) / 2.0)
            * (-0.5 * six.trace()).exp()
            / (2.0f64).powf(self.freedom * p / 2.0)
            / self.chol.determinant().powf(self.freedom / 2.0)
            / mvgamma(p as i64, self.freedom / 2.0)
    }

    /// Calculates the log probability density function for the Wishart
    /// distribution at `x`
    fn ln_pdf(&self, x: DMatrix<f64>) -> f64 {
        let p = self.p() as f64;
        let x_lndet = x.determinant().ln();
        let six = self.chol.solve(&x);

        x_lndet * (self.freedom - p - 1.0) / 2.0
            - 0.5 * six.trace()
            - LN_2 * (self.freedom * p / 2.0)
            - self.chol.determinant().ln() * (self.freedom / 2.0)
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
    use crate::distribution::wishart::Wishart;
    use crate::statistics::{MeanN, Mode, VarianceN};

    fn try_create(freedom: f64, scale: DMatrix<f64>) -> Wishart
    {
        let w = Wishart::new(freedom, scale);
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
        F: FnOnce(Wishart) -> f64,
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
        F: FnOnce(Wishart) -> DMatrix<f64>,
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
            2.0, DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]),
            DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]),
            1e-15, |w| w.mean().unwrap(),
        );
    }

    #[test]
    fn test_mode() {
        test_almost_mat(
            7.0, DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]),
            DMatrix::from_row_slice(2, 2, &[4.0, 0.0, 0.0, 4.0]),
            1e-15, |w| w.mode().unwrap(),
        );
    }

    #[test]
    fn test_variance() {
        test_almost_mat(
            2.0, DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]),
            DMatrix::from_row_slice(2, 2, &[4.0, 2.0, 2.0, 4.0]),
            1e-15, |w| w.variance().unwrap(),
        );
        test_almost_mat(
            3.0, DMatrix::from_row_slice(3, 3, &[
                1.0143, 0.0000, 0.0000,
                0.0000, 0.2034, 0.0000,
                0.0000, 0.0000, 0.9495
            ]),
            DMatrix::from_row_slice(3, 3, &[
                6.17283, 0.618926, 2.88923,
                0.618926, 0.248229, 0.579385,
                2.88923, 0.579385, 5.4093,
            ]),
            1e-4, |w| w.variance().unwrap(),
        );
    }

    #[test]
    fn test_entropy() {
        test_almost(
            2.0, DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]),
            3.9538085820677584,
            1e-14, |w| w.entropy().unwrap(),
        );
        test_almost(
            3.0, DMatrix::from_row_slice(3, 3, &[
                1.0143, 0.0000, 0.0000,
                0.0000, 0.2034, 0.0000,
                0.0000, 0.0000, 0.9495
            ]), 6.315039109555716,
            1e-12, |w| w.entropy().unwrap(),
        );
        test_almost(
            3.0, DMatrix::from_row_slice(3, 3, &[
                1.0143, 0.0000, 0.0000,
                0.0000, 0.2034, 0.0000,
                0.0000, 0.0000, 9.9495
            ]), 11.013723204318477,
            1e-12, |w| w.entropy().unwrap(),
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
            3.0, DMatrix::from_row_slice(3, 3, &[
                1.0143, 0.0000, 0.0000,
                0.0000, 0.2034, 0.0000,
                0.0000, 0.0000, 0.9495
            ]),
            0.028397507846420644,
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
            -3.561453889551996,
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
            2.3729077174800438e-5,
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
            -10.648809376818907,
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
