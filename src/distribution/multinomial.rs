use crate::distribution::Discrete;
use crate::function::factorial;
use crate::statistics::*;
use nalgebra::{Dim, Dyn, OMatrix, OVector};

/// Implements the
/// [Multinomial](https://en.wikipedia.org/wiki/Multinomial_distribution)
/// distribution which is a generalization of the
/// [Binomial](https://en.wikipedia.org/wiki/Binomial_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::Multinomial;
/// use statrs::statistics::MeanN;
/// use nalgebra::vector;
///
/// let n = Multinomial::new_from_nalgebra(vector![0.3, 0.7], 5).unwrap();
/// assert_eq!(n.mean().unwrap(), (vector![1.5, 3.5]));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Multinomial<D>
where
    D: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    /// normalized probabilities for each species
    p: OVector<f64, D>,
    /// count of trials
    n: u64,
}

/// Represents the errors that can occur when creating a [`Multinomial`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum MultinomialError {
    /// Fewer than two probabilities.
    NotEnoughProbabilities,

    /// The sum of all probabilities is zero.
    ProbabilitySumZero,

    /// At least one probability is NaN, infinite or less than zero.
    ProbabilityInvalid,
}

impl std::fmt::Display for MultinomialError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            MultinomialError::NotEnoughProbabilities => write!(f, "Fewer than two probabilities"),
            MultinomialError::ProbabilitySumZero => write!(f, "The probabilities sum up to zero"),
            MultinomialError::ProbabilityInvalid => write!(
                f,
                "At least one probability is NaN, infinity or less than zero"
            ),
        }
    }
}

impl std::error::Error for MultinomialError {}

impl Multinomial<Dyn> {
    /// Constructs a new multinomial distribution with probabilities `p`
    /// and `n` number of trials.
    ///
    /// # Errors
    ///
    /// Returns an error if `p` is empty, the sum of the elements
    /// in `p` is 0, or any element in `p` is less than 0 or is `f64::NAN`
    ///
    /// # Note
    ///
    /// The elements in `p` do not need to be normalized
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Multinomial;
    ///
    /// let mut result = Multinomial::new(vec![0.0, 1.0, 2.0], 3);
    /// assert!(result.is_ok());
    ///
    /// result = Multinomial::new(vec![0.0, -1.0, 2.0], 3);
    /// assert!(result.is_err());
    /// ```
    pub fn new(p: Vec<f64>, n: u64) -> Result<Self, MultinomialError> {
        Self::new_from_nalgebra(p.into(), n)
    }
}

impl<D> Multinomial<D>
where
    D: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    pub fn new_from_nalgebra(mut p: OVector<f64, D>, n: u64) -> Result<Self, MultinomialError> {
        if p.len() < 2 {
            return Err(MultinomialError::NotEnoughProbabilities);
        }

        let mut sum = 0.0;
        for &val in &p {
            if val.is_nan() || val < 0.0 {
                return Err(MultinomialError::ProbabilityInvalid);
            }

            sum += val;
        }

        if sum == 0.0 {
            return Err(MultinomialError::ProbabilitySumZero);
        }

        p.unscale_mut(p.lp_norm(1));
        Ok(Self { p, n })
    }

    /// Returns the probabilities of the multinomial
    /// distribution as a slice
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Multinomial;
    /// use nalgebra::dvector;
    ///
    /// let n = Multinomial::new(vec![0.0, 1.0, 2.0], 3).unwrap();
    /// assert_eq!(*n.p(), dvector![0.0, 1.0/3.0, 2.0/3.0]);
    /// ```
    pub fn p(&self) -> &OVector<f64, D> {
        &self.p
    }

    /// Returns the number of trials of the multinomial
    /// distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Multinomial;
    ///
    /// let n = Multinomial::new(vec![0.0, 1.0, 2.0], 3).unwrap();
    /// assert_eq!(n.n(), 3);
    /// ```
    pub fn n(&self) -> u64 {
        self.n
    }
}

impl<D> std::fmt::Display for Multinomial<D>
where
    D: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Multinom({:#?},{})", self.p, self.n)
    }
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl<D> ::rand::distributions::Distribution<OVector<u64, D>> for Multinomial<D>
where
    D: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    fn sample<R: ::rand::Rng + ?Sized>(&self, rng: &mut R) -> OVector<u64, D> {
        sample_generic(self, rng)
    }
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl<D> ::rand::distributions::Distribution<OVector<f64, D>> for Multinomial<D>
where
    D: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    fn sample<R: ::rand::Rng + ?Sized>(&self, rng: &mut R) -> OVector<f64, D> {
        sample_generic(self, rng)
    }
}

#[cfg(feature = "rand")]
fn sample_generic<D, R, T>(dist: &Multinomial<D>, rng: &mut R) -> OVector<T, D>
where
    D: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
    R: ::rand::Rng + ?Sized,
    T: ::num_traits::Num + ::nalgebra::Scalar + ::std::ops::AddAssign<T>,
{
    use nalgebra::Const;

    let p_cdf = super::categorical::prob_mass_to_cdf(dist.p().as_slice());
    let mut res = OVector::zeros_generic(dist.p.shape_generic().0, Const::<1>);
    for _ in 0..dist.n {
        let i = super::categorical::sample_unchecked(rng, &p_cdf);
        res[i] += T::one();
    }
    res
}

impl<D> MeanN<OVector<f64, D>> for Multinomial<D>
where
    D: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    /// Returns the mean of the multinomial distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// n * p_i for i in 1...k
    /// ```
    ///
    /// where `n` is the number of trials, `p_i` is the `i`th probability,
    /// and `k` is the total number of probabilities
    fn mean(&self) -> Option<OVector<f64, D>> {
        Some(self.p.map(|x| x * self.n as f64))
    }
}

impl<D> VarianceN<OMatrix<f64, D, D>> for Multinomial<D>
where
    D: Dim,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D> + nalgebra::allocator::Allocator<D, D>,
{
    /// Returns the variance of the multinomial distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// n * p_i * (1 - p_i) for i in 1...k
    /// ```
    ///
    /// where `n` is the number of trials, `p_i` is the `i`th probability,
    /// and `k` is the total number of probabilities
    fn variance(&self) -> Option<OMatrix<f64, D, D>> {
        let mut cov = OMatrix::from_diagonal(&self.p.map(|x| x * (1.0 - x)));
        let mut offdiag = |x: usize, y: usize| {
            let elt = -self.p[x] * self.p[y];
            // cov[(x, y)] = elt;
            cov[(y, x)] = elt;
        };

        for i in 0..self.p.len() {
            for j in 0..i {
                offdiag(i, j);
            }
        }
        cov.fill_lower_triangle_with_upper_triangle();
        Some(cov.scale(self.n as f64))
    }
}

// impl Skewness<Vec<f64>> for Multinomial {
//     /// Returns the skewness of the multinomial distribution
//     ///
//     /// # Formula
//     ///
//     /// ```text
//     /// (1 - 2 * p_i) / (n * p_i * (1 - p_i)) for i in 1...k
//     /// ```
//     ///
//     /// where `n` is the number of trials, `p_i` is the `i`th probability,
//     /// and `k` is the total number of probabilities
//     fn skewness(&self) -> Option<Vec<f64>> {
//         Some(
//             self.p
//                 .iter()
//                 .map(|x| (1.0 - 2.0 * x) / (self.n as f64 * (1.0 - x) * x).sqrt())
//                 .collect(),
//         )
//     }
// }

impl<D> Discrete<&OVector<u64, D>, f64> for Multinomial<D>
where
    D: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    /// Calculates the probability mass function for the multinomial
    /// distribution
    /// with the given `x`'s corresponding to the probabilities for this
    /// distribution
    ///
    /// # Panics
    ///
    /// If length of `x` is not equal to length of `p`
    ///
    /// # Formula
    ///
    /// ```text
    /// (n! / x_1!...x_k!) * p_i^x_i for i in 1...k
    /// ```
    ///
    /// where `n` is the number of trials, `p_i` is the `i`th probability,
    /// `x_i` is the `i`th `x` value, and `k` is the total number of
    /// probabilities
    fn pmf(&self, x: &OVector<u64, D>) -> f64 {
        if self.p.len() != x.len() {
            panic!("Expected x and p to have equal lengths.");
        }
        if x.iter().sum::<u64>() != self.n {
            return 0.0;
        }
        let coeff = factorial::multinomial(self.n, x.as_slice());
        let val = coeff
            * self
                .p
                .iter()
                .zip(x.iter())
                .fold(1.0, |acc, (pi, xi)| acc * pi.powf(*xi as f64));
        val
    }

    /// Calculates the log probability mass function for the multinomial
    /// distribution
    /// with the given `x`'s corresponding to the probabilities for this
    /// distribution
    ///
    /// # Panics
    ///
    /// If length of `x` is not equal to length of `p`
    ///
    /// # Formula
    ///
    /// ```text
    /// ln((n! / x_1!...x_k!) * p_i^x_i) for i in 1...k
    /// ```
    ///
    /// where `n` is the number of trials, `p_i` is the `i`th probability,
    /// `x_i` is the `i`th `x` value, and `k` is the total number of
    /// probabilities
    fn ln_pmf(&self, x: &OVector<u64, D>) -> f64 {
        if self.p.len() != x.len() {
            panic!("Expected x and p to have equal lengths.");
        }
        if x.iter().sum::<u64>() != self.n {
            return f64::NEG_INFINITY;
        }
        let coeff = factorial::multinomial(self.n, x.as_slice()).ln();
        let val = coeff
            + self
                .p
                .iter()
                .zip(x.iter())
                .map(|(pi, xi)| *xi as f64 * pi.ln())
                .fold(0.0, |acc, x| acc + x);
        val
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use crate::{
        distribution::{Discrete, Multinomial, MultinomialError},
        statistics::{MeanN, VarianceN},
    };
    use nalgebra::{dmatrix, dvector, vector, DimMin, Dyn, OVector};
    use std::fmt::{Debug, Display};

    fn try_create<D>(p: OVector<f64, D>, n: u64) -> Multinomial<D>
    where
        D: DimMin<D, Output = D>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
    {
        let mvn = Multinomial::new_from_nalgebra(p, n);
        assert!(mvn.is_ok());
        mvn.unwrap()
    }

    fn bad_create_case<D>(p: OVector<f64, D>, n: u64) -> MultinomialError
    where
        D: DimMin<D, Output = D>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
    {
        let dd = Multinomial::new_from_nalgebra(p, n);
        assert!(dd.is_err());
        dd.unwrap_err()
    }

    fn test_almost<F, T, D>(p: OVector<f64, D>, n: u64, expected: T, acc: f64, eval: F)
    where
        T: Debug + Display + approx::RelativeEq<Epsilon = f64>,
        F: FnOnce(Multinomial<D>) -> T,
        D: DimMin<D, Output = D>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
    {
        let dd = try_create(p, n);
        let x = eval(dd);
        assert_relative_eq!(expected, x, epsilon = acc);
    }

    #[test]
    fn test_create() {
        assert_relative_eq!(
            *try_create(vector![1.0, 1.0, 1.0], 4).p(),
            vector![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        );
        try_create(dvector![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 4);
    }

    #[test]
    fn test_bad_create() {
        assert_eq!(
            bad_create_case(vector![0.5], 4),
            MultinomialError::NotEnoughProbabilities,
        );

        assert_eq!(
            bad_create_case(vector![-1.0, 2.0], 4),
            MultinomialError::ProbabilityInvalid,
        );

        assert_eq!(
            bad_create_case(vector![0.0, 0.0], 4),
            MultinomialError::ProbabilitySumZero,
        );
        assert_eq!(
            bad_create_case(vector![1.0, f64::NAN], 4),
            MultinomialError::ProbabilityInvalid,
        );
    }

    #[test]
    fn test_mean() {
        let mean = |x: Multinomial<_>| x.mean().unwrap();
        test_almost(dvector![0.3, 0.7], 5, dvector![1.5, 3.5], 1e-12, mean);
        test_almost(
            dvector![0.1, 0.3, 0.6],
            10,
            dvector![1.0, 3.0, 6.0],
            1e-12,
            mean,
        );
        test_almost(
            dvector![1.0, 3.0, 6.0],
            10,
            dvector![1.0, 3.0, 6.0],
            1e-12,
            mean,
        );
        test_almost(
            dvector![0.15, 0.35, 0.3, 0.2],
            20,
            dvector![3.0, 7.0, 6.0, 4.0],
            1e-12,
            mean,
        );
    }

    #[test]
    fn test_variance() {
        let variance = |x: Multinomial<_>| x.variance().unwrap();
        test_almost(
            dvector![0.3, 0.7],
            5,
            dmatrix![1.05, -1.05; 
                    -1.05,  1.05],
            1e-15,
            variance,
        );
        test_almost(
            dvector![0.1, 0.3, 0.6],
            10,
            dmatrix![0.9, -0.3, -0.6;
                    -0.3,  2.1, -1.8;
                    -0.6, -1.8,  2.4;
            ],
            1e-15,
            variance,
        );
        test_almost(
            dvector![0.15, 0.35, 0.3, 0.2],
            20,
            dmatrix![2.55, -1.05, -0.90, -0.60;
                    -1.05,  4.55, -2.10, -1.40;
                    -0.90, -2.10,  4.20, -1.20;
                    -0.60, -1.40, -1.20,  3.20;
            ],
            1e-15,
            variance,
        );
    }

    //     // #[test]
    //     // fn test_skewness() {
    //     //     let skewness = |x: Multinomial| x.skewness().unwrap();
    //     //     test_almost(&[0.3, 0.7], 5, &[0.390360029179413, -0.390360029179413], 1e-15, skewness);
    //     //     test_almost(&[0.1, 0.3, 0.6], 10, &[0.843274042711568, 0.276026223736942, -0.12909944487358], 1e-15, skewness);
    //     //     test_almost(&[0.15, 0.35, 0.3, 0.2], 20, &[0.438357003759605, 0.140642169281549, 0.195180014589707, 0.335410196624968], 1e-15, skewness);
    //     // }

    #[test]
    fn test_pmf() {
        let pmf = |arg: OVector<u64, Dyn>| move |x: Multinomial<_>| x.pmf(&arg);
        test_almost(
            dvector![0.3, 0.7],
            10,
            0.121060821,
            1e-15,
            pmf(dvector![1, 9]),
        );
        test_almost(
            dvector![0.1, 0.3, 0.6],
            10,
            0.105815808,
            1e-15,
            pmf(dvector![1, 3, 6]),
        );
        test_almost(
            dvector![0.15, 0.35, 0.3, 0.2],
            10,
            0.000145152,
            1e-15,
            pmf(dvector![1, 1, 1, 7]),
        );
    }

    #[test]
    fn test_error_is_sync_send() {
        fn assert_sync_send<T: Sync + Send>() {}
        assert_sync_send::<MultinomialError>();
    }

    //     #[test]
    //     #[should_panic]
    //     fn test_pmf_x_wrong_length() {
    //         let pmf = |arg: &[u64]| move |x: Multinomial| x.pmf(arg);
    //         let n = Multinomial::new(&[0.3, 0.7], 10).unwrap();
    //         n.pmf(&[1]);
    //     }

    //     #[test]
    //     #[should_panic]
    //     fn test_pmf_x_wrong_sum() {
    //         let pmf = |arg: &[u64]| move |x: Multinomial| x.pmf(arg);
    //         let n = Multinomial::new(&[0.3, 0.7], 10).unwrap();
    //         n.pmf(&[1, 3]);
    //     }

    //     #[test]
    //     fn test_ln_pmf() {
    //         let large_p = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    //         let n = Multinomial::new(large_p, 45).unwrap();
    //         let x = &[1, 2, 3, 4, 5, 6, 7, 8, 9];
    //         assert_almost_eq!(n.pmf(x).ln(), n.ln_pmf(x), 1e-13);
    //         let n2 = Multinomial::new(large_p, 18).unwrap();
    //         let x2 = &[1, 1, 1, 2, 2, 2, 3, 3, 3];
    //         assert_almost_eq!(n2.pmf(x2).ln(), n2.ln_pmf(x2), 1e-13);
    //         let n3 = Multinomial::new(large_p, 51).unwrap();
    //         let x3 = &[5, 6, 7, 8, 7, 6, 5, 4, 3];
    //         assert_almost_eq!(n3.pmf(x3).ln(), n3.ln_pmf(x3), 1e-13);
    //     }

    //     #[test]
    //     #[should_panic]
    //     fn test_ln_pmf_x_wrong_length() {
    //         let n = Multinomial::new(&[0.3, 0.7], 10).unwrap();
    //         n.ln_pmf(&[1]);
    //     }

    //     #[test]
    //     #[should_panic]
    //     fn test_ln_pmf_x_wrong_sum() {
    //         let n = Multinomial::new(&[0.3, 0.7], 10).unwrap();
    //         n.ln_pmf(&[1, 3]);
    //     }
}
