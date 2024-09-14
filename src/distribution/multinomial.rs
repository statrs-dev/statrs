use crate::distribution::Discrete;
use crate::function::factorial;
use crate::statistics::*;
use nalgebra::{DVector, Dim, Dyn, OMatrix, OVector};

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
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, D>,
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
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, D>,
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
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, D>,
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
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<u64, D>,
{
    fn sample<R: ::rand::Rng + ?Sized>(&self, rng: &mut R) -> OVector<u64, D> {
        sample_generic(self.p.as_slice(), self.n, self.p.shape_generic().0, rng)
    }
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl<D> ::rand::distributions::Distribution<OVector<f64, D>> for Multinomial<D>
where
    D: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, D>,
{
    fn sample<R: ::rand::Rng + ?Sized>(&self, rng: &mut R) -> OVector<f64, D> {
        sample_generic(self.p.as_slice(), self.n, self.p.shape_generic().0, rng)
    }
}

#[cfg(feature = "rand")]
fn sample_generic<D, R, T>(p: &[f64], n: u64, dim: D, rng: &mut R) -> OVector<T, D>
where
    D: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<T, D>,
    R: ::rand::Rng + ?Sized,
    T: ::num_traits::Num
        + ::nalgebra::Scalar
        + num_traits::AsPrimitive<u64>
        + num_traits::FromPrimitive,
    super::Binomial: rand::distributions::Distribution<T>,
{
    use rand::distributions::Distribution;
    let mut res = OVector::zeros_generic(dim, nalgebra::U1);
    let mut probs_not_taken = 1.0;
    let mut samples_left = n;

    let mut p_sorted_inds: Vec<_> = (0..p.len()).collect();

    // unwrap because NAN elements not allowed from this struct's `new`
    p_sorted_inds.sort_unstable_by(|&i, &j| p[j].partial_cmp(&p[i]).unwrap());

    for ind in p_sorted_inds.into_iter().take(p.len() - 1) {
        let pi = p[ind];
        if pi == 0.0 {
            continue;
        }
        if !(0.0..=1.0).contains(&probs_not_taken) || samples_left == 0 {
            break;
        }
        let p_binom = pi / probs_not_taken;
        res[ind] = super::Binomial::new(p_binom, samples_left)
            .unwrap()
            .sample(rng);
        samples_left -= res[ind].as_();
        probs_not_taken -= pi;
    }
    if samples_left > 0 {
        *res.as_mut_slice().last_mut().unwrap() = T::from_u64(samples_left).unwrap();
    }
    res
}

impl<D> MeanN<DVector<f64>> for Multinomial<D>
where
    D: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, D>,
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
    fn mean(&self) -> Option<DVector<f64>> {
        Some(DVector::from_vec(
            self.p.iter().map(|x| x * self.n as f64).collect(),
        ))
    }
}

impl<D> VarianceN<OMatrix<f64, D, D>> for Multinomial<D>
where
    D: Dim,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<f64, D> + nalgebra::allocator::Allocator<f64, D, D>,
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
        let mut cov = OMatrix::from_diagonal(&self.p.map(|p| p * (1.0 - p)));
        let mut offdiag = |i: usize, j: usize| {
            let elt = -self.p[i] * self.p[j];
            // cov[(x, y)] = elt;
            cov[(j, i)] = elt;
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

impl<'a, D> Discrete<&'a OVector<u64, D>, f64> for Multinomial<D>
where
    D: Dim,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<f64, D> + nalgebra::allocator::Allocator<u64, D>,
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
    use nalgebra::{dmatrix, dvector, matrix, vector, Dyn, OVector};


    use super::boiler::*;

    #[test]
    fn test_create() {
        assert_relative_eq!(
            *create_ok(vector![1.0, 1.0, 1.0], 4).p(),
            vector![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        );
        create_ok(vector![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 4);
    }

    #[test]
    fn test_bad_create() {
        assert_eq!(
            create_err(vector![0.5], 4),
            MultinomialError::NotEnoughProbabilities,
        );

        assert_eq!(
            create_err(vector![-1.0, 2.0], 4),
            MultinomialError::ProbabilityInvalid,
        );

        assert_eq!(
            create_err(vector![0.0, 0.0], 4),
            MultinomialError::ProbabilitySumZero,
        );
        assert_eq!(
            create_err(vector![1.0, f64::NAN], 4),
            MultinomialError::ProbabilityInvalid,
        );
    }

    #[test]
    fn test_mean() {

        test_relative(vector![0.3, 0.7], 5, dvector![1.5, 3.5],  |x: Multinomial<_>| x.mean().unwrap());
        test_relative(
            dvector![0.1, 0.3, 0.6],
            10,
            dvector![1.0, 3.0, 6.0],
            |x: Multinomial<_>| x.mean().unwrap(),
       );
        test_relative(
            vector![1.0, 3.0, 6.0],
            10,
            dvector![1.0, 3.0, 6.0],
            |x: Multinomial<_>| x.mean().unwrap()
       );
        test_relative(
            vector![0.15, 0.35, 0.3, 0.2],
            20,
            dvector![3.0, 7.0, 6.0, 4.0],
            |x: Multinomial<_>| x.mean().unwrap()
       );
    }

    #[test]
    fn test_variance() {
        test_relative(
            vector![0.3, 0.7],
            5,
            matrix![1.05, -1.05; 
                    -1.05,  1.05],
            |x: Multinomial<_>| x.variance().unwrap(),
        );
        test_relative(
            vector![0.1, 0.3, 0.6],
            10,
            matrix![0.9, -0.3, -0.6;
                    -0.3,  2.1, -1.8;
                    -0.6, -1.8,  2.4;
            ],
            |x: Multinomial<_>| x.variance().unwrap(),
        );
        test_relative(
            dvector![0.15, 0.35, 0.3, 0.2],
            20,
            dmatrix![2.55, -1.05, -0.90, -0.60;
                    -1.05,  4.55, -2.10, -1.40;
                    -0.90, -2.10,  4.20, -1.20;
                    -0.60, -1.40, -1.20,  3.20;
            ],
            |x: Multinomial<_>| x.variance().unwrap(),
        );
    }

    // #[test]
    // fn test_skewness() {
    //     test_relative(vector![0.3, 0.7], 5, vector![0.390360029179413, -0.390360029179413], |x: Multinomial<_>| x.skewness().unwrap());
    //     test_relative(dvector![0.1, 0.3, 0.6], 10, dvector![0.843274042711568, 0.276026223736942, -0.12909944487358], |x: Multinomial<_>| x.skewness().unwrap());
    //     test_relative(vector![0.15, 0.35, 0.3, 0.2], 20, vector![0.438357003759605, 0.140642169281549, 0.195180014589707, 0.335410196624968], |x: Multinomial<_>| x.skewness().unwrap());
    // }

    #[test]
    fn test_pmf() {
        test_relative(
            vector![0.3, 0.7],
            10,
            0.121060821,
            move |x: Multinomial<_>| x.pmf(&vector![1, 9]),
        );
        test_relative(
            vector![0.1, 0.3, 0.6],
            10,
            0.105815808,
            move |x: Multinomial<_>| x.pmf(&vector![1, 3, 6]),
        );
        test_relative(
            dvector![0.15, 0.35, 0.3, 0.2],
            10,
            0.000145152,
            move |x: Multinomial<_>| x.pmf(&dvector![1, 1, 1, 7]),
        );
    }

    #[test]
    #[should_panic]
    fn test_pmf_x_wrong_length() {
        let pmf = |arg: OVector<u64, Dyn>| move |x: Multinomial<_>| x.pmf(&arg);
        test_relative(dvector![0.3, 0.7], 10, f64::NAN, pmf(dvector![1]));
    }

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

    #[cfg(feature = "rand")]
    #[test]
    fn test_almost_zero_sample() {
        use ::rand::{distributions::Distribution, prelude::thread_rng};
        let n = 10;
        let weights = vec![0.0, 0.0, 0.0, 0.1];
        let multinomial = Multinomial::new(weights, n).unwrap();
        let sample: OVector<f64, Dyn> = multinomial.sample(&mut thread_rng());
        assert_relative_eq!(sample[3], n as f64);
    }

    #[cfg(feature = "rand")]
    #[test]
    #[ignore = "this test is designed to assess approximately normal results within 2σ"]
    fn test_stochastic_uniform_samples() {
        use ::rand::{distributions::Distribution, prelude::thread_rng};
        use crate::statistics::Statistics;

        // describe multinomial such that each binomial variable is viable normal approximation
        let k: f64 = 60.0;
        let n: f64 = 1000.0;
        let weights = vec![1.0; k as usize];
        let multinomial = Multinomial::new(weights, n as u64).unwrap();

        // obtain sample statistics for multiple draws from this multinomial distribution
        // during iteration, verify that each event is ~ Binom(n, 1/k) under normal approx
        let n_trials = 20;
        let stats_across_multinom_events = std::iter::repeat_with(|| {
        let samples: OVector<f64, Dyn> = multinomial.sample(&mut thread_rng());
            samples.iter().enumerate().for_each(|(i, &s)| {
                assert_abs_diff_eq!(s, n / k, epsilon = multinomial.variance().unwrap()[(i, i)],)
            });
            samples
        })
        .take(n_trials)
        .map(|sample| (sample.mean(), sample.population_variance()));

        println!("{:#?}", stats_across_multinom_events.clone().collect::<Vec<_>>());

        // claim: for X from a given trial, Var[X] ~ χ^2(k-1)
        // the variance across this multinomial sample should be against the true mean
        //   Var[X] = sum (X_i - bar{X})^2 = sum (X_i - n/k)^2
        // alternatively, variance is linear, so we should also have
        //   Var[X] = k Var[X_i] = k npq = k n (k-1)/k^2 = n (k-1)/k as these are iid Binom(n, 1/k)
        //
        // since parameters of the binomial variable are around np ~ 20, each binomial variable is approximately normal
        //
        // therefore, population variance should be a sum of squares of normal variables from their mean.
        // i.e. population variances of these multinomial samples should be a scaling of χ^2 squared variables
        // with k-1 dof
        //
        // normal approximation of χ^2(k-1) should be valid for k = 50, so assert against 2σ_normal
        for (_mu, var) in stats_across_multinom_events {
            assert_abs_diff_eq!(
                var,
                k - 1.0,
                epsilon = ((2.0*k - 2.0)/n_trials as f64).sqrt()
            )
        }

    }
}

#[cfg(test)]
mod boiler {
    use super::{Multinomial, MultinomialError};
    use nalgebra::OVector;

    pub fn make_param_text<D>(p: &OVector<f64, D>, n: u64) -> String
    where
        D: nalgebra::Dim,
        nalgebra::DefaultAllocator:
            nalgebra::allocator::Allocator<f64, D> + nalgebra::allocator::Allocator<u64, D>,
    {
        // ""
        format!("n: {n}, p: {p}")
    }

    /// Creates and returns a distribution with the given parameters,
    /// panicking if `::new` fails.
    pub fn create_ok<D>(p: OVector<f64, D>, n: u64) -> Multinomial<D>
    where
        D: nalgebra::Dim,
        nalgebra::DefaultAllocator:
            nalgebra::allocator::Allocator<f64, D> + nalgebra::allocator::Allocator<u64, D>,
    {
        let param_text = make_param_text(&p, n);
        match Multinomial::new_from_nalgebra(p, n) {
            Ok(d) => d,
            Err(e) => panic!(
                "{}::new was expected to succeed, but failed for {} with error: '{}'",
                stringify!(Multinomial<D>),
                param_text,
                e
            ),
        }
    }

    /// Returns the error when creating a distribution with the given parameters,
    /// panicking if `::new` succeeds.
    pub fn create_err<D>(p: OVector<f64, D>, n: u64) -> MultinomialError
    where
        D: nalgebra::Dim,
        nalgebra::DefaultAllocator:
            nalgebra::allocator::Allocator<f64, D> + nalgebra::allocator::Allocator<u64, D>,
    {
        let param_text = make_param_text(&p, n);
        match Multinomial::new_from_nalgebra(p, n) {
            Err(e) => e,
            Ok(d) => panic!(
                "{}::new was expected to fail, but succeeded for {} with result: {:?}",
                stringify!(Multinomial<D>),
                param_text,
                d
            ),
        }
    }

    /// Creates a distribution with the given parameters, calls the `get_fn`
    /// function with the new distribution and returns the result of `get_fn`.
    ///
    /// Panics if ctor fails.
    pub fn create_and_get<F, T, D>(p: OVector<f64, D>, n: u64, get_fn: F) -> T
    where
        F: Fn(Multinomial<D>) -> T,
        D: nalgebra::Dim,
        nalgebra::DefaultAllocator:
            nalgebra::allocator::Allocator<f64, D> + nalgebra::allocator::Allocator<u64, D>,
    {
        let n = create_ok(p, n);
        get_fn(n)
    }

    /// Creates a distribution with the given parameters, calls the `get_fn`
    /// function with the new distribution and compares the result of `get_fn`
    /// to `expected` exactly.
    ///
    /// Panics if `::new` fails.
    #[allow(dead_code)]
    pub fn test_exact<F, T, D>(p: OVector<f64, D>, n: u64, expected: T, get_fn: F)
    where
        F: Fn(Multinomial<D>) -> T,
        T: ::core::cmp::PartialEq + ::core::fmt::Debug,
        D: nalgebra::Dim,
        nalgebra::DefaultAllocator:
            nalgebra::allocator::Allocator<f64, D> + nalgebra::allocator::Allocator<u64, D>,
    {
        let param_text = make_param_text(&p, n);
        let x = create_and_get(p, n, get_fn);
        if x != expected {
            panic!("Expected {:?}, got {:?} for {}", expected, x, param_text);
        }
    }

    /// Gets a value for the given parameters by calling `create_and_get`
    /// and compares it to `expected`.
    ///
    /// Allows relative error of up to [`crate::consts::ACC`].
    ///
    /// Panics if `::new` fails.
    #[allow(dead_code)]
    pub fn test_relative<F, T, U, D>(p: OVector<f64, D>, n: u64, expected: T, get_fn: F)
    where
        F: Fn(Multinomial<D>) -> U,
        T: ::std::fmt::Debug + ::approx::RelativeEq<U, Epsilon = f64>,
        U: ::std::fmt::Debug,
        D: nalgebra::Dim,
        nalgebra::DefaultAllocator:
            nalgebra::allocator::Allocator<f64, D> + nalgebra::allocator::Allocator<u64, D>,
    {
        let param_text = make_param_text(&p, n);
        let x = create_and_get(p, n, get_fn);
        let max_relative = crate::consts::ACC;

        if !::approx::relative_eq!(expected, x, max_relative = max_relative) {
            panic!(
                "Expected {:?} to be almost equal to {:?} (max. relative error of {:?}), but wasn't for {}",
                x,
                expected,
                max_relative,
                param_text
            );
        }
    }

    /// Gets a value for the given parameters by calling `create_and_get`
    /// and compares it to `expected`.
    ///
    /// Allows absolute error of up to `acc`.
    ///
    /// Panics if `::new` fails.
    #[allow(dead_code)]
    pub fn test_absolute<F, T, U, D>(p: OVector<f64, D>, n: u64, expected: T, acc: f64, get_fn: F)
    where
        F: Fn(Multinomial<D>) -> U,
        T: ::std::fmt::Display + ::approx::RelativeEq<U, Epsilon = f64> + num_traits::Float,
        U: ::std::fmt::Display,
        D: nalgebra::Dim,
        nalgebra::DefaultAllocator:
            nalgebra::allocator::Allocator<f64, D> + nalgebra::allocator::Allocator<u64, D>,
    {
        let param_text = make_param_text(&p, n);
        let x = create_and_get(p, n, get_fn);

        // abs_diff_eq! cannot handle infinities, so we manually accept them here
        if expected.is_infinite() {
            return;
        }

        if ::approx::abs_diff_ne!(expected, x, epsilon = acc) {
            panic!(
                "Expected {} to be almost equal to {} (max. absolute error of {:e}), but wasn't for {}",
                x,
                expected,
                acc,
                param_text
            );
        }
    }

    /// Purposely fails creating a distribution with the given
    /// parameters and compares the returned error to `expected`.
    ///
    /// Panics if `::new` succeeds.
    #[allow(dead_code)]
    pub fn test_create_err<D>(p: OVector<f64, D>, n: u64, expected: MultinomialError)
    where
        D: nalgebra::Dim,
        nalgebra::DefaultAllocator:
            nalgebra::allocator::Allocator<f64, D> + nalgebra::allocator::Allocator<u64, D>,
    {
        let param_text = make_param_text(&p, n);
        let err = create_err(p, n);
        if err != expected {
            panic!(
                "{}::new was expected to fail with error {expected}, but failed with error {err} for {param_text}",
                stringify!(Multinomial<D>),
            )
        }
    }

    /// Gets a value for the given parameters by calling `create_and_get`
    /// and asserts that it is [`NAN`].
    ///
    /// Panics if `::new` fails.
    #[allow(dead_code)]
    pub fn test_is_nan<F, D>(p: OVector<f64, D>, n: u64, get_fn: F)
    where
        F: Fn(Multinomial<D>) -> f64,
        D: nalgebra::Dim,
        nalgebra::DefaultAllocator:
            nalgebra::allocator::Allocator<f64, D> + nalgebra::allocator::Allocator<u64, D>,
    {
        let x = create_and_get(p, n, get_fn);
        assert!(x.is_nan());
    }

    /// Gets a value for the given parameters by calling `create_and_get`
    /// and asserts that it is [`None`].
    ///
    /// Panics if `::new` fails.
    #[allow(dead_code)]
    pub fn test_none<F, T, D>(p: OVector<f64, D>, n: u64, get_fn: F)
    where
        F: Fn(Multinomial<D>) -> Option<T>,
        T: ::core::fmt::Debug,
        D: nalgebra::Dim,
        nalgebra::DefaultAllocator:
            nalgebra::allocator::Allocator<f64, D> + nalgebra::allocator::Allocator<u64, D>,
    {
        let param_text = make_param_text(&p, n);
        let x = create_and_get(p, n, get_fn);

        if let Some(inner) = x {
            panic!("Expected None, got {:?} for {}", inner, param_text)
        }
    }

    /// Asserts that associated error type is Send and Sync
    #[test]
    pub fn test_error_is_sync_send() {
        pub fn assert_sync_send<T: Sync + Send>() {}
        assert_sync_send::<MultinomialError>();
    }
}
