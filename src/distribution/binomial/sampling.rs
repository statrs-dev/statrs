//! Random variate generation for the [`Binomial`] distribution.
//!
//! Two algorithms from Kachitvichyanukul & Schmeiser (1988): BINV, sequential
//! inversion, and BTPE, triangle/parallelogram/exponential rejection. Sampling
//! a [`Binomial`] picks between them per draw; [`Binomial::sampler`] lets a
//! caller choose instead.
//!
//! Kachitvichyanukul, V. and Schmeiser, B. W. (1988). Binomial random variate
//! generation. Communications of the ACM 31(2), 216-222.
//! <https://doi.org/10.1145/42372.42381>

use super::Binomial;
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl ::rand::distr::Distribution<u64> for Binomial {
    /// Samples in `O(1)` expected time (independent of `n`) via the BINV
    /// inversion algorithm for `n * min(p, 1 - p) < 10` and the BTPE
    /// triangle-parallelogram-exponential rejection algorithm otherwise.
    ///
    /// Kachitvichyanukul, V. and Schmeiser, B. W. (1988). Binomial random
    /// variate generation. Communications of the ACM 31(2), 216-222.
    /// <https://doi.org/10.1145/42372.42381>
    ///
    /// # Remarks
    ///
    /// This picks the algorithm for you, which is what you want unless you have
    /// a specific reason otherwise. To choose explicitly, build a
    /// [`BinomialSampler`] with [`Binomial::sampler`]; both algorithms are
    /// selectable, subject to BTPE being defined for the parameters.
    fn sample<R: ::rand::Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        sample_unchecked(rng, self.n, self.p)
    }
}

// The BINV/BTPE implementation below is adapted from the `rand_distr` crate
// (https://github.com/rust-random/rand_distr, Copyright 2018 Developers of the
// Rand project, Copyright 2016-2017 The Rust Project Developers; MIT or
// Apache-2.0), which implements Kachitvichyanukul & Schmeiser (1988) with the
// corrected Stirling-series signs from GSL.

/// Which algorithm `sample_unchecked` uses, after reflecting `p > 0.5` onto
/// `[0, 0.5]`. Carries the reduced parameters so the reduction happens once.
#[cfg(feature = "rand")]
#[derive(Debug, Clone, Copy, PartialEq)]
enum Path {
    /// `p` is 0 or 1, or there are no trials: the outcome is deterministic and
    /// already accounts for any reflection.
    Degenerate(u64),
    /// `1 - p` rounds to 1, so the distribution is Poisson(`mean`) to `O(p)`.
    PoissonLimit { mean: f64 },
    /// Sequential inversion, for `n * p < 10`.
    Binv { p: f64, q: f64 },
    /// Triangle/parallelogram/exponential rejection, for `n * p >= 10`.
    Btpe { p: f64, q: f64 },
}

/// A chosen [`Path`] together with whether the variate it produces counts
/// failures and so must be reflected back to `n - x`.
#[cfg(feature = "rand")]
#[derive(Debug, Clone, Copy, PartialEq)]
struct Plan {
    path: Path,
    flipped: bool,
}

#[cfg(feature = "rand")]
impl Plan {
    /// Chooses the sampling path for `(n, p)`.
    ///
    /// Kept free of the RNG so the branch selection can be tested directly. That
    /// matters most for [`Path::PoissonLimit`], which is otherwise unreachable
    /// from the public API in its reflected form: entering it needs
    /// `p <= 2^-54`, while reflection cannot produce a `p` below `2^-53`, the
    /// largest `f64` under one being `1 - 2^-53`. The reflection is still
    /// applied to that path -- see [`sample_unchecked`] -- rather than resting
    /// correctness on a one-bit spacing argument.
    fn new(n: u64, p: f64) -> Self {
        if p <= 0.0 || n == 0 {
            return Self {
                path: Path::Degenerate(0),
                flipped: false,
            };
        }
        if p >= 1.0 {
            return Self {
                path: Path::Degenerate(n),
                flipped: false,
            };
        }

        // the distribution is symmetric under p -> 1 - p
        let flipped = p > 0.5;
        let p = if flipped { 1.0 - p } else { p };
        let q = 1.0 - p;

        let path = if q == 1.0 {
            Path::PoissonLimit { mean: n as f64 * p }
        } else if n as f64 * p < 10.0 {
            // Threshold for preferring BINV; the paper suggests 10.
            Path::Binv { p, q }
        } else {
            Path::Btpe { p, q }
        };
        Self { path, flipped }
    }
}

/// Which algorithm draws a sample, for callers who want to choose rather than
/// take the default.
///
/// Obtain one via [`Binomial::sampler`]. Sampling a [`Binomial`] directly uses
/// [`Automatic`](Self::Automatic), which is the right choice unless you have a
/// specific reason otherwise.
#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum BinomialAlgorithm {
    /// Choose per draw from `n * min(p, 1 - p)`: inversion below 10, rejection
    /// at or above it. The threshold is the one suggested by the paper.
    #[default]
    Automatic,
    /// Sequential inversion (BINV).
    ///
    /// Defined for every parameter set, but the expected number of iterations
    /// is proportional to `n * min(p, 1 - p)`, so it grows without bound. Worth
    /// forcing if you want a shorter, more predictable code path and know the
    /// mean is small.
    Inversion,
    /// Triangle/parallelogram/exponential rejection (BTPE).
    ///
    /// Constant expected time regardless of `n`. Only defined once the triangle
    /// region is non-degenerate -- see
    /// [`BinomialAlgorithmError`] -- so it cannot be forced for a small mean.
    Rejection,
}

/// Returned by [`Binomial::sampler`] when the requested algorithm is not
/// defined for the distribution's parameters.
#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
#[derive(Copy, Clone, PartialEq, Debug)]
#[non_exhaustive]
pub enum BinomialAlgorithmError {
    /// [`BinomialAlgorithm::Rejection`] was requested for too small a mean.
    ///
    /// BTPE's triangle region has radius
    /// `floor(2.195 sqrt(n p q) - 4.6 q) + 0.5`, which is negative once
    /// `2.195 sqrt(n p q) < 4.6 q`, inverting the region boundaries. The
    /// sampler still terminates, but the distribution it produces is not
    /// binomial: at `n = 100, p = 0.04` a chi-square test against the exact pmf
    /// gives a statistic over 11 million on 14 degrees of freedom. Rejecting
    /// the request is therefore the only safe option.
    RejectionMeanTooSmall,
}

#[cfg(feature = "rand")]
impl core::fmt::Display for BinomialAlgorithmError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match self {
            BinomialAlgorithmError::RejectionMeanTooSmall => write!(
                f,
                "BTPE rejection sampling is undefined for this mean; 2.195*sqrt(n*p*q) must be at least 4.6*q"
            ),
        }
    }
}

#[cfg(feature = "rand")]
impl core::error::Error for BinomialAlgorithmError {}

/// A [`Binomial`] bound to a particular [`BinomialAlgorithm`], from
/// [`Binomial::sampler`].
///
/// The algorithm is validated once, when this is built, so drawing from it is
/// infallible and composes with the usual `rand` machinery.
///
/// ```
/// # #[cfg(feature = "rand")] {
/// use rand::{SeedableRng, distr::Distribution, rngs::StdRng};
/// use statrs::distribution::{Binomial, BinomialAlgorithm};
///
/// let dist = Binomial::new(0.3, 1_000).unwrap();
/// let sampler = dist.sampler(BinomialAlgorithm::Rejection).unwrap();
/// let mut rng = StdRng::seed_from_u64(0);
/// let draws: Vec<u64> = sampler.sample_iter(&mut rng).take(5).collect();
/// assert_eq!(draws.len(), 5);
/// # }
/// ```
#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BinomialSampler {
    n: u64,
    plan: Plan,
}

#[cfg(feature = "rand")]
impl ::rand::distr::Distribution<u64> for BinomialSampler {
    fn sample<R: ::rand::Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        sample_plan(rng, self.n, self.plan)
    }
}

#[cfg(feature = "rand")]
impl Binomial {
    /// Binds this distribution to a specific sampling algorithm.
    ///
    /// # Errors
    ///
    /// Returns [`BinomialAlgorithmError::RejectionMeanTooSmall`] if
    /// [`BinomialAlgorithm::Rejection`] is requested for parameters where BTPE
    /// is undefined. [`BinomialAlgorithm::Inversion`] and
    /// [`BinomialAlgorithm::Automatic`] never fail.
    ///
    /// # Remarks
    ///
    /// The choice applies only where there is one. Degenerate parameters
    /// (`p == 0`, `p == 1`, `n == 0`) return a constant, and a `p` so small that
    /// `1 - p` rounds to `1` is drawn from the Poisson limit; neither runs
    /// BINV or BTPE, so both ignore the request rather than fail.
    pub fn sampler(
        &self,
        algorithm: BinomialAlgorithm,
    ) -> Result<BinomialSampler, BinomialAlgorithmError> {
        let plan = Plan::new(self.n, self.p);
        let path = match (algorithm, plan.path) {
            (BinomialAlgorithm::Automatic, path) => path,
            (BinomialAlgorithm::Inversion, Path::Binv { p, q } | Path::Btpe { p, q }) => {
                Path::Binv { p, q }
            }
            (BinomialAlgorithm::Rejection, Path::Binv { p, q } | Path::Btpe { p, q }) => {
                if !btpe_is_defined(self.n, p, q) {
                    return Err(BinomialAlgorithmError::RejectionMeanTooSmall);
                }
                Path::Btpe { p, q }
            }
            // no algorithm runs on these, so there is nothing to override
            (_, path @ (Path::Degenerate(_) | Path::PoissonLimit { .. })) => path,
        };
        Ok(BinomialSampler {
            n: self.n,
            plan: Plan { path, ..plan },
        })
    }
}

/// Whether BTPE's triangle region is non-degenerate for these parameters, which
/// is what makes the algorithm valid. Mirrors the `p1` expression in [`btpe`].
#[cfg(feature = "rand")]
fn btpe_is_defined(n: u64, p: f64, q: f64) -> bool {
    2.195 * (n as f64 * p * q).sqrt() - 4.6 * q >= 0.0
}

/// Samples from a binomial distribution with the given `n` and `p`, without
/// validating the parameters.
#[cfg(feature = "rand")]
pub fn sample_unchecked<R: ::rand::Rng + ?Sized>(rng: &mut R, n: u64, p: f64) -> u64 {
    sample_plan(rng, n, Plan::new(n, p))
}

/// Draws according to an already-chosen [`Plan`], shared by [`sample_unchecked`]
/// and [`BinomialSampler`] so that both reflect identically.
#[cfg(feature = "rand")]
fn sample_plan<R: ::rand::Rng + ?Sized>(rng: &mut R, n: u64, plan: Plan) -> u64 {
    let Plan { path, flipped } = plan;
    let sample = match path {
        Path::Degenerate(x) => return x,
        // The clamp is needed because a Poisson variate has no upper bound,
        // unlike a binomial one.
        Path::PoissonLimit { mean } => {
            (crate::distribution::poisson::sample_unchecked(rng, mean) as u64).min(n)
        }
        Path::Binv { p, q } => binv(rng, n, p, q),
        Path::Btpe { p, q } => btpe(rng, n, p, q),
    };
    // Every non-degenerate path reflects here, in one place, so a path cannot
    // be added or changed without inheriting it.
    if flipped { n - sample } else { sample }
}

/// BINV: sequential inversion from x = 0. Expected iterations ~ n * p.
#[cfg(feature = "rand")]
fn binv<R: ::rand::Rng + ?Sized>(rng: &mut R, n: u64, p: f64, q: f64) -> u64 {
    // BINV can get numerically stuck accumulating `u -= r`; a result beyond
    // 110 is > 31 standard deviations out for n * p < 10, so restart instead
    // (same guard value as GSL and rand_distr).
    const BINV_MAX_X: u64 = 110;

    let s = p / q;
    let a = (n as f64 + 1.0) * s;
    // q^n, via ln_1p to keep accuracy for small p
    let r0 = ((-p).ln_1p() * n as f64).exp();

    'restart: loop {
        let mut r = r0;
        let mut u: f64 = ::rand::RngExt::random(rng);
        let mut x = 0u64;
        while u > r {
            u -= r;
            x += 1;
            if x > BINV_MAX_X {
                continue 'restart;
            }
            r *= a / (x as f64) - s;
        }
        return x;
    }
}

/// BTPE: rejection from a triangle + parallelogram + two exponential tails
/// envelope around the scaled pmf. Requires `p <= 0.5` and `n * p >= 10`.
#[cfg(feature = "rand")]
#[allow(clippy::many_single_char_names)] // same names as the reference paper
fn btpe<R: ::rand::Rng + ?Sized>(rng: &mut R, n: u64, p: f64, q: f64) -> u64 {
    use core::cmp::Ordering;

    // Below this |y - m| the pmf ratio f(y)/f(m) is evaluated directly.
    const SQUEEZE_THRESHOLD: u64 = 20;

    // Step 0: constants as functions of n and p.
    let n_f = n as f64;
    let np = n_f * p;
    let npq = np * q;
    let f_m = np + p;
    let m = f_m as u64; // mode
    // radius (and, height being 1, area) of the triangle region
    let p1 = (2.195 * npq.sqrt() - 4.6 * q).floor() + 0.5;
    let x_m = m as f64 + 0.5; // tip of the triangle
    let x_l = x_m - p1; // left edge of the triangle
    let x_r = x_m + p1; // right edge of the triangle
    let c = 0.134 + 20.5 / (15.3 + m as f64);
    // p1 + area of the parallelogram region
    let p2 = p1 * (1.0 + 2.0 * c);

    let lambda = |a: f64| a * (1.0 + 0.5 * a);
    let lambda_l = lambda((f_m - x_l) / (f_m - x_l * p));
    let lambda_r = lambda((x_r - f_m) / (x_r * q));
    let p3 = p2 + c / lambda_l;
    let p4 = p3 + c / lambda_r;

    loop {
        // Step 1: select the region via u; v decides acceptance within it.
        let u: f64 = ::rand::RngExt::random::<f64>(rng) * p4;
        let mut v: f64 = ::rand::RngExt::random(rng);

        let y: u64;
        if u <= p1 {
            // triangle: accept immediately
            return (x_m - p1 * v + u) as u64;
        } else if u <= p2 {
            // Step 2: parallelogram
            let x = x_l + (u - p1) / c;
            v = v * c + 1.0 - (x - x_m).abs() / p1;
            if v > 1.0 {
                continue;
            }
            y = x as u64;
        } else if u <= p3 {
            // Step 3: left exponential tail (v == 0 gives -inf and retries)
            let y_tmp = x_l + v.ln() / lambda_l;
            if y_tmp < 0.0 {
                continue;
            }
            y = y_tmp as u64;
            v *= (u - p2) * lambda_l;
        } else {
            // Step 4: right exponential tail (the `as` cast saturates)
            let y_tmp = x_r - v.ln() / lambda_r;
            if y_tmp > n_f {
                continue;
            }
            y = y_tmp as u64;
            v *= (u - p3) * lambda_r;
        }

        // Step 5: acceptance/rejection comparison of v against f(y)/f(m).
        let k = y.abs_diff(m);
        if k <= SQUEEZE_THRESHOLD || (k as f64) >= 0.5 * npq - 1.0 {
            // Step 5.1: evaluate the ratio via the recurrence from the mode.
            let s = p / q;
            let a = s * (n_f + 1.0);
            let mut f = 1.0;
            match m.cmp(&y) {
                Ordering::Less => {
                    for i in (m + 1)..=y {
                        f *= a / (i as f64) - s;
                    }
                }
                Ordering::Greater => {
                    for i in (y + 1)..=m {
                        f /= a / (i as f64) - s;
                    }
                }
                Ordering::Equal => {}
            }
            if v <= f {
                return y;
            }
            continue;
        }

        // Step 5.2: squeeze ln(v) between quadratic bounds on ln(f(y)/f(m)).
        let kf = k as f64;
        let rho = (kf / npq) * ((kf * (kf / 3.0 + 0.625) + 1.0 / 6.0) / npq + 0.5);
        let t = -0.5 * kf * kf / npq;
        let alpha = v.ln();
        if alpha < t - rho {
            return y;
        }
        if alpha > t + rho {
            continue;
        }

        // Step 5.3: final comparison against ln(f(y)/f(m)) via Stirling series.
        let x1 = (y + 1) as f64;
        let f1 = (m + 1) as f64;
        let z = ((n - m) + 1) as f64;
        let w = ((n - y) + 1) as f64;

        // 13860/166320 = 1/12, 462/166320 = 1/360, ...: the ln k! tail
        let stirling = |a: f64| {
            let a2 = a * a;
            (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / a2) / a2) / a2) / a2) / a / 166320.0
        };

        let y_sub_m = if y > m {
            (y - m) as f64
        } else {
            -((m - y) as f64)
        };
        // Sign convention on the Stirling terms follows GSL (verified correct
        // by one of the algorithm's original designers), not the paper.
        if alpha
            <= x_m * (f1 / x1).ln()
                + (((n - m) as f64) + 0.5) * (z / w).ln()
                + y_sub_m * (w * p / (x1 * q)).ln()
                + stirling(f1)
                + stirling(z)
                - stirling(x1)
                - stirling(w)
        {
            return y;
        }
    }
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl ::rand::distr::Distribution<f64> for Binomial {
    fn sample<R: ::rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        ::rand::RngExt::sample::<u64, _>(rng, self) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::Discrete;
    use crate::prec;

    fn create_ok(p: f64, n: u64) -> Binomial {
        let dist = Binomial::new(p, n);
        assert!(dist.is_ok());
        dist.unwrap()
    }

    /// Forcing an algorithm must still sample the right distribution. Same
    /// chi-square check as the default path, run for each explicit choice.
    #[cfg(all(feature = "rand", feature = "std"))]
    #[test]
    fn test_forced_algorithms_sample_correctly() {
        use crate::distribution::{BinomialAlgorithm, BinomialAlgorithmError};
        use crate::stats_tests::chisquare::chisquare;
        use ::rand::SeedableRng;
        use ::rand::distr::Distribution as _;
        use ::rand::rngs::StdRng;

        const SAMPLES: usize = 100_000;
        // (n, p, which algorithms are valid here)
        let cases: &[(u64, f64, &[BinomialAlgorithm])] = &[
            // small mean: inversion only, BTPE's triangle is degenerate
            (
                20,
                0.1,
                &[BinomialAlgorithm::Automatic, BinomialAlgorithm::Inversion],
            ),
            // large mean: both, so inversion is being forced off its default
            (
                200,
                0.4,
                &[
                    BinomialAlgorithm::Automatic,
                    BinomialAlgorithm::Inversion,
                    BinomialAlgorithm::Rejection,
                ],
            ),
            // flipped, both valid
            (
                500,
                0.85,
                &[
                    BinomialAlgorithm::Automatic,
                    BinomialAlgorithm::Inversion,
                    BinomialAlgorithm::Rejection,
                ],
            ),
        ];

        for &(n, p, algorithms) in cases {
            let dist = create_ok(p, n);
            for &algorithm in algorithms {
                let sampler = dist.sampler(algorithm).unwrap();
                let mut rng = StdRng::seed_from_u64(0xA1_60 + n);

                let mut counts = vec![0usize; (n + 1) as usize];
                for _ in 0..SAMPLES {
                    let x: u64 = sampler.sample(&mut rng);
                    assert!(x <= n, "{algorithm:?} n={n} p={p}: sampled {x} > n");
                    counts[x as usize] += 1;
                }

                let (mut observed, mut expected) = (Vec::new(), Vec::new());
                let mut acc = (0.0f64, 0usize);
                for k in 0..=n {
                    acc.0 += SAMPLES as f64 * dist.pmf(k);
                    acc.1 += counts[k as usize];
                    if acc.0 >= 5.0 {
                        expected.push(acc.0);
                        observed.push(acc.1);
                        acc = (0.0, 0);
                    }
                }
                *expected.last_mut().unwrap() += acc.0;
                *observed.last_mut().unwrap() += acc.1;
                let last = expected.len() - 1;
                let rest: f64 = expected[..last].iter().sum();
                expected[last] = SAMPLES as f64 - rest;

                let (statistic, pvalue) = chisquare(&observed, Some(&expected), None).unwrap();
                assert!(
                    pvalue > 1e-6,
                    "{algorithm:?} n={n} p={p}: chi-square = {statistic:.1}, p = {pvalue:.3e}"
                );
            }
        }

        // Rejection is refused where BTPE is undefined, rather than silently
        // producing the wrong distribution.
        let small = create_ok(0.04, 100);
        assert_eq!(
            small.sampler(BinomialAlgorithm::Rejection),
            Err(BinomialAlgorithmError::RejectionMeanTooSmall)
        );
        // the other two always work
        assert!(small.sampler(BinomialAlgorithm::Inversion).is_ok());
        assert!(small.sampler(BinomialAlgorithm::Automatic).is_ok());
    }

    /// The validity bound tracks BTPE's own triangle radius, and the default
    /// path never selects BTPE where it would be undefined.
    #[cfg(feature = "rand")]
    #[test]
    fn test_rejection_validity_bound() {
        use super::{Path, Plan, btpe_is_defined};
        use crate::distribution::BinomialAlgorithm;

        // measured boundary: n=100 is wrong at p=0.04, correct from p=0.044
        assert!(!btpe_is_defined(100, 0.04, 0.96));
        assert!(btpe_is_defined(100, 0.044, 0.956));

        for n in [1u64, 5, 20, 100, 1000, 100_000] {
            for i in 1..100 {
                let p = i as f64 / 100.0;
                // whenever Automatic picks BTPE, BTPE must be defined
                if let Path::Btpe { p: rp, q } = Plan::new(n, p).path {
                    assert!(
                        btpe_is_defined(n, rp, q),
                        "automatic chose BTPE where it is undefined: n={n} p={p}"
                    );
                }
                // and `sampler` agrees with the predicate
                let dist = create_ok(p, n);
                let forced = dist.sampler(BinomialAlgorithm::Rejection);
                let plan = Plan::new(n, p);
                if let Path::Binv { p: rp, q } | Path::Btpe { p: rp, q } = plan.path {
                    assert_eq!(forced.is_ok(), btpe_is_defined(n, rp, q), "n={n} p={p}");
                } else {
                    // degenerate or Poisson limit: no algorithm to reject
                    assert!(forced.is_ok(), "n={n} p={p}");
                }
            }
        }
    }

    /// Chi-square goodness-of-fit of the BINV/BTPE sampler against the exact
    /// pmf, over parameter sets covering every code path: BINV (np < 10), BTPE
    /// (np >= 10), and both flipped (p > 0.5) variants. Seeds are fixed, so
    /// this is deterministic; the 6-sigma acceptance threshold means a failure
    /// indicates a real sampler defect, not chance.
    ///
    /// Requires `std` as well as `rand`: the histogram and the pooled cells are
    /// sized from `n` and `p` at run time, and the crate is `no_std` without
    /// `alloc`. The sampler itself is exercised without `std` by
    /// [`test_sample_extreme_parameters_moments`], which needs no collections.
    #[cfg(all(feature = "rand", feature = "std"))]
    #[test]
    fn test_sample_chi_square_goodness_of_fit() {
        use ::rand::SeedableRng;
        use ::rand::distr::Distribution as _;
        use ::rand::rngs::StdRng;

        use crate::stats_tests::chisquare::chisquare;

        const SAMPLES: usize = 100_000;
        for &(n, p) in &[
            (20u64, 0.3f64), // BINV
            (100, 0.4),      // BTPE
            (1000, 0.02),    // BTPE, skewed
            (100, 0.93),     // BINV, flipped
            (2000, 0.995),   // BTPE, flipped
        ] {
            let dist = create_ok(p, n);
            let mut rng = StdRng::seed_from_u64(0x5EED + n);

            let mut counts = vec![0usize; (n + 1) as usize];
            for _ in 0..SAMPLES {
                let x: u64 = dist.sample(&mut rng);
                assert!(x <= n, "n={n} p={p}: sampled {x}, outside the support");
                counts[x as usize] += 1;
            }

            // Bin over the whole support, pooling adjacent outcomes until each
            // cell expects at least 5 -- the usual condition for the chi-square
            // approximation. Covering the full support rather than a window
            // also makes the observed and expected totals agree exactly, which
            // `chisquare` checks.
            let mut observed: Vec<usize> = Vec::new();
            let mut expected: Vec<f64> = Vec::new();
            let mut acc = (0.0f64, 0usize);
            for k in 0..=n {
                acc.0 += SAMPLES as f64 * dist.pmf(k);
                acc.1 += counts[k as usize];
                if acc.0 >= 5.0 {
                    expected.push(acc.0);
                    observed.push(acc.1);
                    acc = (0.0, 0);
                }
            }
            // Fold the leftover tail into the final cell so nothing is dropped.
            *expected.last_mut().unwrap() += acc.0;
            *observed.last_mut().unwrap() += acc.1;

            assert_eq!(
                observed.iter().sum::<usize>(),
                SAMPLES,
                "n={n} p={p}: binning lost samples"
            );

            // `chisquare` rejects inputs whose observed and expected totals
            // disagree, and summing the pmf over the support accumulates enough
            // rounding to trip that check. Pin the last cell to the exact
            // remainder instead: the sum of the others is unchanged, so the
            // total then lands on `SAMPLES` to the last bit (the subtraction is
            // exact by Sterbenz, the operands being within a factor of two).
            let last = expected.len() - 1;
            let rest: f64 = expected[..last].iter().sum();
            expected[last] = SAMPLES as f64 - rest;
            debug_assert_eq!(expected.iter().sum::<f64>(), SAMPLES as f64);

            let (statistic, pvalue) = chisquare(&observed, Some(&expected), None)
                .expect("observed and expected totals agree by construction");

            // The seeds are fixed, so this is deterministic. The threshold is
            // deliberately far out in the tail: a correct sampler yields uniform
            // p-values, so 1e-6 will not fire by chance, and anything below it
            // indicates a real defect rather than an unlucky run.
            assert!(
                pvalue > 1e-6,
                "n={n} p={p}: chi-square = {statistic:.1} over {} cells, p = {pvalue:.3e}",
                observed.len()
            );
        }
    }

    /// Moment sanity for parameter ranges the chi-square test can't cover:
    /// n far too large for the old O(n) Bernoulli sampler, and the
    /// Poisson-limit path (q rounds to 1.0, i.e. p below ~2^-54).
    #[cfg(feature = "rand")]
    #[test]
    fn test_sample_extreme_parameters_moments() {
        use ::rand::SeedableRng;
        use ::rand::distr::Distribution as _;
        use ::rand::rngs::StdRng;

        // n = 1e9: a single draw used to cost a billion Bernoulli trials
        let dist = create_ok(0.4, 1_000_000_000);
        let mut rng = StdRng::seed_from_u64(99);
        const SAMPLES: usize = 20_000;
        let mean = 4.0e8_f64;
        let sd = (1.0e9_f64 * 0.4 * 0.6).sqrt();
        let mut sum = 0.0;
        for _ in 0..SAMPLES {
            let x: u64 = dist.sample(&mut rng);
            assert!(
                (x as f64 - mean).abs() < 8.0 * sd,
                "sample {x} implausibly far out"
            );
            sum += x as f64;
        }
        let observed_mean = sum / SAMPLES as f64;
        // sample mean of 2e4 draws is within ~6 sd / sqrt(SAMPLES) of the mean
        prec::assert_abs_diff_eq!(
            observed_mean,
            mean,
            epsilon = 6.0 * sd / (SAMPLES as f64).sqrt()
        );

        // Poisson-limit path: p < 2^-54 so that 1 - p rounds to 1.0
        // n * p = 0.01
        let dist = create_ok(1e-18, 10_000_000_000_000_000);
        let mut rng = StdRng::seed_from_u64(7);
        let mut total = 0u64;
        for _ in 0..200_000 {
            total += <Binomial as ::rand::distr::Distribution<u64>>::sample(&dist, &mut rng);
        }
        // total ~ Poisson(200_000 * 0.01 = 2000); 6 sigma is +-268
        assert!(
            (total as f64 - 2000.0).abs() < 268.0,
            "Poisson-limit path total {total}"
        );
    }

    /// The branch selection, tested directly rather than by probing `f64` bit
    /// patterns through the sampler.
    #[cfg(feature = "rand")]
    #[test]
    fn test_sampling_path_selection() {
        use super::{Path, Plan};

        // degenerate parameters, which carry no reflection of their own
        assert_eq!(
            Plan::new(50, 0.0),
            Plan {
                path: Path::Degenerate(0),
                flipped: false
            }
        );
        assert_eq!(
            Plan::new(50, 1.0),
            Plan {
                path: Path::Degenerate(50),
                flipped: false
            }
        );
        assert_eq!(
            Plan::new(0, 0.5),
            Plan {
                path: Path::Degenerate(0),
                flipped: false
            }
        );

        // n * p straddling the BINV/BTPE threshold of 10
        assert!(matches!(
            Plan::new(20, 0.3),
            Plan {
                path: Path::Binv { .. },
                flipped: false
            }
        ));
        assert!(matches!(
            Plan::new(100, 0.4),
            Plan {
                path: Path::Btpe { .. },
                flipped: false
            }
        ));

        // p > 0.5 is reflected, and the threshold then applies to the reduced p
        assert!(matches!(
            Plan::new(100, 0.93),
            Plan {
                path: Path::Binv { .. },
                flipped: true
            }
        ));
        assert!(matches!(
            Plan::new(2000, 0.995),
            Plan {
                path: Path::Btpe { .. },
                flipped: true
            }
        ));

        // the Poisson limit: p small enough that 1 - p rounds to 1
        assert_eq!(
            Plan::new(10_000_000_000_000_000, 1e-18),
            Plan {
                path: Path::PoissonLimit { mean: 0.01 },
                flipped: false
            }
        );
    }

    /// The Poisson limit is unreachable in its reflected form: entering it needs
    /// `p <= 2^-54`, while reflection cannot produce a `p` below `2^-53`. The
    /// margin is one bit, so pin it -- a change to the guard that makes the path
    /// reachable should fail here rather than silently return an unreflected
    /// variate. Expressed over the path selection, so it tests the branch logic
    /// and not the float representation.
    #[cfg(feature = "rand")]
    #[test]
    fn test_poisson_limit_never_reached_flipped() {
        use super::{Path, Plan};

        // walk the f64 values immediately below 1, which are the only candidates
        let mut p = 1.0f64;
        for _ in 0..64 {
            p = f64::from_bits(p.to_bits() - 1);
            let plan = Plan::new(u64::MAX, p);
            assert!(
                !(plan.flipped && matches!(plan.path, Path::PoissonLimit { .. })),
                "p = {p:e} reached the Poisson limit while flipped"
            );
        }

        // and the same for the reflected value the guard would have to admit
        let largest_below_one = f64::from_bits(1.0f64.to_bits() - 1);
        assert_eq!(
            1.0 - largest_below_one,
            (-53f64).exp2(),
            "reflection bottoms out at 2^-53"
        );
        assert_eq!(
            1.0 - (-54f64).exp2(),
            1.0,
            "the guard admits 2^-54 and below"
        );
    }

    /// The reflected side at its extreme: `p` is the largest value below one, so
    /// the reduced `p` is `2^-53`, one bit clear of the Poisson guard and on the
    /// BINV path. Counts failures rather than successes, which is what the
    /// reflection has to get right.
    #[cfg(feature = "rand")]
    #[test]
    fn test_sample_p_adjacent_to_one() {
        use ::rand::SeedableRng;
        use ::rand::distr::Distribution as _;
        use ::rand::rngs::StdRng;

        let n = 10_000_000_000_000_000u64;
        let p = f64::from_bits(1.0f64.to_bits() - 1);
        let dist = create_ok(p, n);
        let mut rng = StdRng::seed_from_u64(11);

        const DRAWS: usize = 200_000;
        let mut failures = 0u64;
        for _ in 0..DRAWS {
            let x: u64 = dist.sample(&mut rng);
            assert!(x <= n, "sample {x} exceeds n");
            failures += n - x;
        }

        // failures ~ Poisson(DRAWS * n * 2^-53); 6 sigma either side
        let mean = DRAWS as f64 * n as f64 * (-53f64).exp2();
        let tol = 6.0 * mean.sqrt();
        assert!(
            (failures as f64 - mean).abs() < tol,
            "{failures} failures, expected {mean:.0} +- {tol:.0}"
        );
    }

    /// Degenerate parameters short-circuit.
    #[cfg(feature = "rand")]
    #[test]
    fn test_sample_degenerate() {
        use ::rand::SeedableRng;
        use ::rand::rngs::StdRng;

        let mut rng = StdRng::seed_from_u64(3);
        for _ in 0..10 {
            assert_eq!(
                <Binomial as ::rand::distr::Distribution<u64>>::sample(
                    &create_ok(0.0, 50),
                    &mut rng
                ),
                0
            );
            assert_eq!(
                <Binomial as ::rand::distr::Distribution<u64>>::sample(
                    &create_ok(1.0, 50),
                    &mut rng
                ),
                50
            );
            assert_eq!(
                <Binomial as ::rand::distr::Distribution<u64>>::sample(
                    &create_ok(0.5, 0),
                    &mut rng
                ),
                0
            );
        }
    }
}
