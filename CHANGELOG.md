v0.16.0

- Adds an `sf` method to the `ContinuousCDF` and `DiscreteCDF` traits
  - Calculates the survival function (CDF complement) for the distribution.
- Survival function implemented for all distributions implementing `ContinuousCDF` and `DiscreteCDF`
  - See [PR description](https://github.com/statrs-dev/statrs/pull/172) for in-depth changes

v0.15.0

- upgrade `nalgebra` to `0.27.1` to avoid RUSTSEC-2021-0070

v0.14.0

- upgrade `rand` dependency to `0.8`
- fix inaccurate sampling of `Gamma`
- Implemented Empirical distribution
- Implemented Laplace distribution
- Removed Checked\* traits
- Almost clippy-clean
- Almost fully enabled rustfmt
- Begin applying consistent numeric relative-accuracy targets with the approx crate
- Introduce macro to generate testing boilerplate, yet not all tests use this yet
- Moved to dynamic vectors in the MultivariateNormal distribution
- Reduced a number of distribution-specific traits into the Distribution and DiscreteDistribution traits

v0.13.0

- Implemented `MultivariateNormal` distribution (depends on `nalgebra 0.19`)
- Implemented `Dirac` distribution
- Implemented `Negative Binomial` distribution

v0.12.0

- upgrade `rand` dependency to `0.7`

v0.11.0

- upgrade `rand` dependency to `0.6`
- Implement `CheckedInverseCDF` and `InverseCDF` for `Normal` distribution

v0.10.0

- upgrade `rand` dependency to `0.5`
- Removes the `Distribution` trait in favor of the `rand::distributions::Distribution` trait
- Removed functions deprecated in `0.8.0` (`periodic`, `periodic_custom`, `sinusoidal`, `sinusoidal_custom`)

v0.9.0

- implemented infinite sequence generator for periodic sequence
- implemented infinite sequence generator for sinusoidal sequence
- implemented infinite sequence generator for square sequence
- implemented infinite sequence generator for triangle sequence
- implemented infinite sequence generator for sawtooth sequence
- deprecate old non-infinite iterators in favor of new infinite iterators with `take`
- Implemented `Pareto` distribution
- Implemented `Entropy` trait for the `Categorical` distribution
- Add a `checked_` interface to all distribution methods and functions that may panic

v0.8.0

- `cdf(x)`, `pdf(x)` and `pmf(x)` now return the correct value instead of panicking when `x` is outside the range of values that the distribution can attain.
- Fixed a bug in the `Uniform` distribution implementation where samples were drawn from range `[min, max + 1)` instead of `[min, max]`. The samples are now drawn correctly from the range `[min, max]`.
- Implement `generate::log_spaced` function
- Implement `generate::Periodic` iterator
- Implement `generate::Sinusoidal` iterator
- Implement `generate::Square` iterator
- Implement `generate::Triangle` iterator
- Implement `generate::Sawtooth` iterator
- Deprecate `generate::periodic` and `generate::periodic_custom`
- Deprecate `generate::sinusoidal` and `generate::sinusoidal_custom`

Note: A recent commit to the Rust nightly build causes compile errors when using
empty slices with the `Statistics` trait, specifically the `Statistics::min` and
`Statistics::max` methods. This only affects the case where the compiler must infer
the type of the empty slice:

```
use statrs::statistics::Statistics;

// compile error! Assumes the use of Ord::min rather than
// Statistcs::min
let x = [];
assert!(x.min().is_nan());
```

The fix is to pin the type of the empty slice:

```
// no compile error
let x: [f64; 0] = [];
assert!(x.min().is_nan());
```

Since the regression affects a very slim edge-case and the fix is very simple, no breaking changes to the `Statistics` API was deemed necessary

v0.7.0

- Implemented `Categorical` distribution
- Implemented `Erlang` distribution
- Implemented `Multinomial` distribution
- New `InverseCDF` trait for distributions that implement the inverse cdf function

v0.6.0

- `gamma::gamma_ur`, `gamma::gamma_ui`, `gamma::gamma_lr`, and `gamma::gamma_li` now follow strict gamma function domain, panicking if `a` or `x` are not in `(0, +inf)`
- `beta::beta_reg` no longer allows `0.0` for `a` or `b` arguments
- `InverseGamma` distribution no longer accepts `f64::INFINITY` as valid arguments for `shape` or `rate` as the value is nonsense
- `Binomial::cdf` no longer accepts arguments outside the domain of `[0, n]`
- `Bernoulli::cdf` no longer accepts arguments outside the domain of `[0, 1]`
- `DiscreteUniform::cdf` no longer accepts arguments outside the domain of `[min, max]`
- `Uniform::cdf` no longer accepts arguments outside the domain of `[min, max]`
- `Triangular::cdf` no longer accepts arguments outside the domain of `[min, max]`
- `FisherSnedecor` no longer accepts `f64::INFINITY` as a valid argument for `freedom_1` or `freedom_2`
- `FisherSnedecor::cdf` no longer accepts arguments outside the domain of `[0, +inf)`
- `Geometric::cdf` no longer accepts non-positive arguments
- `Normal` now uses the Ziggurat method to generate random samples. This also affects all distributions depending on `Normal` for sampling
  including `Chi`, `LogNormal`, `Gamma`, and `StudentsT`
- `Exponential` now uses the Ziggurat methd to generate random samples.
- `Binomial` now implements `Univariate<u64, f64>` rather than `Univariate<i64, f64>`, meaning `Binomial::min` and `Binomial::max` now return `u64`
- `Bernoulli` now implements `Univariate<u64, f64>` rather than `Univariate<i64, f64>`, meaning `Bernoulli::min` and `Bernoulli::min` now return `u64`
- `Geometric` now implements `Univariate<u64, f64>` rather than `Univariate<i64, f64>`, meaning `Geometric::min` and `Geometric::min` now return `u64`
- `Poisson` now implements `Univariate<u64, f64>` rather than `Univariate<i64, f64>`, meaning `Poisson::min` and `Poisson::min` now return `u64`
- `Binomial` now implements `Mode<u64>` instead of `Mode<i64>`
- `Bernoulli` now implements `Mode<u64>` instead of `Mode<i64>`
- `Poisson` now implements `Mode<u64>` instead of `Mode<i64>`
- `Geometric` now implements `Mode<u64>` instead of `Mode<i64>`
- `Hypergeometric` now implements `Mode<u64>` instead of `Mode<i64>`
- `Binomial` now implements `Discrete<u64, f64>` rather than `Discrete<i64, f64>`
- `Bernoulli` now implements `Discrete<u64, f64>` rather than `Discrete<i64, f64>`
- `Geometric` now implements `Discrete<u64, f64>` rather than `Discrete<i64, f64>`
- `Hypergeometric` now implements `Discrete<u64, f64>` rather than `Discrete<i64, f64>`
- `Poisson` now implements `Discrete<u64, f64>` rather than `Discrete<i64, f64>`

v0.5.1

- Fixed critical bug in `normal::sample_unchecked` where it was returning `NaN`

v0.5.0

- Implemented the `logistic::logistic` special function
- Implemented the `logistic::logit` special function
- Implemented the `factorial::multinomial` special function
- Implemented the `harmonic::harmonic` special function
- Implemented the `harmonic::gen_harmonic` special function
- Implemented the `InverseGamma` distribution
- Implemented the `Geometric` distribution
- Implemented the `Hypergeometric` distribution
- `gamma::gamma_ur` now panics when `x > 0` or `a == f64::NEG_INFINITY`. In addition, it also returns `f64::NAN` when `a == f64::INFINITY` and `0.0` when `x == f64::INFINITY`
- `Gamma::pdf` and `Gamma::ln_pdf` now return `f64::NAN` if any of `shape`, `rate`, or `x` are `f64::INFINITY`
- `Binomial::pdf` and `Binomial::ln_pdf` now panic if `x > n` or `x < 0`
- `Bernoulli::pdf` and `Bernoulli::ln_pdf` now panic if `x > 1` or `x < 0`

v0.4.0

- Implemented the `exponential::integral` special function
- Implemented the `Cauchy` (otherwise known as the `Lorenz`) distribution
- Implemented the `Dirichlet` distribution
- `Continuous` and `Discrete` traits no longer dependent on `Distribution` trait

v0.3.2

- Implemented the `FisherSnedecor` (F) distribution

v0.3.1

- Removed print statements from `ln_pdf` method in `Beta` distribution

v0.3.0

- Moved methods `min` and `max` out of trait `Univariate` into their own respective traits `Min` and `Max`
- Traits `Min`, `Max`, `Mean`, `Variance`, `Entropy`, `Skewness`, `Median`, and `Mode` moved from `distribution` module to `statistics` module
- `Mean`, `Variance`, `Entropy`, `Skewness`, `Median`, and `Mode` no longer depend on `Distribution` trait
- `Mean`, `Variance`, `Skewness`, and `Mode` are now generic over only one type, the return type, due to not depending on `Distribution` anymore
- `order_statistic`, `median`, `quantile`, `percentile`, `lower_quartile`, `upper_quartile`, `interquartile_range`, and `ranks` methods removed
  from `Statistics` trait.
- `min`, `max`, `mean`, `variance`, and `std_dev` methods added to `Statistics` trait
- `Statistics` trait now implemented for all types implementing `IntoIterator` where `Item` implements `Borrow<f64>`. Slice now implicitly implements
  `Statistics` through this new implementation.
- Slice still implements `Min`, `Max`, `Mean`, and `Variance` but now through the `Statistics` implementation rather than its own implementation
- `InplaceStatistics` renamed to `OrderStatistics`, all methods in `InplaceStatistics` have `_inplace` trimmed from method name.
- Inverse DiGamma function implemented with signature `gamma::inv_digamma(x: f64) -> f64`

v0.2.0

- Created `statistics` module and `Statistics` trait
- `Statistics` trait implementation for `[f64]`
- Implemented `Beta` distribution
- Added `Modulus` trait and implementations for `f32`, `f64`, `i32`, `i64`, `u32`, and `u64` in `euclid` module
- Added periodic and sinusoidal vector generation functions in `generate` module
