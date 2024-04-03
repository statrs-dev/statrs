# statrs

![tests](https://github.com/statrs-dev/statrs/actions/workflows/test.yml/badge.svg)
[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE.md)
[![Crates.io](https://img.shields.io/crates/v/statrs.svg)](https://crates.io/crates/statrs)

## Current Version: v0.16.0

Should work for both nightly and stable Rust.

**NOTE:** While I will try to maintain backwards compatibility as much as possible, since this is still a 0.x.x project the API is not considered stable and thus subject to possible breaking changes up until v1.0.0

## Description

Statrs provides a host of statistical utilities for Rust scientific computing.
Included are a number of common distributions that can be sampled (i.e. Normal, Exponential, Student's T, Gamma, Uniform, etc.) plus common statistical functions like the gamma function, beta function, and error function.

This library is a work-in-progress port of the statistical capabilities in the C# Math.NET library.
All unit tests in the library borrowed from Math.NET when possible and filled-in when not.

This library is a work-in-progress and not complete.
Planned for future releases are continued implementations of distributions as well as porting over more statistical utilities.

Please check out the documentation [here](https://docs.rs/statrs/*/statrs/).

## Usage

Add the most recent release to your `Cargo.toml`

```Rust
[dependencies]
statrs = "0.16"
```

## Examples

Statrs comes with a number of commonly used distributions including Normal, Gamma, Student's T, Exponential, Weibull, etc.
The common use case is to set up the distributions and sample from them which depends on the `Rand` crate for random number generation.

```Rust
use statrs::distribution::Exp;
use rand::distributions::Distribution;

let mut r = rand::rngs::OsRng;
let n = Exp::new(0.5).unwrap();
print!("{}", n.sample(&mut r));
```

Statrs also comes with a number of useful utility traits for more detailed introspection of distributions.

```Rust
use statrs::distribution::{Exp, Continuous, ContinuousCDF};
use statrs::statistics::Distribution;

let n = Exp::new(1.0).unwrap();
assert_eq!(n.mean(), Some(1.0));
assert_eq!(n.variance(), Some(1.0));
assert_eq!(n.entropy(), Some(1.0));
assert_eq!(n.skewness(), Some(2.0));
assert_eq!(n.cdf(1.0), 0.6321205588285576784045);
assert_eq!(n.pdf(1.0), 0.3678794411714423215955);
```

as well as utility functions including `erf`, `gamma`, `ln_gamma`, `beta`, etc.

```Rust
use statrs::statistics::Distribution;
use statrs::distribution::FisherSnedecor;

let n = FisherSnedecor::new(1.0, 1.0).unwrap();
assert!(n.variance().is_none());
```

## Contributing

Want to contribute?
Check out some of the issues marked [help wanted](https://github.com/statrs-dev/statrs/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)

### How to contribute

Clone the repo:

```
git clone https://github.com/statrs-dev/statrs
```

Create a feature branch:

```
git checkout -b <feature_branch> master
```

Write your code and docs, then ensure it is formatted:

The below sample modify in-place, use `--check` flag to view diff without making file changes.
Not using `fmt` from +nightly may result in some warnings and different formatting.
Our CI will `fmt`, but less chores in commit history are appreciated.

```
cargo +nightly fmt
```

After commiting your code:

```
git push -u origin <feature_branch>
```

Then submit a PR, preferably referencing the relevant issue, if it exists.

### Commit messages

Please be explicit and and purposeful with commit messages.
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/#summary) encouraged.

#### Bad

```
Modify test code
```

#### Good

```
test: Update statrs::distribution::Normal test_cdf
```

### Communication Expectations

Please allow at least one week before pinging issues/pr's.

