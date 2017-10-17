# statrs  
  
[![Build Status](https://travis-ci.org/boxtown/statrs.svg?branch=master)](https://travis-ci.org/boxtown/statrs)
[![Codecov](https://codecov.io/gh/boxtown/statrs/branch/master/graph/badge.svg)](https://codecov.io/gh/boxtown/statrs)
[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE.md)
[![Crates.io](https://img.shields.io/crates/v/statrs.svg?maxAge=2592000)](https://crates.io/crates/statrs)  

## Current Version: v0.8.0

Should work for both nightly and stable Rust.

**NOTE:** While I will try to maintain backwards compatibility as much as possible, since this is still a 0.x.x project the API is not considered stable and thus subject to possible breaking changes up until v1.0.0

## Description
  
Statrs provides a host of statistical utilities for Rust scientific computing.
Included are a number of common distributions that can be sampled (i.e. Normal, Exponential,
Student's T, Gamma, Uniform, etc.) plus common statistical functions like the gamma function,
beta function, and error function.  
  
This library is a work-in-progress port of the statistical capabilities
in the C# Math.NET library. All unit tests in the library borrowed from Math.NET when possible
and filled-in when not.  
  
This library is a work-in-progress and not complete. Planned for future releases are continued implementations
of distributions as well as porting over more statistical utilities

Please check out the documentation [here](https://docs.rs/statrs/0.8.0/statrs/)

## Usage

Add the following to your `Cargo.toml`

```Rust
[dependencies]
statrs = "0.8.0"
```

and this to your crate root

```Rust
extern crate statrs;
```
  
## Examples

Statrs v0.8.0 comes with a number of commonly used distributions including Normal, Gamma, Student's T, Exponential, Weibull, etc.
The common use case is to set up the distributions and sample from them which depends on the `Rand` crate for random number generation

```Rust
use rand;
use statrs::distribution::{Exponential, Distribution};

let mut r = rand::StdRng::new().unwrap();
let n = Exponential::new(0.5).unwrap();
print!("{}", n.Sample::<StdRng>(&mut r);
```

Statrs also comes with a number of useful utility traits for more detailed introspection of distributions

```Rust
use statrs::distribution::{Exponential, Univariate, Continuous};
use statrs::statistics::{Mean, Variance, Entropy, Skewness};

let n = Exponential::new(1.0).unwrap();
assert_eq!(n.mean(), 1.0);
assert_eq!(n.variance(), 1.0);
assert_eq!(n.entropy(), 1.0);
assert_eq!(n.skewness(), 2.0);
assert_eq!(n.cdf(1.0), 0.6321205588285576784045);
assert_eq!(n.pdf(1.0), 0.3678794411714423215955);
```

as well as utility functions including `erf`, `gamma`, `ln_gamma`, `beta`, etc

## Contributing

Want to contribute? Check out some of the issues marked [help wanted](https://github.com/boxtown/statrs/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)

### How to contribute

Clone the repo:

```
git clone https://github.com/boxtown/starts
```

Create a feature branch:

```
git checkout -b <feature_branch> master
```

After commiting your code:

```
git push -u origin <feature_branch>
```

Then submit a PR, preferably referencing the relevant issue.

### Style

This repo makes use of `rustfmt` with the configuration specified in `rustfmt.toml`.
See https://github.com/rust-lang-nursery/rustfmt for instructions on installation
and usage and run the formatter using `rustfmt --write-mode overwrite *.rs` in
the `src` directory before committing.

### Commit messages

Please be explicit and and purposeful with commit messages.

#### Bad
```
Modify test code
```
#### Good
```
test: Update statrs::distribution::Normal test_cdf
```
