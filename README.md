# statrs

![tests][actions-test-badge]
[![MIT licensed][license-badge]](./LICENSE.md)
[![Crate][crates-badge]][crates-url]
[![docs.rs][docsrs-badge]][docs-url]
[![codecov-statrs][codecov-badge]][codecov-url]
![Crates.io MSRV][crates-msrv-badge]

[actions-test-badge]: https://github.com/statrs-dev/statrs/actions/workflows/test.yml/badge.svg
[crates-badge]: https://img.shields.io/crates/v/statrs.svg
[crates-url]: https://crates.io/crates/statrs
[license-badge]: https://img.shields.io/badge/license-MIT-blue.svg
[docsrs-badge]: https://img.shields.io/docsrs/statrs
[docs-url]: https://docs.rs/statrs/*/statrs
[codecov-badge]: https://codecov.io/gh/statrs-dev/statrs/graph/badge.svg?token=XtMSMYXvIf
[codecov-url]: https://codecov.io/gh/statrs-dev/statrs
[crates-msrv-badge]: https://img.shields.io/crates/msrv/statrs

Statrs provides a host of statistical utilities for Rust scientific computing.

Included are a number of common distributions that can be sampled (i.e. Normal, Exponential, Student's T, Gamma, Uniform, etc.) plus common statistical functions like the gamma function, beta function, and error function.

This library began as port of the statistical capabilities in the C# Math.NET library.
All unit tests in the library borrowed from Math.NET when possible and filled-in when not.
Planned for future releases are continued implementations of distributions as well as porting over more statistical utilities.

Please check out the documentation [here][docs-url].

## Usage

Add the most recent release to your `Cargo.toml`

```toml
[dependencies]
statrs = "*" # replace * by the latest version of the crate.
```

For examples, view [the docs](https://docs.rs/statrs/*/statrs/).

### Running tests

If you'd like to run all suggested tests, you'll need to download some data from
NIST, we have a script for this and formatting the data in the `tests/` folder.

```sh
cargo test
./tests/gather_nist_data.sh && cargo test -- --include-ignored nist_
```

If you'd like to modify where the data is downloaded, you can use the environment variable,
`STATRS_NIST_DATA_DIR` for running the script and the tests.

## Minimum supported Rust version (MSRV)

This crate requires a Rust version of 1.65.0 or higher. Increases in MSRV will be considered a semver non-breaking API change and require a version increase (PATCH until 1.0.0, MINOR after 1.0.0).

## Contributing

Thanks for your help to improve the project!
**No contribution is too small and all contributions are valued.**

Suggestions if you don't know where to start,
- if you're an existing user, file an issue or feature request.
- [documentation][docs-url] is a great place to start, as you'll be able to identify the value of existing documentation better than its authors.
- tests are valuable in demonstrating correct behavior, you can review test coverage on the [CodeCov Report][codecov-url]
- check out some of the issues marked [help wanted](https://github.com/statrs-dev/statrs/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22).
- look at features you'd like to see in statrs
  - Math.NET
    - [Distributions](https://github.com/mathnet/mathnet-numerics/tree/master/src/Numerics/Distributions)
    - [Statistics](https://github.com/mathnet/mathnet-numerics/tree/master/src/Numerics/Statistics)
  - scipy.stats
  - KDE, see (issue #193)[https://github.com/statrs-dev/statrs/issues/193]

### How to contribute

Clone the repo:

```sh
git clone https://github.com/statrs-dev/statrs
```

Create a feature branch:

```sh
git checkout -b <feature_branch> master
```

Write your code and docs, then ensure it is formatted:

```sh
cargo fmt
```

Add `--check` to view the diff without making file changes.
Our CI will check format without making changes.

After commiting your code:

```shell
git push -u <your_remote_name> <your_branch> # with `git`
gh pr create --head <your_branch> # with GitHub's cli
```

Then submit a PR, preferably referencing the relevant issue, if it exists.

### Precision for contributors
Take a look at the (Precision section)[#precision] to get a sense of what we have built in. Also take a look at tests and various assertions.
If you'd like to improve precision, please also make existing tests more precise, perhaps up to the crate's default set in the `crate::prec` module or better.

### Commit messages

Please be explicit and and purposeful with commit messages.
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/#summary) encouraged.

### Communication Expectations

Please allow at least one week before pinging issues/pr's.

## Precision
Floating-point numbers cannot always represent decimal values exactly, which can introduce small (and in some cases catastrophically large) errors in computations.
In statistical applications, these errors can accumulate, making careful precision control important.

To help us consistently inspect for inaccuracy, `statrs` provides a `prec` module that wraps and standardizes common approximation checks from the `approx` crate, but sets default values relevant to the crate.
The crate-wide defaults are in the `prec` module, and in some sense establish expectations across the crate.
One case of module level defaults are in the special function `beta` module.

We encourage contributors to maintain or improve precision in existing tests when making changes.
Our default precision levels aim to balance correctness and performance.
To see where precision is explicited outside the default, you can search for relevant comparisons using:

```sh
rg '(?:abs_diff|relative|ulps)_eq\(.+?(?:epsilon|max_relative|ulps)\s?='
```

or using `grep` as an alternative:

```sh
grep -E '(abs_diff|relative|ulps)_eq' -r .
```

If you're new to floating-point precision, and wish to understand what kind of precision checking to use, these resources provide helpful introductions:

- [Comparing Floating Point Numbers, 2012 Edition](https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/)
- [The Floating Point Guide - Comparison](http://floating-point-gui.de/errors/comparison/)
- [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)
This section on precision was a concerted effort with GPT-4o mini to ensure that it's inclusive to folks within and outside of numerics programming.
