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

## Precision
Floating-point numbers cannot always represent decimal values exactly, which can introduce small (and in some cases catastrophically large) errors in computations.
In statistical applications, these errors can accumulate, making careful precision control important.

### For Users and Evaluators

The `statrs` crate takes precision seriously:

- We use standardized precision checks throughout the codebase
- Default precision levels are carefully chosen to balance correctness and performance
- Module-specific precision requirements are explicitly documented where they differ from defaults
- Our test suite verifies numerical accuracy against common reference libraries

Key precision constants in the crate are set by pub consts in the `prec` module:
- Default relative accuracy: `pub const DEFAULT_RELATIVE_ACC`
- Default epsilon: `pub const DEFAULT_EPS`
- Default ULPs (Units in Last Place): `pub const DEFAULT_ULPS`

Some modules/submodules have default precision that is different from the crate defaults, for searchability the names of such constants are the `MODULE_RELATIVE_ACC`, `MODULE_EPS`, and `MODULE_ULPS`.

> [!IMPORTANT]
> Starting from v0.19.0, the `prec` module is no longer public (`pub mod prec` → `mod prec`). This change reflects that precision handling is an internal implementation detail.
>
> The precision constants mentioned above remain stable and documented and will be reexported at the crate level, but direct access to the module's utilities is now restricted to maintain better API boundaries.


### For Contributors
// express your sentiment about the intended use of `prec` module in this section. The reason is that this section is for contributors and the users need not know about internal functionality.
To help maintain consistent precision checking, `statrs` provides:

1. A `prec` module that wraps and standardizes common approximation checks from the `approx` crate with crate-specific defaults
2. Macros for common precision comparison patterns
3. Helper functions for convergence testing

When contributing:
- Use the provided precision utilities rather than hard-coding values
- Maintain or improve precision in existing tests when making changes, new modules can start at lesser precision than the crate defaults if need be
  - when doing so, one should use the same names as defined in the `prec` module, this helps with searchabiliy.
- Document any module-specific precision requirements

### Learning Resources

If you're new to floating-point precision, these resources provide helpful introductions:

- [Comparing Floating Point Numbers, 2012 Edition](https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/)
- [The Floating Point Guide - Comparison](http://floating-point-gui.de/errors/comparison/)
- [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)

## Contributing

Thanks for your help to improve the project!
**No contribution is too small and all contributions are valued.**

If you're not familiar with precision in floating point operations, please read the section on [precision](#precision) specifically, the [For Contributors](#for-contributors) section.

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

### Commit messages

Please be explicit and and purposeful with commit messages.
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/#summary) encouraged.

### Communication Expectations

Please allow at least one week before pinging issues/pr's.
