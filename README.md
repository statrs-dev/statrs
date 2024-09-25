# statrs

![tests][actions-test-badge]
[![MIT licensed][license-badge]](./LICENSE.md)
[![Crate][crates-badge]][crates-url]
[![docs.rs](https://img.shields.io/docsrs/statrs)][docs-url]

[actions-test-badge]: https://github.com/statrs-dev/statrs/actions/workflows/test.yml/badge.svg
[crates-badge]: https://img.shields.io/crates/v/statrs.svg
[crates-url]: https://crates.io/crates/statrs
[license-badge]: https://img.shields.io/badge/license-MIT-blue.svg
[docsrs-badge]: https://img.shields.io/docsrs/statrs
[docs-url]: https://docs.rs/statrs/*/statrs
[codecov-badge]: https://codecov.io/gh/statrs-dev/statrs/graph/badge.svg?token=XtMSMYXvIf
[codecov-url]: https://codecov.io/gh/statrs-dev/statrs

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
- documentation is a great place to start, as you'll be able to identify the value of existing documentation better than its authors.
- tests are valuable in demonstrating correct behavior, you can review test coverage on the [CodeCov Report][codecov-url]*, not live until [#229](https://github.com/statrs-dev/statrs/pull/229) merged.
- check out some of the issues marked [help wanted](https://github.com/statrs-dev/statrs/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22).
- look at what's not included from Math.NET's [Distributions](https://github.com/mathnet/mathnet-numerics/tree/master/src/Numerics/Distributions), [Statistics](https://github.com/mathnet/mathnet-numerics/tree/master/src/Numerics/Statistics), or related.

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

```
cargo fmt
```

Add `--check` to view the diff without making file changes.
Our CI will `fmt`, but less chores in commit history are appreciated.

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

