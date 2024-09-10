//! This test relies on data that is reusable but not distributable by statrs as
//! such, the data will need to be downloaded from the relevant NIST StRD dataset
//! the parsing for testing assumes data to be of form,
//! ```text
//!     sample mean       : <possibly signed float for mean>
//!     sample std_dev    : <possibly signed float for standard deviation>
//!     sample correlation: <possibly signed float for correlation coefficient>
//!     [zero or more blank lines]
//!     data0
//!     data1
//!     data2
//!     ...
//! ```
//! This test can be run on it's own from the shell from this folder as
//! ```sh
//!     ./gather_nist_data.sh && cargo test -- --ignored nist_
//! ```
use anyhow::Result;
use approx::assert_relative_eq;
use statrs::statistics::Statistics;

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::{env, fs};

struct TestCase {
    certified: CertifiedValues,
    values: Vec<f64>,
}

impl std::fmt::Debug for TestCase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TestCase({:?}, [...]", self.certified)
    }
}

#[derive(Debug)]
struct CertifiedValues {
    mean: f64,
    std_dev: f64,
    corr: f64,
}

impl std::fmt::Display for CertifiedValues {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "μ={:.3e}, σ={:.3e}, r={:.3e}",
            self.mean, self.std_dev, self.corr
        )
    }
}

const NIST_DATA_DIR_ENV: &str = "STATRS_NIST_DATA_DIR";
const FILENAMES: [&str; 7] = [
    "Lottery.dat",
    "Lew.dat",
    "Mavro.dat",
    "Michelso.dat",
    "NumAcc1.dat",
    "NumAcc2.dat",
    "NumAcc3.dat",
];

fn get_path(fname: &str, prefix: Option<&str>) -> PathBuf {
    if let Some(prefix) = prefix {
        [prefix, fname].iter().collect()
    } else {
        ["tests", fname].iter().collect()
    }
}

#[test]
#[ignore = "NIST tests should not run from typical `cargo test` calls"]
fn nist_strd_univariate_mean() {
    for fname in FILENAMES {
        let filepath = get_path(fname, env::var(NIST_DATA_DIR_ENV).ok().as_deref());
        let case = parse_file(filepath)
            .unwrap_or_else(|e| panic!("failed parsing file {fname} with `{e:?}`"));
        assert_relative_eq!(case.values.mean(), case.certified.mean, epsilon = 1e-12);
    }
}

#[test]
#[ignore]
fn nist_strd_univariate_std_dev() {
    for fname in FILENAMES {
        let filepath = get_path(fname, env::var(NIST_DATA_DIR_ENV).ok().as_deref());
        let case = parse_file(filepath)
            .unwrap_or_else(|e| panic!("failed parsing file {fname} with `{e:?}`"));
        assert_relative_eq!(
            case.values.std_dev(),
            case.certified.std_dev,
            epsilon = 1e-10
        );
    }
}

fn parse_certified_value(line: String) -> Result<f64> {
    line.chars()
        .skip_while(|&c| c != ':')
        .skip(1) // skip through ':' delimiter
        .skip_while(|&c| c.is_whitespace()) // effectively `String` trim
        .take_while(|&c| matches!(c, '0'..='9' | '-' | '.'))
        .collect::<String>()
        .parse::<f64>()
        .map_err(|e| e.into())
}

fn parse_file(path: impl AsRef<std::path::Path>) -> anyhow::Result<TestCase> {
    let f = fs::File::open(path)?;
    let reader = BufReader::new(f);
    let mut lines = reader.lines();

    let mean = parse_certified_value(lines.next().expect("file should not be exhausted")?)?;
    let std_dev = parse_certified_value(lines.next().expect("file should not be exhausted")?)?;
    let corr = parse_certified_value(lines.next().expect("file should not be exhausted")?)?;

    Ok(TestCase {
        certified: CertifiedValues {
            mean,
            std_dev,
            corr,
        },
        values: lines
            .map_while(|line| line.ok()?.trim().parse().ok())
            .collect(),
    })
}

#[test]
#[ignore = "NIST tests should not run from typical `cargo test` calls"]
fn nist_test_covariance_consistent_with_variance() {}

#[test]
#[ignore = "NIST tests should not run from typical `cargo test` calls"]
fn nist_test_covariance_is_symmetric() {}
