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

#[test]
#[ignore = "NIST tests should not run from typical `cargo test` calls"]
fn nist_strd_univariate_mean() {
    let path_prefix = env::var(NIST_DATA_DIR_ENV).unwrap_or_else(|e| panic!("{}", e));

    for fname in FILENAMES {
        let case = parse_file([&path_prefix, fname].iter().collect::<PathBuf>())
            .unwrap_or_else(|e| panic!("failed parsing file {} with {:?}", fname, e));
        assert_relative_eq!(
            case.values.iter().mean(),
            case.certified.mean,
            epsilon = 1e-12
        );
    }
}

#[test]
#[ignore]
fn nist_strd_univariate_std_dev() {
    let path_prefix = env::var(NIST_DATA_DIR_ENV).unwrap_or_else(|e| panic!("{}", e));

    for fname in FILENAMES {
        let case = parse_file([&path_prefix, fname].iter().collect::<PathBuf>())
            .unwrap_or_else(|e| panic!("failed parsing file {} with {:?}", fname, e));
        assert_relative_eq!(
            case.values.iter().std_dev(),
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
