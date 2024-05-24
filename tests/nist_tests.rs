// #![cfg(test)]
use statrs::assert_almost_eq;
use statrs::statistics::Statistics;
use std::io::{BufRead, BufReader};
use std::{env, fs};

#[cfg(test)]
const NIST_DATA_DIR_ENV: &str = "STATRS_NIST_DATA_DIR";

fn load_data(pathname: String) -> Vec<f64> {
    let f = fs::File::open(pathname).unwrap();
    let mut reader = BufReader::new(f);

    let mut buf = String::new();
    let mut data: Vec<f64> = vec![];
    while reader.read_line(&mut buf).unwrap() > 0 {
        data.push(buf.trim().parse::<f64>().unwrap());
        buf.clear();
    }
    data
}

#[test]
#[ignore = "NIST tests should not run from typical `cargo test` calls"]
fn nist_test_mean() {
    let path_dir = env::var(NIST_DATA_DIR_ENV).unwrap();
    let mut data = load_data(dbg!(path_dir.clone() + "lottery.txt"));
    assert_almost_eq!((&data).mean(), 518.958715596330, 1e-12);

    data = load_data(dbg!(path_dir.clone() + "lew.txt"));
    assert_almost_eq!((&data).mean(), -177.435000000000, 1e-13);

    data = load_data(dbg!(path_dir.clone() + "mavro.txt"));
    assert_almost_eq!((&data).mean(), 2.00185600000000, 1e-15);

    data = load_data(dbg!(path_dir.clone() + "michaelso.txt"));
    assert_almost_eq!((&data).mean(), 299.852400000000, 1e-13);

    data = load_data(dbg!(path_dir.clone() + "numacc1.txt"));
    assert_eq!((&data).mean(), 10000002.0);

    data = load_data(dbg!(path_dir.clone() + "numacc2.txt"));
    assert_almost_eq!((&data).mean(), 1.2, 1e-15);

    data = load_data(dbg!(path_dir.clone() + "numacc3.txt"));
    assert_eq!((&data).mean(), 1000000.2);

    data = load_data(dbg!(path_dir.clone() + "numacc4.txt"));
    assert_almost_eq!((&data).mean(), 10000000.2, 1e-8);
}

#[test]
#[ignore = "NIST tests should not run from typical `cargo test` calls"]
fn nist_test_std_dev() {
    let path_dir = env::var(NIST_DATA_DIR_ENV).unwrap();
    let mut data = load_data(dbg!(path_dir.clone() + "lottery.txt"));
    assert_almost_eq!((&data).std_dev(), 291.699727470969, 1e-13);

    data = load_data(dbg!(path_dir.clone() + "lew.txt"));
    assert_almost_eq!((&data).std_dev(), 277.332168044316, 1e-12);

    data = load_data(dbg!(path_dir.clone() + "mavro.txt"));
    assert_almost_eq!((&data).std_dev(), 0.000429123454003053, 1e-15);

    data = load_data(dbg!(path_dir.clone() + "michaelso.txt"));
    assert_almost_eq!((&data).std_dev(), 0.0790105478190518, 1e-13);

    data = load_data(dbg!(path_dir.clone() + "numacc1.txt"));
    assert_eq!((&data).std_dev(), 1.0);

    data = load_data(dbg!(path_dir.clone() + "numacc2.txt"));
    assert_almost_eq!((&data).std_dev(), 0.1, 1e-16);

    data = load_data(dbg!(path_dir.clone() + "numacc3.txt"));
    assert_almost_eq!((&data).std_dev(), 0.1, 1e-10);

    data = load_data(dbg!(path_dir.clone() + "numacc4.txt"));
    assert_almost_eq!((&data).std_dev(), 0.1, 1e-9);
}

#[test]
#[ignore = "NIST tests should not run from typical `cargo test` calls"]
fn nist_test_covariance_consistent_with_variance() {
    let path_dir = env::var(NIST_DATA_DIR_ENV).unwrap();
    let mut data = load_data(dbg!(path_dir.clone() + "lottery.txt"));
    assert_almost_eq!((&data).variance(), (&data).covariance(&data), 1e-10);

    data = load_data(dbg!(path_dir.clone() + "lew.txt"));
    assert_almost_eq!((&data).variance(), (&data).covariance(&data), 1e-10);

    data = load_data(dbg!(path_dir.clone() + "mavro.txt"));
    assert_almost_eq!((&data).variance(), (&data).covariance(&data), 1e-10);

    data = load_data(dbg!(path_dir.clone() + "michaelso.txt"));
    assert_almost_eq!((&data).variance(), (&data).covariance(&data), 1e-10);

    data = load_data(dbg!(path_dir.clone() + "numacc1.txt"));
    assert_almost_eq!((&data).variance(), (&data).covariance(&data), 1e-10);
}

#[test]
#[ignore = "NIST tests should not run from typical `cargo test` calls"]
fn nist_test_pop_covar_consistent_with_pop_var() {
    let path_dir = env::var(NIST_DATA_DIR_ENV).unwrap();
    let mut data = load_data(dbg!(path_dir.clone() + "lottery.txt"));
    assert_almost_eq!(
        (&data).population_variance(),
        (&data).population_covariance(&data),
        1e-10,
    );

    data = load_data(dbg!(path_dir.clone() + "lew.txt"));
    assert_almost_eq!(
        (&data).population_variance(),
        (&data).population_covariance(&data),
        1e-10,
    );

    data = load_data(dbg!(path_dir.clone() + "mavro.txt"));
    assert_almost_eq!(
        (&data).population_variance(),
        (&data).population_covariance(&data),
        1e-10,
    );

    data = load_data(dbg!(path_dir.clone() + "michaelso.txt"));
    assert_almost_eq!(
        (&data).population_variance(),
        (&data).population_covariance(&data),
        1e-10,
    );

    data = load_data(dbg!(path_dir.clone() + "numacc1.txt"));
    assert_almost_eq!(
        (&data).population_variance(),
        (&data).population_covariance(&data),
        1e-10,
    );
}

#[test]
#[ignore = "NIST tests should not run from typical `cargo test` calls"]
fn nist_test_covariance_is_symmetric() {
    let path_dir = env::var(NIST_DATA_DIR_ENV).unwrap();
    let data_a = &load_data(dbg!(path_dir.clone() + "lottery.txt"))[0..200];
    let data_b = &load_data(dbg!(path_dir.clone() + "lew.txt"))[0..200];
    assert_almost_eq!(data_a.covariance(data_b), data_b.covariance(data_a), 1e-10);
    assert_almost_eq!(
        data_a.population_covariance(data_b),
        data_b.population_covariance(data_a),
        1e-11,
    );
}
