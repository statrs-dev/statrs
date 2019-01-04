//! Provides testing helpers and utilities

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::str;

/// Loads a test data file into a vector of `f64`'s.
/// Path is relative to /data.
///
/// # Panics
///
/// Panics if the file does not exist or could not be opened, or
/// there was an error reading the file.
pub fn load_data(path: &str) -> Vec<f64> {
    // note: the copious use of unwrap is because this is a test helper and
    // if reading the data file fails, we want to panic immediately

    let path_prefix = "./data/".to_string();
    let true_path = path_prefix + path.trim().trim_start_matches('/');

    let f = File::open(true_path).unwrap();
    let mut reader = BufReader::new(f);

    let mut buf = String::new();
    let mut data: Vec<f64> = vec![];
    while reader.read_line(&mut buf).unwrap() > 0 {
        data.push(buf.trim().parse::<f64>().unwrap());
        buf.clear();
    }
    data
}
