pub mod f_oneway;
pub mod fisher;

/// Specifies an [alternative hypothesis](https://en.wikipedia.org/wiki/Alternative_hypothesis)
#[derive(Debug, Copy, Clone)]
pub enum Alternative {
    #[doc(alias = "two-tailed")]
    #[doc(alias = "two tailed")]
    TwoSided,
    #[doc(alias = "one-tailed")]
    #[doc(alias = "one tailed")]
    Less,
    #[doc(alias = "one-tailed")]
    #[doc(alias = "one tailed")]
    Greater,
}

/// Specifies how to deal with NaNs provided in input data
/// based on scipy treatment
#[derive(Debug, Copy, Clone)]
pub enum NaNPolicy {
    /// allow for NaNs; if exist fcuntion will return NaN
    Propogate,
    /// filter out the NaNs before calculations
    Emit,
    /// if NaNs are in the input data, return an Error
    Error,
}

pub use fisher::{fishers_exact, fishers_exact_with_odds_ratio};
