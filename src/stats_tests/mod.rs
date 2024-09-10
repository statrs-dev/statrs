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

pub use fisher::{fishers_exact, fishers_exact_with_odds_ratio};
