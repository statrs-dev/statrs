pub mod fisher;

#[derive(Debug, Copy, Clone)]
pub enum Alternative {
    TwoSided,
    Less,
    Greater,
}

pub use fisher::{fishers_exact, fishers_exact_with_odds_ratio};
