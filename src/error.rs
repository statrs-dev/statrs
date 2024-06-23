use std::error::Error;
use std::fmt;
use std::ops::Bound;

/// Enumeration of possible errors thrown within the `statrs` library
#[derive(Copy, Clone, PartialEq, Debug, thiserror::Error)]
pub enum StatsError {
    #[error("Bad parameters, unspecified")]
    BadParams,
    #[error("value must not be NAN")]
    NotNan,
    #[error("given `{}`, but must be finite and not NAN", .0)]
    Finite(f64),
    #[error("given `{}`, but must be finite, non-negative and not NAN", .0)]
    FiniteNonNegative(f64),
    #[error("given `{}`, but must be on interval {:?}", .1, .0)]
    Bounded((Bound<f64>, Bound<f64>), f64),
    #[error("given `{}`, but another value {} requires it be on {:?}", .1, .2, .0)]
    ParametrizedBounded((Bound<f64>, Bound<f64>), f64, f64),
    #[error("Iterator exhausted earlier than expected")]
    IteratorExhaustedEarly,
    #[error("Expected containers of same length, found one len=`{}`", .0)]
    ContainersMustBeSameLength(usize),
    #[error("Computation failed to converge, last iteration reached `{}` but stepped relative prec `{}`", .0, .1)]
    FailedConvergence(f64, f64),
    #[error("sum found to be {}, expected {}", .0, .1)]
    ContainerExpectedSum(f64, f64),
    #[error("sum found to be {}, but other value specifies should be {}", .0, .1)]
    ContainerExpectedSumVar(f64, f64),
    #[error("{}", .0)]
    SpecialCase(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_sync<T: Sync>() {}
    fn assert_send<T: Send>() {}

    #[test]
    fn test_sync_send() {
        // Error types should implement Sync and Send
        assert_sync::<StatsError>();
        assert_send::<StatsError>();
    }
}
