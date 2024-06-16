use std::error::Error;
use std::fmt;
use std::ops::Bound;

/// Enumeration of possible errors thrown within the `statrs` library
#[derive(Clone, PartialEq, Debug)]
pub enum StatsError {
    /// Generic bad input parameter error
    BadParams,
    /// value must not be NAN
    NotNan,
    /// value must be finite and must not be NAN
    Finite(f64),
    /// value must be finite, non negative and must not be NAN
    FiniteNonNegative(f64),
    /// value must be within specified bounds
    Bounded((Bound<f64>, Bound<f64>), f64),
    /// first value must be within bounds defined by second value
    ParametrizedBounded((Bound<f64>, Bound<f64>), f64, f64),
    /// Expected one iterator to not exhaust before another
    IteratorExhaustedEarly,
    /// Containers of the same length were expected
    ContainersMustBeSameLength(usize),
    /// Computation failed to converge,
    FailedConvergence(f64, f64),
    /// Elements in a container were expected to sum to a value but didn't
    ContainerExpectedSum(f64, f64),
    /// Elements in a container were expected to sum to a variable but didn't
    ContainerExpectedSumVar(f64, f64),
    /// Special case exception
    SpecialCase(&'static str),
}

impl Error for StatsError {}

impl fmt::Display for StatsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            StatsError::BadParams => write!(f, "Bad parameters, unspecified"),
            StatsError::NotNan => write!(f, "value must not be NAN"),
            StatsError::Finite(x) => write!(f, "given `{}`, but must be finite and not NAN", x),
            StatsError::FiniteNonNegative(x) => write!(f, "given `{}`, but must be finite, non-negative and not NAN", x),
            StatsError::Bounded(bound, x) => {
                write!(f, "given `{}`, but must be on interval {:?}", x, bound)
            }
            StatsError::ParametrizedBounded(bound, x, y) => write!(
                f,
                "given `{}`, but another value {} requires it be on {:?}",
                x, y, bound
            ),
            StatsError::ContainersMustBeSameLength(size) => write!(
                f,
                "Expected containers of same length, found only one of size `{}`",
                size
            ),
            StatsError::FailedConvergence(x,prec) => write!(f, "Computation failed to converge, last iteration reached `{}` but stepped relative prec `{}`", x, prec),
            StatsError::IteratorExhaustedEarly => write!(f, "Iterator exhausted earlier than expected"),
            StatsError::ContainerExpectedSum(s, sum) => {
                write!(f, "sum found to be {}, expected {}", s, sum)
            }
            StatsError::ContainerExpectedSumVar(s, sum) => {
                write!(f, "sum found to be {}, but other value specifies should be {}", s, sum)
            }
            StatsError::SpecialCase(s) => write!(f, "{}", s),
        }
    }
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
