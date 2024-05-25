use std::error::Error;
use std::fmt;

/// Enumeration of possible errors thrown within the `statrs` library
#[derive(Debug)]
pub enum StatsError {
    /// Generic bad input parameter error
    BadParams,
    /// An argument must be finite
    ArgFinite(&'static str),
    /// An argument should have been positive and was not
    ArgMustBePositive(&'static str),
    /// An argument should have been non-negative and was not
    ArgNotNegative(&'static str),
    /// An argument should have fallen between an inclusive range but didn't
    ArgIntervalIncl(&'static str, f64, f64),
    /// An argument should have fallen between an exclusive range but didn't
    ArgIntervalExcl(&'static str, f64, f64),
    /// An argument should have fallen in a range excluding the min but didn't
    ArgIntervalExclMin(&'static str, f64, f64),
    /// An argument should have falled in a range excluding the max but didn't
    ArgIntervalExclMax(&'static str, f64, f64),
    /// An argument must have been greater than a value but wasn't
    ArgGt(&'static str, f64),
    /// An argument must have been greater than another argument but wasn't
    ArgGtArg(&'static str, &'static str),
    /// An argument must have been greater than or equal to a value but wasn't
    ArgGte(&'static str, f64),
    /// An argument must have been greater than or equal to another argument
    /// but wasn't
    ArgGteArg(&'static str, &'static str),
    /// An argument must have been less than a value but wasn't
    ArgLt(&'static str, f64),
    /// An argument must have been less than another argument but wasn't
    ArgLtArg(&'static str, &'static str),
    /// An argument must have been less than or equal to a value but wasn't
    ArgLte(&'static str, f64),
    /// An argument must have been less than or equal to another argument but
    /// wasn't
    ArgLteArg(&'static str, &'static str),
    /// Containers of the same length were expected
    ContainersMustBeSameLength,
    /// Computation failed to converge,
    ComputationFailedToConverge,
    /// Elements in a container were expected to sum to a value but didn't
    ContainerExpectedSum(&'static str, f64),
    /// Elements in a container were expected to sum to a variable but didn't
    ContainerExpectedSumVar(&'static str, &'static str),
    /// Special case exception
    SpecialCase(&'static str),
}

impl Error for StatsError {
    fn description(&self) -> &str {
        "Error performing statistical calculation"
    }
}

impl fmt::Display for StatsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            StatsError::BadParams => write!(f, "Bad distribution parameters"),
            StatsError::ArgFinite(s) => write!(f, "Argument {} must be finite", s),
            StatsError::ArgMustBePositive(s) => write!(f, "Argument {} must be positive", s),
            StatsError::ArgNotNegative(s) => write!(f, "Argument {} must be non-negative", s),
            StatsError::ArgIntervalIncl(s, min, max) => {
                write!(f, "Argument {} not within interval [{}, {}]", s, min, max)
            }
            StatsError::ArgIntervalExcl(s, min, max) => {
                write!(f, "Argument {} not within interval ({}, {})", s, min, max)
            }
            StatsError::ArgIntervalExclMin(s, min, max) => {
                write!(f, "Argument {} not within interval ({}, {}]", s, min, max)
            }
            StatsError::ArgIntervalExclMax(s, min, max) => {
                write!(f, "Argument {} not within interval [{}, {})", s, min, max)
            }
            StatsError::ArgGt(s, val) => write!(f, "Argument {} must be greater than {}", s, val),
            StatsError::ArgGtArg(s, val) => {
                write!(f, "Argument {} must be greater than {}", s, val)
            }
            StatsError::ArgGte(s, val) => {
                write!(f, "Argument {} must be greater than or equal to {}", s, val)
            }
            StatsError::ArgGteArg(s, val) => {
                write!(f, "Argument {} must be greater than or equal to {}", s, val)
            }
            StatsError::ArgLt(s, val) => write!(f, "Argument {} must be less than {}", s, val),
            StatsError::ArgLtArg(s, val) => write!(f, "Argument {} must be less than {}", s, val),
            StatsError::ArgLte(s, val) => {
                write!(f, "Argument {} must be less than or equal to {}", s, val)
            }
            StatsError::ArgLteArg(s, val) => {
                write!(f, "Argument {} must be less than or equal to {}", s, val)
            }
            StatsError::ContainersMustBeSameLength => {
                write!(f, "Expected containers of same length")
            }
            StatsError::ComputationFailedToConverge => write!(f, "Computation failed to converge"),
            StatsError::ContainerExpectedSum(s, sum) => {
                write!(f, "Elements in container {} expected to sum to {}", s, sum)
            }
            StatsError::ContainerExpectedSumVar(s, sum) => {
                write!(f, "Elements in container {} expected to sum to {}", s, sum)
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
