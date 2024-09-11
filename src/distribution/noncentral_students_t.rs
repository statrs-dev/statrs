use crate::distribution::{Continuous, ContinuousCDF};
use crate::function::{beta, gamma};
use crate::statistics::{Max, Min};

/// the non-central students T distribution
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct NoncentralStudentsT {
    location: f64,
    scale: f64,
    freedom: f64,
}

/// Represents the errors that can occur when creating a [`NoncentralStudentsT`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum NoncentralStudentsTError {
    /// The location is NaN.
    LocationInvalid,

    /// The scale is NaN, zero or less than zero.
    ScaleInvalid,

    /// The degrees of freedom are NaN, zero or less than zero.
    FreedomInvalid,
}

impl std::fmt::Display for NoncentralStudentsTError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            NoncentralStudentsTError::LocationInvalid => write!(f, "Location is NaN"),
            NoncentralStudentsTError::ScaleInvalid => {
                write!(f, "Scale is NaN, zero or less than zero")
            }
            NoncentralStudentsTError::FreedomInvalid => {
                write!(f, "Degrees of freedom are NaN, zero or less than zero")
            }
        }
    }
}

impl std::error::Error for NoncentralStudentsTError {}

impl NoncentralStudentsT {
    /// Constructs a new student's t-distribution with location `location`,
    /// scale `scale`, and `freedom` freedom.
    ///
    /// # Errors
    ///
    /// Returns an error if any of `location`, `scale`, or `freedom` are `NaN`.
    /// Returns an error if `scale <= 0.0` or `freedom <= 0.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::NoncentralStudentsT;
    ///
    /// let mut result = NoncentralStudentsT::new(0.0, 1.0, 2.0);
    /// assert!(result.is_ok());
    ///
    /// result = NoncentralStudentsT::new(0.0, 0.0, 0.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(
        location: f64,
        scale: f64,
        freedom: f64,
    ) -> Result<NoncentralStudentsT, NoncentralStudentsTError> {
        if location.is_nan() {
            return Err(NoncentralStudentsTError::LocationInvalid);
        }

        if scale.is_nan() || scale <= 0.0 {
            return Err(NoncentralStudentsTError::ScaleInvalid);
        }

        if freedom.is_nan() || freedom <= 0.0 {
            return Err(NoncentralStudentsTError::FreedomInvalid);
        }

        Ok(NoncentralStudentsT {
            location,
            scale,
            freedom,
        })
    }

    /// Returns the location of the student's t-distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::NoncentralStudentsT;
    ///
    /// let n = NoncentralStudentsT::new(0.0, 1.0, 2.0).unwrap();
    /// assert_eq!(n.location(), 0.0);
    /// ```
    pub fn location(&self) -> f64 {
        self.location
    }

    /// Returns the scale of the student's t-distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::NoncentralStudentsT;
    ///
    /// let n = NoncentralStudentsT::new(0.0, 1.0, 2.0).unwrap();
    /// assert_eq!(n.scale(), 1.0);
    /// ```
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Returns the freedom of the student's t-distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::NoncentralStudentsT;
    ///
    /// let n = NoncentralStudentsT::new(0.0, 1.0, 2.0).unwrap();
    /// assert_eq!(n.freedom(), 2.0);
    /// ```
    pub fn freedom(&self) -> f64 {
        self.freedom
    }
}

impl std::fmt::Display for NoncentralStudentsT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "t_{}({},{})", self.freedom, self.location, self.scale)
    }
}

impl ContinuousCDF<f64, f64> for NoncentralStudentsT {
    /// computes the distribution function for noncentral students t distribution
    ///
    /// ```math
    /// F(t;\nu,\delta) = \Phi(-\delta) + \frac{1}{2}\sum_{i=0}^\infty{\left[P_i I_x(i+ 1/2, n/2) + Q_i I_x(i+ 1/2, n/2)\right]}
    /// ```
    fn cdf(&self, x: f64) -> f64 {
        unimplemented!()
    }
}

impl Min<f64> for NoncentralStudentsT {
    /// Returns the minimum value in the domain of the student's t-distribution
    /// representable by a double precision float
    ///
    /// # Formula
    ///
    /// ```text
    /// f64::NEG_INFINITY
    /// ```
    fn min(&self) -> f64 {
        f64::NEG_INFINITY
    }
}

impl Max<f64> for NoncentralStudentsT {
    /// Returns the maximum value in the domain of the student's t-distribution
    /// representable by a double precision float
    ///
    /// # Formula
    ///
    /// ```text
    /// f64::INFINITY
    /// ```
    fn max(&self) -> f64 {
        f64::INFINITY
    }
}
