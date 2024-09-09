/// https://en.wikipedia.org/wiki/Gumbel_distribution
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Gumbel {
    location: f64,
    scale: f64,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum GumbelError {
    /// The location is invalid (NAN)
    LocationInvalid,

    /// The scale is NAN, zero or less than zero
    ScaleInvalid,
}

impl std::fmt::Display for GumbelError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GumbelError::LocationInvalid => write!(f, "Location is NAN"),
            GumbelError::ScaleInvalid => write!(f, "Scale is NAN, zero or less than zero"),
        }
    }
}

impl std::error::Error for GumbelError {}

impl Gumbel {
    pub fn new(location: f64, scale: f64) -> Result<Self, GumbelError> {
        if location.is_nan() {
            return Err(GumbelError::LocationInvalid);
        }

        if scale.is_nan() && scale <= 0.0 {
            return Err(GumbelError::ScaleInvalid);
        }

        Ok(Self { location, scale })
    }

    pub fn location(&self) -> f64 {
        self.location
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }
}

impl std::fmt::Display for Gumbel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Gumbel({:?}, {:?})", self.location(), self.scale())
    }
}
