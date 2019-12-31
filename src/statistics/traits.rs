use crate::Result;

/// The `Min` trait specifies than an object has a minimum value
pub trait Min<T> {
    /// Returns the minimum value in the domain of a given distribution
    /// representable by a double-precision float. May panic depending on
    /// the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Min;
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.0, n.min());
    /// ```
    fn min(&self) -> T;
}

/// The `Max` trait specifies that an object has a maximum value
pub trait Max<T> {
    /// Returns the maximum value in the domain of a given distribution
    /// representable by a double-precision float. May panic depending on
    /// the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Max;
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(1.0, n.max());
    /// ```
    fn max(&self) -> T;
}

/// The `Mean` trait specifies that an object has a closed form
/// solution for its mean(s)
pub trait Mean<T> {
    /// Returns the mean. May panic depending
    /// on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Mean;
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.5, n.mean());
    /// ```
    fn mean(&self) -> T;
}

/// The `CheckedMean` trait specifies that an object has a closed form
/// solution for its mean(s) with possible failure modes
pub trait CheckedMean<T> {
    /// Returns the mean.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::CheckedMean;
    /// use statrs::distribution::FisherSnedecor;
    ///
    /// let n = FisherSnedecor::new(1.0, 1.0).unwrap();
    /// assert!(n.checked_mean().is_err());
    /// ```
    fn checked_mean(&self) -> Result<T>;
}

/// The `Variance` trait specifies that an object has a closed form solution for
/// its variance(s). Requires `Mean` since a closed form solution to
/// variance by definition requires a closed form mean.
pub trait Variance<T>: Mean<T> {
    /// Returns the variance. May panic depending
    /// on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Variance;
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(1.0 / 12.0, n.variance());
    /// ```
    fn variance(&self) -> T;

    /// Returns the standard deviation. May panic depending
    /// on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Variance;
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!((1f64 / 12f64).sqrt(), n.std_dev());
    /// ```
    fn std_dev(&self) -> T;
}

pub trait Covariance<T> {
    fn variance(&self) -> T;
}

pub trait CheckedVariance<T>: CheckedMean<T> {
    /// Returns the variance.
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::CheckedVariance;
    /// use statrs::distribution::FisherSnedecor;
    ///
    /// let n = FisherSnedecor::new(1.0, 1.0).unwrap();
    /// assert!(n.checked_variance().is_err());
    /// ```
    fn checked_variance(&self) -> Result<T>;

    /// Returns the standard deviation.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::CheckedVariance;
    /// use statrs::distribution::FisherSnedecor;
    ///
    /// let n = FisherSnedecor::new(1.0, 1.0).unwrap();
    /// assert!(n.checked_std_dev().is_err());
    /// ```
    fn checked_std_dev(&self) -> Result<T>;
}

/// The `Entropy` trait specifies an object that has a closed form solution
/// for its entropy
pub trait Entropy<T> {
    /// Returns the entropy. May panic depending
    /// on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Entropy;
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.0, n.entropy());
    /// ```
    fn entropy(&self) -> T;
}

/// The `CheckedEntropy` trait specifies an object that has a closed form
/// solutions for its entropy wih possible failure modes
pub trait CheckedEntropy<T> {
    /// Returns the entropy.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::CheckedEntropy;
    /// use statrs::distribution::StudentsT;
    ///
    /// let n = StudentsT::new(0.0, 2.0, 1.0).unwrap();
    /// assert!(n.checked_entropy().is_err());
    /// ```
    fn checked_entropy(&self) -> Result<T>;
}

/// The `Skewness` trait specifies an object that has a closed form solution
/// for its skewness(s)
pub trait Skewness<T> {
    /// Returns the skewness. May panic depending
    /// on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Skewness;
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.0, n.skewness());
    /// ```
    fn skewness(&self) -> T;
}

/// The `CheckedSkewness` trait specifies an object that has a closed form
/// solution for its skewness(s) with possible failure modes
pub trait CheckedSkewness<T> {
    /// Returns the skewness.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::CheckedSkewness;
    /// use statrs::distribution::FisherSnedecor;
    ///
    /// let n = FisherSnedecor::new(1.0, 1.0).unwrap();
    /// assert!(n.checked_skewness().is_err());
    /// ```
    fn checked_skewness(&self) -> Result<T>;
}

/// The `Median` trait specifies than an object has a closed form solution
/// for its median
pub trait Median<T> {
    /// Returns the median. May panic depending
    /// on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Median;
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.5, n.median());
    /// ```
    fn median(&self) -> T;
}

/// The `Mode` trait specififies that an object has a closed form solution
/// for its mode(s)
pub trait Mode<T> {
    /// Returns the mode. May panic depending on
    /// the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Mode;
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.5, n.mode());
    /// ```
    fn mode(&self) -> T;
}

/// The `CheckedMode` trait specifies that an object has a closed form solution
/// for its mode(s) with a possible failure mode
pub trait CheckedMode<T> {
    /// Returns the mode.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::CheckedMode;
    /// use statrs::distribution::Beta;
    ///
    /// let n = Beta::new(1.0, 1.0).unwrap();
    /// assert!(n.checked_mode().is_err());
    /// ```
    fn checked_mode(&self) -> Result<T>;
}
