use ::nalgebra::{
    base::allocator::Allocator, base::dimension::DimName, DefaultAllocator, Dim, DimMin, U1,
};
use ::num_traits::float::Float;

const STEPS: usize = 1_000;

/// The `Min` trait specifies than an object has a minimum value
pub trait Min<T> {
    /// Returns the minimum value in the domain of a given distribution
    /// if it exists, otherwise `None`.
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
    /// if it exists, otherwise `None`.
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
pub trait DiscreteDistribution<T: Float>: ::rand::distributions::Distribution<u64> {
    /// Returns the mean, if it exists.
    fn mean(&self) -> Option<T> {
        None
    }
    /// Returns the variance, if it exists.
    fn variance(&self) -> Option<T> {
        None
    }
    /// Returns the standard deviation, if it exists.
    fn std_dev(&self) -> Option<T> {
        self.variance().map(|var| var.sqrt())
    }
    /// Returns the entropy, if it exists.
    fn entropy(&self) -> Option<T> {
        None
    }
    /// Returns the skewness, if it exists.
    fn skewness(&self) -> Option<T> {
        None
    }
}

pub trait Distribution<T: Float>: ::rand::distributions::Distribution<T> {
    /// Returns the mean, if it exists.
    /// The default implementation returns an estimation
    /// based on random samples. This is a crude estimate
    /// for when no further information is known about the
    /// distribution. More accurate statements about the
    /// mean can and should be given by overriding the
    /// default implementation.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Distribution;
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.5, n.mean().unwrap());
    /// ```
    fn mean(&self) -> Option<T> {
        // TODO: Does not need cryptographic rng
        let mut rng = ::rand::rngs::OsRng;
        let mut mean = T::zero();
        let mut steps = T::zero();
        for _ in 0..STEPS {
            steps = steps + T::one();
            mean = mean + Self::sample(self, &mut rng);
        }
        Some(mean / steps)
    }
    /// Returns the variance, if it exists.
    /// The default implementation returns an estimation
    /// based on random samples. This is a crude estimate
    /// for when no further information is known about the
    /// distribution. More accurate statements about the
    /// variance can and should be given by overriding the
    /// default implementation.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Distribution;
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(1.0 / 12.0, n.variance().unwrap());
    /// ```
    fn variance(&self) -> Option<T> {
        // TODO: Does not need cryptographic rng
        let mut rng = ::rand::rngs::OsRng;
        let mut mean = T::zero();
        let mut variance = T::zero();
        let mut steps = T::zero();
        for _ in 0..STEPS {
            steps = steps + T::one();
            let sample = Self::sample(self, &mut rng);
            variance = variance + (steps - T::one()) * (sample - mean) * (sample - mean) / steps;
            mean = mean + (sample - mean) / steps;
        }
        steps = steps - T::one();
        Some(variance / steps)
    }
    /// Returns the standard deviation, if it exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Distribution;
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!((1f64 / 12f64).sqrt(), n.std_dev().unwrap());
    /// ```
    fn std_dev(&self) -> Option<T> {
        self.variance().map(|var| var.sqrt())
    }
    /// Returns the entropy, if it exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Distribution;
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.0, n.entropy().unwrap());
    /// ```
    fn entropy(&self) -> Option<T> {
        None
    }
    /// Returns the skewness, if it exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Distribution;
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.0, n.skewness().unwrap());
    /// ```
    fn skewness(&self) -> Option<T> {
        None
    }
}

/// The `Mean` trait implements the calculation of a mean.
// TODO: Clarify the traits of multidimensional distributions
pub trait MeanN<T> {
    fn mean(&self) -> Option<T>;
}

// TODO: Clarify the traits of multidimensional distributions
pub trait VarianceN<T> {
    fn variance(&self) -> Option<T>;
}

/// The `Median` trait returns the median of the distribution.
pub trait Median<T> {
    /// Returns the median.
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

/// The `Mode` trait specifies that an object has a closed form solution
/// for its mode(s)
pub trait Mode<T> {
    /// Returns the mode, if one exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Mode;
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(Some(0.5), n.mode());
    /// ```
    fn mode(&self) -> T;
}
