use crate::error::StatsError;
use crate::statistics::*;
use std::borrow::Borrow;
use std::f64;

impl<T> Statistics<f64> for T
where
    T: IntoIterator,
    T::Item: Borrow<f64>,
{
    fn min(self) -> f64 {
        let mut iter = self.into_iter();
        match iter.next() {
            None => f64::NAN,
            Some(x) => iter.map(|x| *x.borrow()).fold(*x.borrow(), |acc, x| {
                if x < acc || x.is_nan() {
                    x
                } else {
                    acc
                }
            }),
        }
    }

    fn max(self) -> f64 {
        let mut iter = self.into_iter();
        match iter.next() {
            None => f64::NAN,
            Some(x) => iter.map(|x| *x.borrow()).fold(*x.borrow(), |acc, x| {
                if x > acc || x.is_nan() {
                    x
                } else {
                    acc
                }
            }),
        }
    }

    fn abs_min(self) -> f64 {
        let mut iter = self.into_iter();
        match iter.next() {
            None => f64::NAN,
            Some(init) => iter
                .map(|x| x.borrow().abs())
                .fold(init.borrow().abs(), |acc, x| {
                    if x < acc || x.is_nan() {
                        x
                    } else {
                        acc
                    }
                }),
        }
    }

    fn abs_max(self) -> f64 {
        let mut iter = self.into_iter();
        match iter.next() {
            None => f64::NAN,
            Some(init) => iter
                .map(|x| x.borrow().abs())
                .fold(init.borrow().abs(), |acc, x| {
                    if x > acc || x.is_nan() {
                        x
                    } else {
                        acc
                    }
                }),
        }
    }

    fn mean(self) -> f64 {
        let mut i = 0.0;
        let mut mean = 0.0;
        for x in self {
            i += 1.0;
            mean += (x.borrow() - mean) / i;
        }
        if i > 0.0 {
            mean
        } else {
            f64::NAN
        }
    }

    fn geometric_mean(self) -> f64 {
        let mut i = 0.0;
        let mut sum = 0.0;
        for x in self {
            i += 1.0;
            sum += x.borrow().ln();
        }
        if i > 0.0 {
            (sum / i).exp()
        } else {
            f64::NAN
        }
    }

    fn harmonic_mean(self) -> f64 {
        let mut i = 0.0;
        let mut sum = 0.0;
        for x in self {
            i += 1.0;

            let borrow = *x.borrow();
            if borrow < 0f64 {
                return f64::NAN;
            }
            sum += 1.0 / borrow;
        }
        if i > 0.0 {
            i / sum
        } else {
            f64::NAN
        }
    }

    fn variance(self) -> f64 {
        let mut iter = self.into_iter();
        let mut sum = match iter.next() {
            None => f64::NAN,
            Some(x) => *x.borrow(),
        };
        let mut i = 1.0;
        let mut variance = 0.0;

        for x in iter {
            i += 1.0;
            let borrow = *x.borrow();
            sum += borrow;
            let diff = i * borrow - sum;
            variance += diff * diff / (i * (i - 1.0))
        }
        if i > 1.0 {
            variance / (i - 1.0)
        } else {
            f64::NAN
        }
    }

    fn std_dev(self) -> f64 {
        self.variance().sqrt()
    }

    fn population_variance(self) -> f64 {
        let mut iter = self.into_iter();
        let mut sum = match iter.next() {
            None => return f64::NAN,
            Some(x) => *x.borrow(),
        };
        let mut i = 1.0;
        let mut variance = 0.0;

        for x in iter {
            i += 1.0;
            let borrow = *x.borrow();
            sum += borrow;
            let diff = i * borrow - sum;
            variance += diff * diff / (i * (i - 1.0));
        }
        variance / i
    }

    fn population_std_dev(self) -> f64 {
        self.population_variance().sqrt()
    }

    fn covariance(self, other: Self) -> f64 {
        let mut n = 0.0;
        let mut mean1 = 0.0;
        let mut mean2 = 0.0;
        let mut comoment = 0.0;

        let mut iter = other.into_iter();
        for x in self {
            let borrow = *x.borrow();
            let borrow2 = match iter.next() {
                None => panic!("{}", StatsError::ContainersMustBeSameLength),
                Some(x) => *x.borrow(),
            };
            let old_mean2 = mean2;
            n += 1.0;
            mean1 += (borrow - mean1) / n;
            mean2 += (borrow2 - mean2) / n;
            comoment += (borrow - mean1) * (borrow2 - old_mean2);
        }
        if iter.next().is_some() {
            panic!("{}", StatsError::ContainersMustBeSameLength);
        }

        if n > 1.0 {
            comoment / (n - 1.0)
        } else {
            f64::NAN
        }
    }

    fn population_covariance(self, other: Self) -> f64 {
        let mut n = 0.0;
        let mut mean1 = 0.0;
        let mut mean2 = 0.0;
        let mut comoment = 0.0;

        let mut iter = other.into_iter();
        for x in self {
            let borrow = *x.borrow();
            let borrow2 = match iter.next() {
                None => panic!("{}", StatsError::ContainersMustBeSameLength),
                Some(x) => *x.borrow(),
            };
            let old_mean2 = mean2;
            n += 1.0;
            mean1 += (borrow - mean1) / n;
            mean2 += (borrow2 - mean2) / n;
            comoment += (borrow - mean1) * (borrow2 - old_mean2);
        }
        if iter.next().is_some() {
            panic!("{}", StatsError::ContainersMustBeSameLength)
        }
        if n > 0.0 {
            comoment / n
        } else {
            f64::NAN
        }
    }

    fn quadratic_mean(self) -> f64 {
        let mut i = 0.0;
        let mut mean = 0.0;
        for x in self {
            let borrow = *x.borrow();
            i += 1.0;
            mean += (borrow * borrow - mean) / i;
        }
        if i > 0.0 {
            mean.sqrt()
        } else {
            f64::NAN
        }
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use std::f64::consts;
    use rand::rngs::StdRng;
    use rand::{SeedableRng};
    use rand::distributions::Distribution;
    use crate::distribution::Normal;
    use crate::statistics::Statistics;
    use crate::generate::{InfinitePeriodic, InfiniteSinusoidal};
    use crate::testing;

    #[test]
    fn test_empty_data_returns_nan() {
        let data = [0.0; 0];
        assert!(data.min().is_nan());
        assert!(data.max().is_nan());
        assert!(data.mean().is_nan());
        assert!(data.quadratic_mean().is_nan());
        assert!(data.variance().is_nan());
        assert!(data.population_variance().is_nan());
    }

    // TODO: test github issue 137 (Math.NET)

    #[test]
    fn test_large_samples() {
        let shorter = InfinitePeriodic::default(4.0, 1.0).take(4*4096).collect::<Vec<f64>>();
        let longer = InfinitePeriodic::default(4.0, 1.0).take(4*32768).collect::<Vec<f64>>();
        assert_almost_eq!((&shorter).mean(), 0.375, 1e-14);
        assert_almost_eq!((&longer).mean(), 0.375, 1e-14);
        assert_almost_eq!((&shorter).quadratic_mean(), (0.21875f64).sqrt(), 1e-14);
        assert_almost_eq!((&longer).quadratic_mean(), (0.21875f64).sqrt(), 1e-14);
    }

    #[test]
    fn test_quadratic_mean_of_sinusoidal() {
        let data = InfiniteSinusoidal::default(64.0, 16.0, 2.0).take(128).collect::<Vec<f64>>();
        assert_almost_eq!((&data).quadratic_mean(), 2.0 / consts::SQRT_2, 1e-15);
    }
}
