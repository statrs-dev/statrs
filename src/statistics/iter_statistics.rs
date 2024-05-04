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
    fn test_mean() {
        let mut data = testing::load_data("nist/lottery.txt");
        assert_almost_eq!((&data).mean(), 518.958715596330, 1e-12);

        data = testing::load_data("nist/lew.txt");
        assert_almost_eq!((&data).mean(), -177.435000000000, 1e-13);

        data = testing::load_data("nist/mavro.txt");
        assert_almost_eq!((&data).mean(), 2.00185600000000, 1e-15);

        data = testing::load_data("nist/michaelso.txt");
        assert_almost_eq!((&data).mean(), 299.852400000000, 1e-13);

        data = testing::load_data("nist/numacc1.txt");
        assert_eq!((&data).mean(), 10000002.0);

        data = testing::load_data("nist/numacc2.txt");
        assert_almost_eq!((&data).mean(), 1.2, 1e-15);

        data = testing::load_data("nist/numacc3.txt");
        assert_eq!((&data).mean(), 1000000.2);

        data = testing::load_data("nist/numacc4.txt");
        assert_almost_eq!((&data).mean(), 10000000.2, 1e-8);
    }

    #[test]
    fn test_std_dev() {
        let mut data = testing::load_data("nist/lottery.txt");
        assert_almost_eq!((&data).std_dev(), 291.699727470969, 1e-13);

        data = testing::load_data("nist/lew.txt");
        assert_almost_eq!((&data).std_dev(), 277.332168044316, 1e-12);

        data = testing::load_data("nist/mavro.txt");
        assert_almost_eq!((&data).std_dev(), 0.000429123454003053, 1e-15);

        data = testing::load_data("nist/michaelso.txt");
        assert_almost_eq!((&data).std_dev(), 0.0790105478190518, 1e-13);

        data = testing::load_data("nist/numacc1.txt");
        assert_eq!((&data).std_dev(), 1.0);

        data = testing::load_data("nist/numacc2.txt");
        assert_almost_eq!((&data).std_dev(), 0.1, 1e-16);

        data = testing::load_data("nist/numacc3.txt");
        assert_almost_eq!((&data).std_dev(), 0.1, 1e-10);

        data = testing::load_data("nist/numacc4.txt");
        assert_almost_eq!((&data).std_dev(), 0.1, 1e-9);
    }

    #[test]
    fn test_min_max_short() {
        let data = [-1.0, 5.0, 0.0, -3.0, 10.0, -0.5, 4.0];
        assert_eq!(data.min(), -3.0);
        assert_eq!(data.max(), 10.0);
    }

    #[test]
    fn test_mean_variance_stability() {
        let seed = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
        ];
        let mut rng: StdRng = SeedableRng::from_seed(seed);
        let normal = Normal::new(1e9, 2.0).unwrap();
        let samples = (0..10000).map(|_| normal.sample::<StdRng>(&mut rng)).collect::<Vec<f64>>();
        assert_almost_eq!((&samples).mean(), 1e9, 10.0);
        assert_almost_eq!((&samples).variance(), 4.0, 0.1);
        assert_almost_eq!((&samples).std_dev(), 2.0, 0.01);
        assert_almost_eq!((&samples).quadratic_mean(), 1e9, 10.0);
    }

    #[test]
    fn test_covariance_consistent_with_variance() {
        let mut data = testing::load_data("nist/lottery.txt");
        assert_almost_eq!((&data).variance(), (&data).covariance(&data), 1e-10);

        data = testing::load_data("nist/lew.txt");
        assert_almost_eq!((&data).variance(), (&data).covariance(&data), 1e-10);

        data = testing::load_data("nist/mavro.txt");
        assert_almost_eq!((&data).variance(), (&data).covariance(&data), 1e-10);

        data = testing::load_data("nist/michaelso.txt");
        assert_almost_eq!((&data).variance(), (&data).covariance(&data), 1e-10);

        data = testing::load_data("nist/numacc1.txt");
        assert_almost_eq!((&data).variance(), (&data).covariance(&data), 1e-10);
    }

    #[test]
    fn test_pop_covar_consistent_with_pop_var() {
        let mut data = testing::load_data("nist/lottery.txt");
        assert_almost_eq!((&data).population_variance(), (&data).population_covariance(&data), 1e-10);

        data = testing::load_data("nist/lew.txt");
        assert_almost_eq!((&data).population_variance(), (&data).population_covariance(&data), 1e-10);

        data = testing::load_data("nist/mavro.txt");
        assert_almost_eq!((&data).population_variance(), (&data).population_covariance(&data), 1e-10);

        data = testing::load_data("nist/michaelso.txt");
        assert_almost_eq!((&data).population_variance(), (&data).population_covariance(&data), 1e-10);

        data = testing::load_data("nist/numacc1.txt");
        assert_almost_eq!((&data).population_variance(), (&data).population_covariance(&data), 1e-10);
    }

    #[test]
    fn test_covariance_is_symmetric() {
        let data_a = &testing::load_data("nist/lottery.txt")[0..200];
        let data_b = &testing::load_data("nist/lew.txt")[0..200];
        assert_almost_eq!(data_a.covariance(data_b), data_b.covariance(data_a), 1e-10);
        assert_almost_eq!(data_a.population_covariance(data_b), data_b.population_covariance(data_a), 1e-11);
    }

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
