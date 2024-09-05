use num_traits::Num;

/// Returns true if there are no elements in `x` in `arr`
/// such that `x <= 0.0` or `x` is `f64::NAN` and `sum(arr) > 0.0`.
/// IF `incl_zero` is true, it tests for `x < 0.0` instead of `x <= 0.0`
pub fn is_valid_multinomial(arr: &[f64], incl_zero: bool) -> bool {
    let mut sum = 0.0;
    for &elt in arr {
        if incl_zero && elt < 0.0 || !incl_zero && elt <= 0.0 || elt.is_nan() {
            return false;
        }
        sum += elt;
    }
    sum != 0.0
}

#[cfg(feature = "nalgebra")]
use nalgebra::{Dim, OVector};

#[cfg(feature = "nalgebra")]
pub fn check_multinomial<D>(arr: &OVector<f64, D>, accept_zeroes: bool) -> crate::Result<()>
where
    D: Dim,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<f64, D>,
{
    use crate::StatsError;

    if arr.len() < 2 {
        return Err(StatsError::BadParams);
    }
    let mut sum = 0.0;
    for &x in arr.iter() {
        #[allow(clippy::if_same_then_else)]
        if x.is_nan() {
            return Err(StatsError::BadParams);
        } else if x.is_infinite() {
            return Err(StatsError::BadParams);
        } else if x < 0.0 {
            return Err(StatsError::BadParams);
        } else if x == 0.0 && !accept_zeroes {
            return Err(StatsError::BadParams);
        } else {
            sum += x;
        }
    }

    if sum != 0.0 {
        Ok(())
    } else {
        Err(StatsError::BadParams)
    }
}

/// Implements univariate function bisection searching for criteria
/// ```text
/// smallest k such that f(k) >= z
/// ```
/// Evaluates to `None` if
/// - provided interval has lower bound greater than upper bound
/// - function found not semi-monotone on the provided interval containing `z`
///
/// Evaluates to `Some(k)`, where `k` satisfies the search criteria
pub fn integral_bisection_search<K: Num + Clone, T: Num + PartialOrd>(
    f: impl Fn(&K) -> T,
    z: T,
    lb: K,
    ub: K,
) -> Option<K> {
    if !(f(&lb)..=f(&ub)).contains(&z) {
        return None;
    }
    let two = K::one() + K::one();
    let mut lb = lb;
    let mut ub = ub;
    loop {
        let mid = (lb.clone() + ub.clone()) / two.clone();
        if !(f(&lb)..=f(&ub)).contains(&f(&mid)) {
            // if f found not monotone on the interval
            return None;
        } else if f(&lb) == z {
            return Some(lb);
        } else if f(&ub) == z {
            return Some(ub);
        } else if (lb.clone() + K::one()) == ub {
            // no more elements to search
            return Some(ub);
        } else if f(&mid) >= z {
            ub = mid;
        } else {
            lb = mid;
        }
    }
}

#[macro_use]
#[cfg(test)]
pub mod test {
    use super::*;
    use crate::distribution::{Continuous, ContinuousCDF, Discrete, DiscreteCDF};

    #[macro_export]
    macro_rules! testing_boiler {
        ($($arg_name:ident: $arg_ty:ty),+; $dist:ty; $dist_err:ty) => {
            fn make_param_text($($arg_name: $arg_ty),+) -> String {
                // ""
                let mut param_text = String::new();

                // "shape=10.0, rate=NaN, "
                $(
                    param_text.push_str(
                        &format!(
                            "{}={:?}, ",
                            stringify!($arg_name),
                            $arg_name,
                        )
                    );
                )+

                // "shape=10.0, rate=NaN" (removes trailing comma and whitespace)
                param_text.pop();
                param_text.pop();

                param_text
            }

            /// Creates and returns a distribution with the given parameters,
            /// panicking if `::new` fails.
            fn create_ok($($arg_name: $arg_ty),+) -> $dist {
                match <$dist>::new($($arg_name),+) {
                    Ok(d) => d,
                    Err(e) => panic!(
                        "{}::new was expected to succeed, but failed for {} with error: '{}'",
                        stringify!($dist),
                        make_param_text($($arg_name),+),
                        e
                    )
                }
            }

            /// Returns the error when creating a distribution with the given parameters,
            /// panicking if `::new` succeeds.
            #[allow(dead_code)]
            fn create_err($($arg_name: $arg_ty),+) -> $dist_err {
                match <$dist>::new($($arg_name),+) {
                    Err(e) => e,
                    Ok(d) => panic!(
                        "{}::new was expected to fail, but succeeded for {} with result: {:?}",
                        stringify!($dist),
                        make_param_text($($arg_name),+),
                        d
                    )
                }
            }

            /// Creates a distribution with the given parameters, calls the `get_fn`
            /// function with the new distribution and returns the result of `get_fn`.
            ///
            /// Panics if `::new` fails.
            fn create_and_get<F, T>($($arg_name: $arg_ty),+, get_fn: F) -> T
            where
                F: Fn($dist) -> T,
            {
                let n = create_ok($($arg_name),+);
                get_fn(n)
            }

            /// Creates a distribution with the given parameters, calls the `get_fn`
            /// function with the new distribution and compares the result of `get_fn`
            /// to `expected` exactly.
            ///
            /// Panics if `::new` fails.
            #[allow(dead_code)]
            fn test_exact<F, T>($($arg_name: $arg_ty),+, expected: T, get_fn: F)
            where
                F: Fn($dist) -> T,
                T: ::core::cmp::PartialEq + ::core::fmt::Debug
            {
                let x = create_and_get($($arg_name),+, get_fn);
                if x != expected {
                    panic!(
                        "Expected {:?}, got {:?} for {}",
                        expected,
                        x,
                        make_param_text($($arg_name),+)
                    );
                }
            }

            /// Gets a value for the given parameters by calling `create_and_get`
            /// and compares it to `expected`.
            ///
            /// Allows relative error of up to [`crate::consts::ACC`].
            ///
            /// Panics if `::new` fails.
            #[allow(dead_code)]
            fn test_relative<F>($($arg_name: $arg_ty),+, expected: f64, get_fn: F)
            where
                F: Fn($dist) -> f64,
            {
                let x = create_and_get($($arg_name),+, get_fn);
                let max_relative = $crate::consts::ACC;

                if !::approx::relative_eq!(expected, x, max_relative = max_relative) {
                    panic!(
                        "Expected {:?} to be almost equal to {:?} (max. relative error of {:?}), but wasn't for {}",
                        x,
                        expected,
                        max_relative,
                        make_param_text($($arg_name),+)
                    );
                }
            }

            /// Gets a value for the given parameters by calling `create_and_get`
            /// and compares it to `expected`.
            ///
            /// Allows absolute error of up to `acc`.
            ///
            /// Panics if `::new` fails.
            #[allow(dead_code)]
            fn test_absolute<F>($($arg_name: $arg_ty),+, expected: f64, acc: f64, get_fn: F)
            where
                F: Fn($dist) -> f64,
            {
                let x = create_and_get($($arg_name),+, get_fn);

                // abs_diff_eq! cannot handle infinities, so we manually accept them here
                if expected.is_infinite() && x == expected {
                    return;
                }

                if !::approx::abs_diff_eq!(expected, x, epsilon = acc) {
                    panic!(
                        "Expected {:?} to be almost equal to {:?} (max. absolute error of {:?}), but wasn't for {}",
                        x,
                        expected,
                        acc,
                        make_param_text($($arg_name),+)
                    );
                }
            }

            /// Gets a value for the given parameters by calling `create_and_get`
            /// and asserts that it is [`NAN`].
            ///
            /// Panics if `::new` fails.
            #[allow(dead_code)]
            fn test_is_nan<F>($($arg_name: $arg_ty),+, get_fn: F)
            where
                F: Fn($dist) -> f64
            {
                let x = create_and_get($($arg_name),+, get_fn);
                assert!(x.is_nan());
            }

            /// Gets a value for the given parameters by calling `create_and_get`
            /// and asserts that it is [`None`].
            ///
            /// Panics if `::new` fails.
            #[allow(dead_code)]
            fn test_none<F, T>($($arg_name: $arg_ty),+, get_fn: F)
            where
                F: Fn($dist) -> Option<T>,
                T: ::core::fmt::Debug,
            {
                let x = create_and_get($($arg_name),+, get_fn);

                if let Some(inner) = x {
                    panic!(
                        "Expected None, got {:?} for {}",
                        inner,
                        make_param_text($($arg_name),+)
                    )
                }
            }
        };
    }

    pub mod boiler_tests {
        use crate::distribution::Binomial;
        use crate::statistics::*;
        use crate::StatsError;

        testing_boiler!(p: f64, n: u64; Binomial; StatsError);

        #[test]
        fn create_ok_success() {
            let b = create_ok(0.8, 1200);
            assert_eq!(b.p(), 0.8);
            assert_eq!(b.n(), 1200);
        }

        #[test]
        #[should_panic]
        fn create_err_failure() {
            create_err(0.8, 1200);
        }

        #[test]
        fn create_err_success() {
            let err = create_err(-0.5, 1000);
            assert_eq!(err, StatsError::BadParams);
        }

        #[test]
        #[should_panic]
        fn create_ok_failure() {
            create_ok(-0.5, 1000);
        }

        #[test]
        fn test_exact_success() {
            test_exact(0.0, 4, 0.0, |dist| dist.mean().unwrap());
        }

        #[test]
        #[should_panic]
        fn test_exact_failure() {
            test_exact(0.3, 3, 0.9, |dist| dist.mean().unwrap());
        }

        #[test]
        fn test_relative_success() {
            test_relative(0.3, 3, 0.9, |dist| dist.mean().unwrap());
        }

        #[test]
        #[should_panic]
        fn test_relative_failure() {
            test_relative(0.3, 3, 0.8, |dist| dist.mean().unwrap());
        }

        #[test]
        fn test_absolute_success() {
            test_absolute(0.3, 3, 0.9, 1e-15, |dist| dist.mean().unwrap());
        }

        #[test]
        #[should_panic]
        fn test_absolute_failure() {
            test_absolute(0.3, 3, 0.9, 1e-17, |dist| dist.mean().unwrap());
        }

        #[test]
        fn test_is_nan_success() {
            // Not sure that any Binomial API can return a NaN, so we force the issue
            test_is_nan(0.8, 1200, |_| f64::NAN);
        }

        #[test]
        #[should_panic]
        fn test_is_nan_failure() {
            test_is_nan(0.8, 1200, |dist| dist.mean().unwrap());
        }

        #[test]
        fn test_is_none_success() {
            // Same as test_is_nan_success, force returning `None` here
            test_none(0.8, 1200, |_| Option::<f64>::None);
        }

        #[test]
        #[should_panic]
        fn test_is_none_failure() {
            test_none(0.8, 1200, |dist| dist.mean());
        }
    }

    /// cdf should be the integral of the pdf
    fn check_integrate_pdf_is_cdf<D: ContinuousCDF<f64, f64> + Continuous<f64, f64>>(
        dist: &D,
        x_min: f64,
        x_max: f64,
        step: f64,
    ) {
        let mut prev_x = x_min;
        let mut prev_density = dist.pdf(x_min);
        let mut sum = 0.0;

        loop {
            let x = prev_x + step;
            let density = dist.pdf(x);

            assert!(density >= 0.0);

            let ln_density = dist.ln_pdf(x);

            assert_almost_eq!(density.ln(), ln_density, 1e-10);

            // triangle rule
            sum += (prev_density + density) * step / 2.0;

            let cdf = dist.cdf(x);
            if (sum - cdf).abs() > 1e-3 {
                println!("Integral of pdf doesn't equal cdf!");
                println!("Integration from {} by {} to {} = {}", x_min, step, x, sum);
                println!("cdf = {}", cdf);
                panic!();
            }

            if x >= x_max {
                break;
            } else {
                prev_x = x;
                prev_density = density;
            }
        }

        assert!(sum > 0.99);
        assert!(sum <= 1.001);
    }

    /// cdf should be the sum of the pmf
    fn check_sum_pmf_is_cdf<D: DiscreteCDF<u64, f64> + Discrete<u64, f64>>(dist: &D, x_max: u64) {
        let mut sum = 0.0;

        // go slightly beyond x_max to test for off-by-one errors
        for i in 0..x_max + 3 {
            let prob = dist.pmf(i);

            assert!(prob >= 0.0);
            assert!(prob <= 1.0);

            sum += prob;

            if i == x_max {
                assert!(sum > 0.99);
            }

            assert_almost_eq!(sum, dist.cdf(i), 1e-10);
            // assert_almost_eq!(sum, dist.cdf(i as f64), 1e-10);
            // assert_almost_eq!(sum, dist.cdf(i as f64 + 0.1), 1e-10);
            // assert_almost_eq!(sum, dist.cdf(i as f64 + 0.5), 1e-10);
            // assert_almost_eq!(sum, dist.cdf(i as f64 + 0.9), 1e-10);
        }

        assert!(sum > 0.99);
        assert!(sum <= 1.0 + 1e-10);
    }

    /// Does a series of checks that all continuous distributions must obey.
    /// 99% of the probability mass should be between x_min and x_max.
    pub fn check_continuous_distribution<D: ContinuousCDF<f64, f64> + Continuous<f64, f64>>(
        dist: &D,
        x_min: f64,
        x_max: f64,
    ) {
        assert_eq!(dist.pdf(f64::NEG_INFINITY), 0.0);
        assert_eq!(dist.pdf(f64::INFINITY), 0.0);
        assert_eq!(dist.ln_pdf(f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert_eq!(dist.ln_pdf(f64::INFINITY), f64::NEG_INFINITY);
        assert_eq!(dist.cdf(f64::NEG_INFINITY), 0.0);
        assert_eq!(dist.cdf(f64::INFINITY), 1.0);

        check_integrate_pdf_is_cdf(dist, x_min, x_max, (x_max - x_min) / 100000.0);
    }

    /// Does a series of checks that all positive discrete distributions must
    /// obey.
    /// 99% of the probability mass should be between 0 and x_max (inclusive).
    pub fn check_discrete_distribution<D: DiscreteCDF<u64, f64> + Discrete<u64, f64>>(
        dist: &D,
        x_max: u64,
    ) {
        // assert_eq!(dist.cdf(f64::NEG_INFINITY), 0.0);
        // assert_eq!(dist.cdf(-10.0), 0.0);
        // assert_eq!(dist.cdf(-1.0), 0.0);
        // assert_eq!(dist.cdf(-0.01), 0.0);
        // assert_eq!(dist.cdf(f64::INFINITY), 1.0);

        check_sum_pmf_is_cdf(dist, x_max);
    }

    #[cfg(feature = "nalgebra")]
    #[test]
    fn test_is_valid_multinomial() {
        use std::f64;

        let invalid = [1.0, f64::NAN, 3.0];
        assert!(!is_valid_multinomial(&invalid, true));
        assert!(check_multinomial(&invalid.to_vec().into(), true).is_err());
        let invalid2 = [-2.0, 5.0, 1.0, 6.2];
        assert!(!is_valid_multinomial(&invalid2, true));
        assert!(check_multinomial(&invalid2.to_vec().into(), true).is_err());
        let invalid3 = [0.0, 0.0, 0.0];
        assert!(!is_valid_multinomial(&invalid3, true));
        assert!(check_multinomial(&invalid3.to_vec().into(), true).is_err());
        let valid = [5.2, 0.0, 1e-15, 1000000.12];
        assert!(is_valid_multinomial(&valid, true));
        assert!(check_multinomial(&valid.to_vec().into(), true).is_ok());
    }

    #[test]
    fn test_is_valid_multinomial_no_zero() {
        let invalid = [5.2, 0.0, 1e-15, 1000000.12];
        assert!(!is_valid_multinomial(&invalid, false));
    }

    #[test]
    fn test_integer_bisection() {
        fn search(z: usize, data: &[usize]) -> Option<usize> {
            integral_bisection_search(|idx: &usize| data[*idx], z, 0, data.len() - 1)
        }

        let needle = 3;
        let data = (0..5)
            .map(|n| if n >= needle { n + 1 } else { n })
            .collect::<Vec<_>>();

        for i in 0..(data.len()) {
            assert_eq!(search(data[i], &data), Some(i),)
        }
        {
            let infimum = search(needle, &data);
            let found_element = search(needle + 1, &data); // 4 > needle && member of range
            assert_eq!(found_element, Some(needle));
            assert_eq!(infimum, found_element)
        }
    }
}
