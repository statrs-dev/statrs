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
        ($($arg_name:ident: $arg_ty:ty),+; $dist:ty) => {
            fn try_create($($arg_name: $arg_ty),+) -> $dist {
                let n = <$dist>::new($($arg_name),+);
                assert!(n.is_ok());
                n.unwrap()
            }

            fn bad_create_case($($arg_name: $arg_ty),+) {
                let n = <$dist>::new($($arg_name),+);
                assert!(n.is_err());
            }

            fn get_value<F, T>($($arg_name: $arg_ty),+, eval: F) -> T
            where
                F: Fn($dist) -> T,
            {
                let n = try_create($($arg_name),+);
                eval(n)
            }

            fn test_case<F, T>($($arg_name: $arg_ty),+, expected: T, eval: F)
            where
                F: Fn($dist) -> T,
                T: ::core::fmt::Debug + ::approx::RelativeEq<Epsilon = f64>,
            {
                let x = get_value($($arg_name),+, eval);
                assert_relative_eq!(expected, x, max_relative = $crate::consts::ACC);
            }

            #[allow(dead_code)] // This is not used by all distributions.
            fn test_case_special<F, T>($($arg_name: $arg_ty),+, expected: T, acc: f64, eval: F)
            where
                F: Fn($dist) -> T,
                T: ::core::fmt::Debug + ::approx::AbsDiffEq<Epsilon = f64>,
            {
                let x = get_value($($arg_name),+, eval);
                assert_abs_diff_eq!(expected, x, epsilon = acc);
            }

            #[allow(dead_code)] // This is not used by all distributions.
            fn test_none<F, T>($($arg_name: $arg_ty),+, eval: F)
            where
                F: Fn($dist) -> Option<T>,
                T: ::core::cmp::PartialEq + ::core::fmt::Debug,
            {
                let x = get_value($($arg_name),+, eval);
                assert_eq!(None, x);
            }
        };
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
