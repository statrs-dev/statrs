use crate::consts;
use crate::function::double_double::{accurate_ln_one_plus, add, divide, multiply, two_sum};
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum LogPrefactor {
    Value((f64, f64)),
    Underflow,
}

pub(super) fn log_ratio(a: f64, b: f64, x: f64) -> (f64, (f64, f64)) {
    let complement = two_sum(1.0, -x);
    let left = multiply((x, 0.0), (b, 0.0));
    let right = multiply(complement, (a, 0.0));
    let residual = add(left, (-right.0, -right.1));
    let left_ratio = divide(residual, (a, 0.0));
    let right_ratio = divide((-residual.0, -residual.1), (b, 0.0));
    let left_log = add(
        accurate_ln_one_plus(left_ratio),
        (-left_ratio.0, -left_ratio.1),
    );
    let right_log = add(
        accurate_ln_one_plus(right_ratio),
        (-right_ratio.0, -right_ratio.1),
    );
    let value = add(multiply((a, 0.0), left_log), multiply((b, 0.0), right_log));
    (residual.0 + residual.1, value)
}

pub(super) fn stirling_correction(value: f64) -> f64 {
    let inverse = 1.0 / value;
    let inverse_squared = inverse * inverse;
    let mut series: f64 = 7.0 / 1_092.0;
    for coefficient in [
        -691.0 / 360_360.0,
        1.0 / 1_188.0,
        -1.0 / 1_680.0,
        1.0 / 1_260.0,
        -1.0 / 360.0,
        1.0 / 12.0,
    ] {
        series = series.mul_add(inverse_squared, coefficient);
    }
    inverse * series
}

pub(super) fn log_prefactor(a: f64, b: f64, x: f64) -> Option<LogPrefactor> {
    if !(a.min(b) >= 10.0 && a.is_finite() && b.is_finite() && x > 0.0 && x < 1.0) {
        return None;
    }
    let scale = a.max(b);
    let scaled_a = a / scale;
    let scaled_b = b / scale;
    let scaled_sum = scaled_a + scaled_b;
    let central = 0.5
        * (scale.ln() + scaled_a.ln() + scaled_b.ln()
            - scaled_sum.ln()
            - 2.0 * consts::LN_SQRT_2PI)
        - stirling_correction(a)
        - stirling_correction(b)
        + stirling_correction(a + b);
    let ratio = log_ratio(a, b, x).1;
    if ratio.0.is_finite() && ratio.1.is_finite() {
        let value = add((central, 0.0), ratio);
        let logarithm = value.0 + value.1;
        return if logarithm == f64::NEG_INFINITY {
            Some(LogPrefactor::Underflow)
        } else if logarithm.is_finite() {
            Some(LogPrefactor::Value(value))
        } else {
            None
        };
    }
    let mean = scaled_a / scaled_sum;
    let complement = scaled_b / scaled_sum;
    let scaled_ratio =
        scaled_a * (x.ln() - mean.ln()) + scaled_b * ((-x).ln_1p() - complement.ln());
    let logarithm = scale * scaled_ratio + central;
    if logarithm == f64::NEG_INFINITY {
        Some(LogPrefactor::Underflow)
    } else if logarithm.is_finite() {
        Some(LogPrefactor::Value((logarithm, 0.0)))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::function::gamma;

    #[test]
    fn log_prefactor_rejects_boundaries() {
        assert_eq!(log_prefactor(10.0, 20.0, 0.0), None);
        assert_eq!(log_prefactor(10.0, 20.0, 1.0), None);
    }

    #[test]
    fn log_prefactor_matches_direct_formula() {
        let (a, b, x) = (12.0, 15.0, 0.4);
        let LogPrefactor::Value(parts) = log_prefactor(a, b, x).unwrap() else {
            panic!("expected a finite prefactor");
        };
        let actual = parts.0 + parts.1;
        let direct = gamma::ln_gamma(a + b) - gamma::ln_gamma(a) - gamma::ln_gamma(b)
            + a * x.ln()
            + b * (-x).ln_1p();
        assert!((actual - direct).abs() <= 1e-13);
        assert!(
            actual
                .to_bits()
                .abs_diff((-0.08970040659028639_f64).to_bits())
                <= 8
        );
    }

    #[test]
    fn log_prefactor_reports_extreme_tail_underflow() {
        assert_eq!(
            log_prefactor(1e308, 1e308, f64::from_bits(1)),
            Some(LogPrefactor::Underflow)
        );
    }
}
