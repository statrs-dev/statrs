#[cfg(all(not(feature = "std"), not(test)))]
use super::Float;

pub(super) fn accurate_ln_dd(value: (f64, f64)) -> (f64, f64) {
    let logarithm = accurate_ln(value.0);
    dd_add(logarithm, ((value.1 / value.0).ln_1p(), 0.0))
}

pub(super) fn accurate_ln_one_plus_dd(value: (f64, f64)) -> (f64, f64) {
    if value.0 == 0.0 && value.1 == 0.0 {
        return (0.0, 0.0);
    }
    if value.0.abs() > 0.5 {
        return accurate_ln_dd(dd_add((1.0, 0.0), value));
    }
    let ratio = dd_div(value, dd_add((2.0, 0.0), value));
    let ratio_squared = dd_mul(ratio, ratio);
    let mut term = ratio;
    let mut sum = ratio;
    for index in 1..=24 {
        term = dd_mul(term, ratio_squared);
        if term.0 == 0.0 && term.1 == 0.0 {
            break;
        }
        sum = dd_add(sum, dd_div_f64(term, f64::from(2 * index + 1)));
    }
    dd_mul((2.0, 0.0), sum)
}

pub(super) fn accurate_ln_one_minus_dd(value: f64) -> (f64, f64) {
    if value <= 0.5 {
        accurate_ln_one_plus_dd((-value, 0.0))
    } else {
        let complement = two_sum(1.0, -value);
        accurate_ln_dd(complement)
    }
}

pub(super) fn log1pmx(x: f64) -> f64 {
    if x.abs() > 0.01 {
        return x.ln_1p() - x;
    }

    let mut term = -0.5 * x * x;
    let mut sum = term;
    for n in 3..=64 {
        term *= -x * f64::from(n - 1) / f64::from(n);
        sum += term;
    }
    sum
}

pub(super) fn two_sum(left: f64, right: f64) -> (f64, f64) {
    let sum = left + right;
    let virtual_right = sum - left;
    let error = (left - (sum - virtual_right)) + (right - virtual_right);
    (sum, error)
}

pub(super) fn dd_add(
    (left, left_error): (f64, f64),
    (right, right_error): (f64, f64),
) -> (f64, f64) {
    let (sum, error) = two_sum(left, right);
    two_sum(sum, error + left_error + right_error)
}

pub(super) fn dd_mul(
    (left, left_error): (f64, f64),
    (right, right_error): (f64, f64),
) -> (f64, f64) {
    let product = left * right;
    let error = left.mul_add(right, -product)
        + left * right_error
        + left_error * right
        + left_error * right_error;
    two_sum(product, error)
}

pub(super) fn dd_div_f64((numerator, numerator_error): (f64, f64), denominator: f64) -> (f64, f64) {
    let quotient = numerator / denominator;
    let remainder = (-quotient).mul_add(denominator, numerator) + numerator_error;
    two_sum(quotient, remainder / denominator)
}

pub(super) fn dd_div(numerator: (f64, f64), denominator: (f64, f64)) -> (f64, f64) {
    let quotient = numerator.0 / denominator.0;
    let product = dd_mul((quotient, 0.0), denominator);
    let remainder = dd_add(numerator, (-product.0, -product.1));
    dd_add(
        (quotient, 0.0),
        ((remainder.0 + remainder.1) / denominator.0, 0.0),
    )
}

pub(super) fn dd_exp((value, error): (f64, f64)) -> f64 {
    let combined = value + error;
    if combined < f64::from_bits(1).ln() - core::f64::consts::LN_2 {
        return 0.0;
    }
    let exponential = value.exp();
    let error_expm1 = error.exp_m1();
    if exponential == 0.0 || !error_expm1.is_finite() {
        return combined.exp();
    }
    exponential.mul_add(error_expm1, exponential)
}

pub(super) fn dd_negative_expm1((value, error): (f64, f64)) -> f64 {
    let combined = value + error;
    if combined < f64::from_bits(1).ln() - core::f64::consts::LN_2 {
        return 1.0;
    }
    let exponential = value.exp();
    let error_expm1 = error.exp_m1();
    if exponential == 0.0 || !error_expm1.is_finite() {
        return -combined.exp_m1();
    }
    -value.exp_m1() - exponential * error_expm1
}

pub(super) fn accurate_ln(value: f64) -> (f64, f64) {
    if value == 1.0 {
        return (0.0, 0.0);
    }
    let mut scaled = value;
    let mut exponent_adjustment = 0_i32;
    if scaled < f64::MIN_POSITIVE {
        scaled *= 18_014_398_509_481_984.0;
        exponent_adjustment = -54;
    }
    let value_bits = scaled.to_bits();
    let mut exponent = ((value_bits >> 52) & 0x7ff) as i32 - 1023 + exponent_adjustment;
    let mut mantissa = f64::from_bits((value_bits & 0x000f_ffff_ffff_ffff) | (1023_u64 << 52));
    if mantissa > core::f64::consts::SQRT_2 {
        mantissa *= 0.5;
        exponent += 1;
    }
    let numerator = dd_add((mantissa, 0.0), (-1.0, 0.0));
    let denominator = dd_add((mantissa, 0.0), (1.0, 0.0));
    let ratio = dd_div(numerator, denominator);
    let ratio_squared = dd_mul(ratio, ratio);
    let mut term = ratio;
    let mut sum = ratio;
    for index in 1..=24 {
        term = dd_mul(term, ratio_squared);
        sum = dd_add(sum, dd_div_f64(term, f64::from(2 * index + 1)));
    }
    let log_mantissa = dd_mul((2.0, 0.0), sum);
    let log_two = (core::f64::consts::LN_2, 2.3190468138462996e-17);
    dd_add(dd_mul((f64::from(exponent), 0.0), log_two), log_mantissa)
}

pub(super) fn accurate_ln_one_minus(value: f64) -> (f64, f64) {
    accurate_ln_one_minus_dd(value)
}

pub(super) fn compensated_ln(value: f64) -> (f64, f64) {
    let high = value.ln();
    let low = if value >= f64::MIN_POSITIVE && !(0.5..=2.0).contains(&value) {
        value.mul_add((-high).exp(), -1.0).ln_1p()
    } else {
        0.0
    };
    (high, low)
}

pub(super) fn compensated_ln_one_minus(value: f64) -> (f64, f64) {
    if value <= 0.5 {
        ((-value).ln_1p(), 0.0)
    } else {
        let (complement, complement_error) = two_sum(1.0, -value);
        let (high, low) = compensated_ln(complement);
        (high, low + (complement_error / complement).ln_1p())
    }
}
