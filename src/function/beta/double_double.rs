#[cfg(not(feature = "std"))]
use num_traits::Float as _;

pub(super) fn two_sum(left: f64, right: f64) -> (f64, f64) {
    let sum = left + right;
    let virtual_right = sum - left;
    let error = (left - (sum - virtual_right)) + (right - virtual_right);
    (sum, error)
}

pub(super) fn add((left, left_error): (f64, f64), (right, right_error): (f64, f64)) -> (f64, f64) {
    let (sum, error) = two_sum(left, right);
    two_sum(sum, error + left_error + right_error)
}

pub(super) fn multiply(
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

pub(super) fn divide_f64((numerator, numerator_error): (f64, f64), denominator: f64) -> (f64, f64) {
    let quotient = numerator / denominator;
    let remainder = (-quotient).mul_add(denominator, numerator) + numerator_error;
    two_sum(quotient, remainder / denominator)
}

pub(super) fn divide(numerator: (f64, f64), denominator: (f64, f64)) -> (f64, f64) {
    let quotient = numerator.0 / denominator.0;
    let product = multiply((quotient, 0.0), denominator);
    let remainder = add(numerator, (-product.0, -product.1));
    add(
        (quotient, 0.0),
        ((remainder.0 + remainder.1) / denominator.0, 0.0),
    )
}

pub(super) fn exp((value, error): (f64, f64)) -> f64 {
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

fn accurate_ln(value: f64) -> (f64, f64) {
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
    let ratio = divide(
        add((mantissa, 0.0), (-1.0, 0.0)),
        add((mantissa, 0.0), (1.0, 0.0)),
    );
    let ratio_squared = multiply(ratio, ratio);
    let mut term = ratio;
    let mut sum = ratio;
    for index in 1..=24 {
        term = multiply(term, ratio_squared);
        sum = add(sum, divide_f64(term, f64::from(2 * index + 1)));
        if term.0.abs() <= f64::EPSILON * f64::EPSILON * sum.0.abs() {
            break;
        }
    }
    let log_mantissa = multiply((2.0, 0.0), sum);
    let log_two = (core::f64::consts::LN_2, 2.3190468138462996e-17);
    add(multiply((f64::from(exponent), 0.0), log_two), log_mantissa)
}

fn accurate_ln_dd(value: (f64, f64)) -> (f64, f64) {
    let logarithm = accurate_ln(value.0);
    add(logarithm, ((value.1 / value.0).ln_1p(), 0.0))
}

pub(super) fn accurate_ln_one_plus(value: (f64, f64)) -> (f64, f64) {
    if value.0 == 0.0 && value.1 == 0.0 {
        return (0.0, 0.0);
    }
    if value.0.abs() > 0.5 {
        return accurate_ln_dd(add((1.0, 0.0), value));
    }
    let ratio = divide(value, add((2.0, 0.0), value));
    let ratio_squared = multiply(ratio, ratio);
    let mut term = ratio;
    let mut sum = ratio;
    for index in 1..=24 {
        term = multiply(term, ratio_squared);
        if term.0 == 0.0 && term.1 == 0.0 {
            break;
        }
        sum = add(sum, divide_f64(term, f64::from(2 * index + 1)));
        if term.0.abs() <= f64::EPSILON * f64::EPSILON * sum.0.abs() {
            break;
        }
    }
    multiply((2.0, 0.0), sum)
}
