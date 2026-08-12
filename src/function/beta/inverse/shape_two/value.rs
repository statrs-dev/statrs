use super::super::super::*;

pub(super) fn log_cdf(a: f64, b: f64, x: f64) -> f64 {
    if b == 2.0 {
        a.mul_add(x.ln(), a.mul_add(1.0 - x, 1.0).ln())
    } else if x < 0.5 && b * x < 0.5 {
        let mut term = 0.5;
        let mut sum = term;
        for k in 1..64 {
            let k = f64::from(k);
            term *= -(b - k) / k * x * (k + 1.0) / (k + 2.0);
            sum += term;
            if term.abs() <= f64::EPSILON * sum.abs() {
                return b.ln() + (b + 1.0).ln() + 2.0 * x.ln() + sum.ln();
            }
        }
        panic!("shape-two beta series did not converge for b={b}, x={x}")
    } else {
        let log_tail = b.mul_add((-x).ln_1p(), b.mul_add(x, 1.0).ln());
        log1mexp(log_tail)
    }
}

pub(super) fn log_cdf_parts(a: f64, b: f64, x: f64) -> (f64, f64) {
    if b == 2.0 {
        let complement = two_sum(1.0, -x);
        let factor = dd_add((1.0, 0.0), dd_mul((a, 0.0), complement));
        dd_add(
            dd_mul((a, 0.0), shape_two_ln((x, 0.0))),
            shape_two_ln(factor),
        )
    } else if x < 0.5 && b * x < 0.5 {
        let sum = series_sum_dd(b, x)
            .unwrap_or_else(|| panic!("shape-two beta series did not converge for b={b}, x={x}"));
        dd_add(
            dd_add(
                dd_add(accurate_ln_dd((b, 0.0)), accurate_ln_dd((b + 1.0, 0.0))),
                dd_mul((2.0, 0.0), accurate_ln_dd((x, 0.0))),
            ),
            accurate_ln_dd(sum),
        )
    } else {
        accurate_ln_dd(tail_cdf_parts(b, x))
    }
}

fn tail_cdf_parts(b: f64, x: f64) -> (f64, f64) {
    tail_cdf_parts_dd(b, (x, 0.0))
}

fn tail_cdf_parts_dd(b: f64, x: (f64, f64)) -> (f64, f64) {
    let log_tail = dd_add(
        dd_mul((b, 0.0), shape_two_ln_one_plus((-x.0, -x.1))),
        shape_two_ln_one_plus(dd_mul((b, 0.0), x)),
    );
    let exponential = log_tail.0.exp();
    let exponential_error = exponential * log_tail.1.exp_m1();
    if log_tail.0 < -core::f64::consts::LN_2 {
        dd_add((1.0, 0.0), (-exponential, -exponential_error))
    } else {
        two_sum(-log_tail.0.exp_m1(), -exponential_error)
    }
}

fn shape_two_ln_one_plus(value: (f64, f64)) -> (f64, f64) {
    if value.0.abs() > 0.5 {
        return shape_two_ln(dd_add((1.0, 0.0), value));
    }
    let ratio = dd_div(value, dd_add((2.0, 0.0), value));
    let ratio_squared = dd_mul(ratio, ratio);
    let mut term = ratio;
    let mut sum = ratio;
    for index in 1..=24 {
        term = dd_mul(term, ratio_squared);
        sum = dd_add(sum, dd_div_f64(term, f64::from(2 * index + 1)));
        if term.0.abs() <= f64::EPSILON * f64::EPSILON * sum.0.abs() {
            break;
        }
    }
    dd_mul((2.0, 0.0), sum)
}

fn shape_two_ln(value: (f64, f64)) -> (f64, f64) {
    let mut scaled = value;
    let mut exponent_adjustment = 0_i32;
    if scaled.0 < f64::MIN_POSITIVE {
        scaled.0 *= 18_014_398_509_481_984.0;
        scaled.1 *= 18_014_398_509_481_984.0;
        exponent_adjustment = -54;
    }
    let mut exponent = ((scaled.0.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let mut mantissa =
        f64::from_bits((scaled.0.to_bits() & 0x000f_ffff_ffff_ffff) | (1023_u64 << 52));
    if mantissa > core::f64::consts::SQRT_2 {
        mantissa *= 0.5;
        exponent += 1;
    }
    let scale = 2.0_f64.powi(exponent);
    let mantissa = (mantissa, scaled.1 / scale);
    dd_add(
        dd_mul(
            (f64::from(exponent + exponent_adjustment), 0.0),
            (core::f64::consts::LN_2, 2.3190468138462996e-17),
        ),
        shape_two_ln_one_plus(dd_add(mantissa, (-1.0, 0.0))),
    )
}

fn series_sum_dd(b: f64, x: f64) -> Option<(f64, f64)> {
    let mut term = (0.5, 0.0);
    let mut sum = term;
    for k in 1..64 {
        let k = f64::from(k);
        let coefficient = dd_div_f64(dd_mul(dd_add((b, 0.0), (-k, 0.0)), (x, 0.0)), k);
        let coefficient = dd_mul(coefficient, (-(k + 1.0) / (k + 2.0), 0.0));
        term = dd_mul(term, coefficient);
        sum = dd_add(sum, term);
        if term.0.abs() <= f64::EPSILON * sum.0.abs() {
            return Some(sum);
        }
    }
    None
}

fn integer_power(value: (f64, f64), exponent: f64) -> Option<(f64, f64)> {
    if exponent != exponent.trunc() || !(1.0..18_446_744_073_709_551_616.0).contains(&exponent) {
        return None;
    }
    let mut exponent = exponent as u64;
    let mut factor = value;
    let mut result = (1.0, 0.0);
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = dd_mul(result, factor);
        }
        exponent >>= 1;
        if exponent != 0 {
            factor = dd_mul(factor, factor);
        }
    }
    (result.0 >= f64::MIN_POSITIVE && result.0.is_finite()).then_some(result)
}

pub(super) fn direct_cdf_and_pdf(a: f64, b: f64, x: f64) -> Option<((f64, f64), f64)> {
    if a == 2.0 && (x < 0.5 && b * x < 0.5 || (b == b.trunc() && b <= 64.0 && b * x <= 1.0)) {
        let prefactor = dd_mul(dd_mul((b, 0.0), (x, 0.0)), dd_mul((b + 1.0, 0.0), (x, 0.0)));
        let sum = series_sum_dd(b, x)
            .unwrap_or_else(|| panic!("shape-two beta series did not converge for b={b}, x={x}"));
        let cdf = dd_mul(prefactor, sum);
        let pdf = ((b * x) * (1.0 - x).powf(b - 1.0)) * (b + 1.0);
        if cdf.0 >= f64::MIN_POSITIVE && cdf.0 < 1.0 && pdf > 0.0 && pdf.is_finite() {
            return Some((cdf, pdf));
        }
    }
    if a == 2.0 && !(x < 0.5 && b * x < 0.5) {
        let cdf = if let Some(power) = integer_power(two_sum(1.0, -x), b) {
            let factor = dd_add((1.0, 0.0), dd_mul((b, 0.0), (x, 0.0)));
            let tail = dd_mul(power, factor);
            dd_add((1.0, 0.0), (-tail.0, -tail.1))
        } else {
            tail_cdf_parts(b, x)
        };
        let tail = 1.0 - (cdf.0 + cdf.1);
        let pdf = tail * (b * x) * ((b + 1.0) / ((1.0 - x) * (1.0 + b * x)));
        if cdf.0 >= f64::MIN_POSITIVE && cdf.0 < 1.0 && pdf > 0.0 && pdf.is_finite() {
            return Some((cdf, pdf));
        }
    }
    if b == 2.0 {
        let power = integer_power((x, 0.0), a)?;
        let complement = two_sum(1.0, -x);
        let factor = dd_add((1.0, 0.0), dd_mul((a, 0.0), complement));
        let cdf = dd_mul(power, factor);
        let pdf = dd_mul(
            dd_mul((a * (a + 1.0), 0.0), power),
            dd_div(complement, (x, 0.0)),
        );
        let pdf = pdf.0 + pdf.1;
        if cdf.0 >= f64::MIN_POSITIVE && cdf.0 < 1.0 && pdf > 0.0 && pdf.is_finite() {
            return Some((cdf, pdf));
        }
    }
    None
}

fn integer_power_scalar(value: f64, exponent: f64) -> Option<f64> {
    if exponent != exponent.trunc() || !(1.0..18_446_744_073_709_551_616.0).contains(&exponent) {
        return None;
    }
    let mut exponent = exponent as u64;
    let mut factor = value;
    let mut result = 1.0;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result *= factor;
        }
        exponent >>= 1;
        if exponent != 0 {
            factor *= factor;
        }
    }
    (result >= f64::MIN_POSITIVE && result.is_finite()).then_some(result)
}

fn series_sum(b: f64, x: f64) -> Option<f64> {
    let mut term = 0.5;
    let mut sum = term;
    for k in 1..64 {
        let k = f64::from(k);
        term *= -(b - k) / k * x * (k + 1.0) / (k + 2.0);
        sum += term;
        if term.abs() <= f64::EPSILON * sum.abs() {
            return Some(sum);
        }
    }
    None
}

pub(super) fn fast_cdf_and_pdf(a: f64, b: f64, x: f64) -> Option<(f64, f64)> {
    if a == 2.0 && x < 0.5 && b * x < 0.5 {
        let cdf = (b * x) * ((b + 1.0) * x) * series_sum(b, x)?;
        let pdf = ((b * x) * (1.0 - x).powf(b - 1.0)) * (b + 1.0);
        if (f64::MIN_POSITIVE..1.0).contains(&cdf) && pdf > 0.0 && pdf.is_finite() {
            return Some((cdf, pdf));
        }
    }
    if a == 2.0 && x >= f64::MIN_POSITIVE && !(x < 0.5 && b * x < 0.5) && b <= 1.0 / f64::EPSILON {
        let log_tail = b.mul_add((-x).ln_1p(), (b * x).ln_1p());
        let cdf = -log_tail.exp_m1();
        let tail = 1.0 - cdf;
        let pdf = tail * (b * x) * ((b + 1.0) / ((1.0 - x) * (1.0 + b * x)));
        if (f64::MIN_POSITIVE..1.0).contains(&cdf) && pdf > 0.0 && pdf.is_finite() {
            return Some((cdf, pdf));
        }
    }
    if b == 2.0 {
        let power = integer_power_scalar(x, a).unwrap_or_else(|| x.powf(a));
        let complement = 1.0 - x;
        let cdf = power * (1.0 + a * complement);
        let pdf = (a * power) * ((a + 1.0) * complement / x);
        if (f64::MIN_POSITIVE..1.0).contains(&cdf) && pdf > 0.0 && pdf.is_finite() {
            return Some((cdf, pdf));
        }
    }
    None
}
