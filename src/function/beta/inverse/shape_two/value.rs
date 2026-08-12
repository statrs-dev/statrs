use super::super::super::*;

pub(super) fn log_cdf(a: f64, b: f64, x: f64) -> f64 {
    if b == 2.0 {
        a.mul_add(x.ln(), a.mul_add(1.0 - x, 1.0).ln())
    } else if b * x < 0.5 {
        let mut term = 0.5;
        let mut sum = term;
        for k in 1..64 {
            let k = f64::from(k);
            term *= -(b - k) / k * x * (k + 1.0) / (k + 2.0);
            sum += term;
            if term.abs() <= f64::EPSILON * sum.abs() {
                return (b * (b + 1.0)).ln() + 2.0 * x.ln() + sum.ln();
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
            dd_mul((a, 0.0), accurate_ln_dd((x, 0.0))),
            accurate_ln_dd(factor),
        )
    } else if b * x < 0.5 {
        let sum = series_sum_dd(b, x)
            .unwrap_or_else(|| panic!("shape-two beta series did not converge for b={b}, x={x}"));
        let prefactor = dd_mul((b, 0.0), dd_add((b, 0.0), (1.0, 0.0)));
        dd_add(
            dd_add(
                accurate_ln_dd(prefactor),
                dd_mul((2.0, 0.0), accurate_ln_dd((x, 0.0))),
            ),
            accurate_ln_dd(sum),
        )
    } else {
        let complement = two_sum(1.0, -x);
        let factor = dd_add((1.0, 0.0), dd_mul((b, 0.0), (x, 0.0)));
        let log_tail = dd_add(
            dd_mul((b, 0.0), accurate_ln_dd(complement)),
            accurate_ln_dd(factor),
        );
        let exponential = log_tail.0.exp();
        let exponential_error = exponential * log_tail.1.exp_m1();
        let cdf = if log_tail.0 < -core::f64::consts::LN_2 {
            dd_add((1.0, 0.0), (-exponential, -exponential_error))
        } else {
            two_sum(-log_tail.0.exp_m1(), -exponential_error)
        };
        accurate_ln_dd(cdf)
    }
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

fn integer_power(value: f64, exponent: f64) -> Option<(f64, f64)> {
    if exponent != exponent.trunc() || !(1.0..=(u64::MAX as f64)).contains(&exponent) {
        return None;
    }
    let mut exponent = exponent as u64;
    let mut factor = (value, 0.0);
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
    if a == 2.0 && (b * x < 0.5 || (b == b.trunc() && b <= 64.0 && b * x <= 1.0)) {
        let prefactor = dd_mul(
            dd_mul((b, 0.0), dd_add((b, 0.0), (1.0, 0.0))),
            dd_mul((x, 0.0), (x, 0.0)),
        );
        let sum = series_sum_dd(b, x)
            .unwrap_or_else(|| panic!("shape-two beta series did not converge for b={b}, x={x}"));
        let cdf = dd_mul(prefactor, sum);
        let pdf = b * (b + 1.0) * x * (1.0 - x).powf(b - 1.0);
        if cdf.0 >= f64::MIN_POSITIVE && cdf.0 < 1.0 && pdf > 0.0 && pdf.is_finite() {
            return Some((cdf, pdf));
        }
    }
    if b == 2.0 {
        let power = integer_power(x, a)?;
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
