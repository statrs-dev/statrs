use super::*;

pub(super) fn beta_continued_fraction(a: f64, b: f64, x: f64) -> Result<f64, BetaFuncError> {
    let y = 1.0 - x;
    let tiny = 16.0 * f64::MIN_POSITIVE;
    let mut fraction = a * (a * y - b * x + 1.0) / (a + 1.0);
    if fraction == 0.0 {
        fraction = tiny;
    }
    let mut c = fraction;
    let mut d = 0.0;

    for m in 1..=MAX_BETA_REG_ITERATIONS {
        let m = f64::from(m);
        let denominator = a + 2.0 * m - 1.0;
        let numerator =
            (m * (a + m - 1.0) / denominator) * ((a + b + m - 1.0) / denominator) * (b - m) * x * x;
        let denominator_term = m
            + m * (b - m) * x / denominator
            + (a + m) * (a * y - b * x + 1.0 + m * (2.0 - x)) / (a + 2.0 * m + 1.0);

        d = denominator_term + numerator * d;
        if d == 0.0 {
            d = tiny;
        }
        c = denominator_term + numerator / c;
        if c == 0.0 {
            c = tiny;
        }
        d = 1.0 / d;
        let delta = c * d;
        fraction *= delta;

        if (delta - 1.0).abs() <= prec::F64_PREC {
            return Ok(fraction);
        }
    }

    Err(BetaFuncError::ConvergenceFailed)
}

pub(super) fn beta_continued_fraction_dd(
    a: f64,
    b: f64,
    x: (f64, f64),
) -> Result<(f64, f64), BetaFuncError> {
    let y = dd_add((1.0, 0.0), (-x.0, -x.1));
    let mut residual = dd_mul((a, 0.0), y);
    residual = dd_add(residual, dd_mul((-b, 0.0), x));
    residual = dd_add(residual, (1.0, 0.0));
    let mut fraction = dd_div_f64(dd_mul((a, 0.0), residual), a + 1.0);
    let mut c = fraction;
    let mut d = (0.0, 0.0);

    for integer in 1..=MAX_BETA_REG_ITERATIONS {
        let m = f64::from(integer);
        let denominator = a + 2.0 * m - 1.0;
        let mut numerator = dd_div_f64(dd_mul((m, 0.0), (a + m - 1.0, 0.0)), denominator);
        let a_plus_b_plus_m_minus_one = dd_add((b, 0.0), dd_add((a, 0.0), (m - 1.0, 0.0)));
        numerator = dd_mul(
            numerator,
            dd_div_f64(dd_mul(a_plus_b_plus_m_minus_one, x), denominator),
        );
        let b_minus_m = dd_add((b, 0.0), (-m, 0.0));
        numerator = dd_mul(numerator, dd_mul(b_minus_m, x));

        let first = dd_div_f64(dd_mul((m, 0.0), dd_mul(b_minus_m, x)), denominator);
        let inner = dd_add(residual, dd_mul((m, 0.0), dd_add((2.0, 0.0), (-x.0, -x.1))));
        let second = dd_div_f64(dd_mul((a + m, 0.0), inner), a + 2.0 * m + 1.0);
        let denominator_term = dd_add((m, 0.0), dd_add(first, second));

        d = dd_div((1.0, 0.0), dd_add(denominator_term, dd_mul(numerator, d)));
        c = dd_add(denominator_term, dd_div(numerator, c));
        let delta = dd_mul(c, d);
        fraction = dd_mul(fraction, delta);
        let convergence = dd_add(delta, (-1.0, 0.0));
        if (convergence.0 + convergence.1).abs() <= f64::EPSILON {
            return Ok(fraction);
        }
    }

    Err(BetaFuncError::ConvergenceFailed)
}

pub(super) fn selected_beta_continued_fraction(
    a: f64,
    b: f64,
    x: f64,
) -> Result<(f64, f64), BetaFuncError> {
    if x <= f64::EPSILON {
        beta_continued_fraction_dd(a, b, (x, 0.0))
    } else {
        beta_continued_fraction(a, b, x).map(|fraction| (fraction, 0.0))
    }
}

pub(super) fn use_exact_complement_continued_fraction(
    a: f64,
    b: f64,
    symm_transform: bool,
) -> bool {
    symm_transform && a >= 1.0 && b >= 2.0 * (a + 1.0)
}

pub(super) fn beta_fraction_for_transformed_tail(
    a: f64,
    b: f64,
    x: f64,
    transformed_a: f64,
    transformed_b: f64,
    transformed_x: f64,
    symm_transform: bool,
) -> Result<(f64, f64), BetaFuncError> {
    if use_exact_complement_continued_fraction(a, b, symm_transform) {
        beta_continued_fraction_dd(transformed_a, transformed_b, two_sum(1.0, -x))
    } else {
        selected_beta_continued_fraction(transformed_a, transformed_b, transformed_x)
    }
}
