use super::*;

pub(super) fn upper_gamma_scaled_asymptotic(shape: f64, x: f64) -> Result<f64, BetaFuncError> {
    let mut term = 1.0_f64;
    let mut sum = 1.0_f64;
    for n in 1..=64 {
        term *= (shape - f64::from(n)) / x;
        sum += term;
        if term.abs() <= prec::F64_PREC * sum.abs() {
            return Ok(sum / x);
        }
    }
    Err(BetaFuncError::ConvergenceFailed)
}

pub(super) fn upper_gamma_scaled_continued_fraction(
    shape: f64,
    x: f64,
) -> Result<f64, BetaFuncError> {
    const BIG: f64 = 4_503_599_627_370_496.0;
    const BIG_INVERSE: f64 = 2.220446049250313e-16;

    let mut y = 1.0 - shape;
    let mut z = x + y + 1.0;
    let mut c = 0.0;
    let mut pkm2 = 1.0;
    let mut qkm2 = x;
    let mut pkm1 = x + 1.0;
    let mut qkm1 = z * x;
    let mut result = pkm1 / qkm1;
    for _ in 0..256 {
        y += 1.0;
        z += 2.0;
        c += 1.0;
        let yc = y * c;
        let pk = pkm1 * z - pkm2 * yc;
        let qk = qkm1 * z - qkm2 * yc;

        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        if pk.abs() > BIG {
            pkm2 *= BIG_INVERSE;
            pkm1 *= BIG_INVERSE;
            qkm2 *= BIG_INVERSE;
            qkm1 *= BIG_INVERSE;
        }

        if qk != 0.0 {
            let next = pk / qk;
            let relative_change = ((result - next) / next).abs();
            result = next;
            if relative_change <= 4.0 * prec::F64_PREC {
                return if result > 0.0 && result.is_finite() {
                    Ok(result)
                } else {
                    Err(BetaFuncError::ConvergenceFailed)
                };
            }
        }
    }
    Err(BetaFuncError::ConvergenceFailed)
}

pub(super) fn expm1c(x: f64) -> f64 {
    if x.abs() < 1e-5 {
        1.0 + x * (0.5 + x * (1.0 / 6.0 + x * (1.0 / 24.0 + x / 120.0)))
    } else {
        x.exp_m1() / x
    }
}

pub(super) fn ln_gamma_one_plus_over_x(x: f64) -> f64 {
    if x <= 1e-4 {
        -consts::EULER_MASCHERONI
            + x * (0.8224670334241132
                + x * (-0.40068563438653143
                    + x * (0.27058080842778455
                        + x * (-0.20738555102867398 + x * 0.1695571769974082))))
    } else {
        gamma::ln_gamma(1.0 + x) / x
    }
}

pub(super) fn upper_gamma_scaled_small_shape(shape: f64, x: f64) -> Result<f64, BetaFuncError> {
    let log_x = x.ln();
    let log_gamma_ratio = ln_gamma_one_plus_over_x(shape);
    let difference = log_x - log_gamma_ratio;
    let scaled_difference = shape * difference;
    let mut term = -x / (shape + 1.0);
    let mut sum = term;
    let mut compensation = 0.0_f64;
    for n in 2..=128 {
        let n = f64::from(n);
        term *= (-x / n) * (shape + n - 1.0) / (shape + n);
        let corrected = term - compensation;
        let next = sum + corrected;
        compensation = (next - sum) - corrected;
        sum = next;
        if term.abs() <= prec::F64_PREC * sum.abs() {
            let upper_gamma =
                -difference * expm1c(scaled_difference) - scaled_difference.exp() * sum;
            let result = upper_gamma * (x - scaled_difference).exp();
            return if result > 0.0 && result.is_finite() {
                Ok(result)
            } else {
                Err(BetaFuncError::ConvergenceFailed)
            };
        }
    }
    Err(BetaFuncError::ConvergenceFailed)
}
