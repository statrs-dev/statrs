use super::*;

const SHIFT: f64 = 10.400511;
const BETA_SCALE: (f64, f64) = (3.06725258552748459, -1.3709967328337378e-16);
const LOW_TOTAL_MAX: f64 = 8.0;
const IMBALANCE_RATIO: f64 = 8.0;

// Pugh, "An Analysis of the Lanczos Gamma Approximation", Table 8.5 and Eq. 6.14.
// These polynomials are the Horner form of statrs' existing partial-fraction sum.
fn sum(x: f64) -> f64 {
    const NUMERATOR: [f64; 11] = [
        2.48574089138753550e-5,
        2.59434050880906703e-3,
        1.21848070364446573e-1,
        3.39136624401530806,
        6.19452889142209600e1,
        7.75877940545563547e2,
        6.74876752593456695e3,
        4.02538353814263901e4,
        1.57567999493601179e5,
        3.65505352696257003e5,
        3.81540663397352677e5,
    ];
    const DENOMINATOR: [f64; 10] = [
        1.0, 45.0, 870.0, 9450.0, 63273.0, 269325.0, 723680.0, 1172700.0, 1026576.0, 362880.0,
    ];
    let reciprocal = 1.0 / x;
    let numerator = NUMERATOR[..10]
        .iter()
        .rev()
        .fold(NUMERATOR[10], |value, coefficient| {
            value.mul_add(reciprocal, *coefficient)
        });
    let denominator = DENOMINATOR[..9]
        .iter()
        .rev()
        .fold(DENOMINATOR[9], |value, coefficient| {
            value.mul_add(reciprocal, *coefficient)
        });
    numerator / denominator
}

pub(super) fn use_beta_reg_lanczos_power(a: f64, b: f64) -> bool {
    let smaller = a.min(b);
    let larger = a.max(b);
    let total = a + b;
    smaller >= 2.0
        && total <= STIRLING_MIN
        && (total <= LOW_TOTAL_MAX || larger >= IMBALANCE_RATIO * smaller)
}

pub(super) fn beta_reg_lanczos_power(a: f64, b: f64, x: (f64, f64), y: (f64, f64)) -> (f64, f64) {
    debug_assert!(use_beta_reg_lanczos_power(a, b));

    let total = two_sum(a, b);
    let shifted_a = dd_add((a, 0.0), (SHIFT, 0.0));
    let shifted_b = dd_add((b, 0.0), (SHIFT, 0.0));
    let shifted_total = dd_add(total, (SHIFT, 0.0));
    let delta_a = dd_div(
        dd_add(dd_mul(shifted_total, x), (-shifted_a.0, -shifted_a.1)),
        shifted_a,
    );
    let delta_b = dd_div(
        dd_add(dd_mul(shifted_total, y), (-shifted_b.0, -shifted_b.1)),
        shifted_b,
    );
    let log_a = (delta_a.0.ln_1p(), delta_a.1 / (1.0 + delta_a.0));
    let log_b = (delta_b.0.ln_1p(), delta_b.1 / (1.0 + delta_b.0));
    let exponent = dd_add(dd_mul((a, 0.0), log_a), dd_mul((b, 0.0), log_b));
    let shifted_scale = dd_div(dd_mul(shifted_a, shifted_b), shifted_total);
    let lanczos_scale = sum(total.0 + total.1) / (sum(a) * sum(b));
    let sqrt_scale = shifted_scale.0.sqrt();
    let sqrt_scale = (sqrt_scale, shifted_scale.1 / (2.0 * sqrt_scale));
    let scale = dd_div(dd_mul((lanczos_scale, 0.0), sqrt_scale), BETA_SCALE);
    dd_mul(scale, (dd_exp(exponent), 0.0))
}
