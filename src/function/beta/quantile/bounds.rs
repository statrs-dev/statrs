use super::super::*;

const LOG_EXTREME_MARGIN: f64 = 16.0 * core::f64::consts::LN_2;
const LOG_CENTRAL_MARGIN: f64 = 4.0 * core::f64::consts::LN_2;

type Double = (f64, f64);

struct ScaledParts {
    scale: f64,
    a: Double,
    b: Double,
    sum: Double,
}

fn scaled_parts(a: f64, b: f64) -> ScaledParts {
    let scale = a.max(b);
    let scaled_a = dd_div_f64((a, 0.0), scale);
    let scaled_b = dd_div_f64((b, 0.0), scale);
    ScaledParts {
        scale,
        a: scaled_a,
        b: scaled_b,
        sum: dd_add(scaled_a, scaled_b),
    }
}

fn log_sum(a: f64, b: f64) -> (f64, f64) {
    let parts = scaled_parts(a, b);
    dd_add(accurate_ln(parts.scale), accurate_ln_dd(parts.sum))
}

fn log_variance(a: f64, b: f64) -> (f64, f64) {
    let sum = log_sum(a, b);
    let mut result = dd_add(accurate_ln(a), accurate_ln(b));
    result = dd_add(result, dd_mul((-3.0, 0.0), sum));
    let reciprocal_sum = dd_exp((-sum.0, -sum.1));
    let correction = accurate_ln_one_plus_dd((reciprocal_sum, 0.0));
    dd_add(result, (-correction.0, -correction.1))
}

fn log_cantelli_bound(variance: (f64, f64), distance: (f64, f64)) -> f64 {
    if distance.0 + distance.1 <= 0.0 {
        return 0.0;
    }
    let ratio = dd_add(
        dd_mul((2.0, 0.0), accurate_ln_dd(distance)),
        (-variance.0, -variance.1),
    );
    let ratio = ratio.0 + ratio.1;
    if ratio > 0.0 {
        -ratio - (-ratio).exp().ln_1p()
    } else {
        -ratio.exp().ln_1p()
    }
}

fn relative_logs(a: f64, b: f64, point: (f64, f64)) -> ((f64, f64), (f64, f64), f64) {
    let parts = scaled_parts(a, b);
    let residual = dd_add(dd_mul(point, parts.sum), (-parts.a.0, -parts.a.1));
    let relative_a = dd_div(residual, parts.a);
    let relative_b = dd_div((-residual.0, -residual.1), parts.b);
    let log_a = accurate_ln_one_plus_dd(relative_a);
    let log_b = accurate_ln_one_plus_dd(relative_b);
    let centered_a = dd_add(log_a, (-relative_a.0, -relative_a.1));
    let centered_b = dd_add(log_b, (-relative_b.0, -relative_b.1));
    let exponent = dd_add(dd_mul(parts.a, centered_a), dd_mul(parts.b, centered_b));
    let exponent = exponent.0 + exponent.1;
    let exponent = if exponent < -f64::MAX / parts.scale {
        f64::NEG_INFINITY
    } else {
        parts.scale * exponent
    };
    (log_a, log_b, exponent)
}

// Beta = Ga / (Ga + Gb); Markov applied to exp(-t((1-x)Ga-xGb)) gives the KL/Chernoff tail bound.
fn log_chernoff_bound(a: f64, b: f64, point: (f64, f64)) -> f64 {
    relative_logs(a, b, point).2
}

fn upper_bound_is_below(bound: f64, target: (f64, f64), margin: f64) -> bool {
    let proof = dd_add(target, (-bound - margin, 0.0));
    proof.0 + proof.1 > 0.0
}

fn log_density(a: f64, b: f64, point: (f64, f64)) -> f64 {
    let sum = log_sum(a, b);
    let mut center = dd_mul((1.5, 0.0), sum);
    center = dd_add(center, dd_mul((-0.5, 0.0), accurate_ln(a)));
    center = dd_add(center, dd_mul((-0.5, 0.0), accurate_ln(b)));
    center = dd_add(center, (-consts::LN_SQRT_2PI, 3.8782941580672414e-17));
    center = dd_add(
        center,
        (
            -stirling_correction(a) - stirling_correction(b)
                + stirling_correction_log(sum.0 + sum.1),
            0.0,
        ),
    );
    let (log_a, log_b, exponent) = relative_logs(a, b, point);
    let result = dd_add(center, (exponent, 0.0));
    let result = dd_add(result, (-log_a.0, -log_a.1));
    let result = dd_add(result, (-log_b.0, -log_b.1));
    result.0 + result.1
}

fn mode(a: f64, b: f64) -> (f64, f64) {
    let parts = scaled_parts(a, b);
    let reciprocal_scale = 1.0 / parts.scale;
    dd_div(
        dd_add(parts.a, (-reciprocal_scale, 0.0)),
        dd_add(parts.sum, (-2.0 * reciprocal_scale, 0.0)),
    )
}

fn lower_bound_is_above(
    a: f64,
    b: f64,
    point: (f64, f64),
    target: (f64, f64),
    variance: (f64, f64),
) -> bool {
    let mode_distance = dd_add(mode(a, b), (-point.0, -point.1));
    if mode_distance.0 + mode_distance.1 < 0.0 {
        return false;
    }
    let width = dd_exp(dd_mul((0.5, 0.0), variance)).min(0.5 * (point.0 + point.1));
    if width <= 0.0 {
        return false;
    }
    let left = dd_add(point, (-width, 0.0));
    let bound = width.ln() + log_density(a, b, left) - LOG_CENTRAL_MARGIN;
    let proof = dd_add((bound, 0.0), (-target.0, -target.1));
    proof.0 + proof.1 > 0.0
}

pub(super) fn extreme_ratio_cell_is_certified(
    a: f64,
    b: f64,
    probability: f64,
    mean: (f64, f64),
    candidate: f64,
) -> bool {
    let variance = log_variance(a, b);
    if candidate > 0.0 {
        let previous = f64::from_bits(candidate.to_bits() - 1);
        let midpoint = dd_mul(dd_add((candidate, 0.0), (previous, 0.0)), (0.5, 0.0));
        let distance = dd_add(mean, (-midpoint.0, -midpoint.1));
        let bound = log_cantelli_bound(variance, distance);
        if !upper_bound_is_below(bound, accurate_ln(probability), LOG_EXTREME_MARGIN) {
            return false;
        }
    }
    if candidate < 1.0 {
        let next = f64::from_bits(candidate.to_bits() + 1);
        let midpoint = dd_mul(dd_add((candidate, 0.0), (next, 0.0)), (0.5, 0.0));
        let distance = dd_add(midpoint, (-mean.0, -mean.1));
        let bound = log_cantelli_bound(variance, distance);
        if !upper_bound_is_below(
            bound,
            accurate_ln_one_minus(probability),
            LOG_EXTREME_MARGIN,
        ) {
            return false;
        }
    }
    true
}

pub(super) fn central_cell_is_certified(
    a: f64,
    b: f64,
    probability: f64,
    mean: (f64, f64),
    candidate: f64,
) -> bool {
    let variance = log_variance(a, b);
    let log_probability = accurate_ln(probability);
    let log_complement = accurate_ln_one_minus(probability);
    if candidate > 0.0 {
        let previous = f64::from_bits(candidate.to_bits() - 1);
        let midpoint = dd_mul(dd_add((candidate, 0.0), (previous, 0.0)), (0.5, 0.0));
        let distance = dd_add(mean, (-midpoint.0, -midpoint.1));
        let proven = if distance.0 + distance.1 > 0.0 {
            upper_bound_is_below(
                log_chernoff_bound(a, b, midpoint),
                log_probability,
                LOG_CENTRAL_MARGIN,
            )
        } else {
            let reflected = dd_add((1.0, 0.0), (-midpoint.0, -midpoint.1));
            lower_bound_is_above(b, a, reflected, log_complement, variance)
        };
        if !proven {
            return false;
        }
    }
    if candidate < 1.0 {
        let next = f64::from_bits(candidate.to_bits() + 1);
        let midpoint = dd_mul(dd_add((candidate, 0.0), (next, 0.0)), (0.5, 0.0));
        let distance = dd_add(midpoint, (-mean.0, -mean.1));
        let proven = if distance.0 + distance.1 > 0.0 {
            let reflected = dd_add((1.0, 0.0), (-midpoint.0, -midpoint.1));
            upper_bound_is_below(
                log_chernoff_bound(b, a, reflected),
                log_complement,
                LOG_CENTRAL_MARGIN,
            )
        } else {
            lower_bound_is_above(a, b, midpoint, log_probability, variance)
        };
        if !proven {
            return false;
        }
    }
    true
}
