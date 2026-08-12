use super::super::*;

// The initial estimate in `inverse_beta_initial` is derived from the
// implementation in the `special` crate, which in turn follows John Burkardt's
// implementation of Applied Statistics Algorithms AS 64 and AS 109:
//
// - https://docs.rs/special/0.8.1/
// - https://people.sc.fsu.edu/~jburkardt/c_src/asa109/asa109.html
// - https://www.jstor.org/stable/2346798
// - https://www.jstor.org/stable/2346887
//
// Copyright 2014–2019 The special Developers
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

pub(super) fn lower_tail_initial(a: f64, b: f64, probability: f64, ln_beta: f64) -> (f64, f64) {
    let log_initial = (probability.ln() + a.ln() + ln_beta) / a;
    let initial = log_initial.exp();
    let initial = if initial == 0.0 {
        0.0
    } else if initial < 1.0 {
        initial
    } else {
        let (mean, _, _, _) = beta_shape_statistics(a, b);
        if mean < 1.0 {
            mean
        } else {
            f64::from_bits(1.0_f64.to_bits() - 1)
        }
    };
    (initial, log_initial)
}

pub(super) fn lower_tail_initial_accurate(
    a: f64,
    probability: f64,
    log_beta: (f64, f64),
) -> (f64, (f64, f64)) {
    let mut logarithm = accurate_ln(probability);
    logarithm = dd_add(logarithm, accurate_ln(a));
    logarithm = dd_add(logarithm, log_beta);
    logarithm = dd_div_f64(logarithm, a);
    (dd_exp(logarithm), logarithm)
}

pub(super) fn lower_tail_initial_from_log_normalizer(
    a: f64,
    probability: f64,
    log_normalizer: (f64, f64),
) -> (f64, (f64, f64)) {
    let logarithm = dd_div_f64(dd_add(accurate_ln(probability), log_normalizer), a);
    (dd_exp(logarithm), logarithm)
}

pub(super) fn inverse_beta_initial(a: f64, b: f64, probability: f64, ln_beta: f64) -> (f64, f64) {
    if a > 1.0 && b > 1.0 && (probability >= 1e-4 || a.min(b) >= STIRLING_MIN) {
        let normal_tail = (-2.0 * probability.ln()).sqrt();
        let normal_quantile = normal_tail
            - (2.30753 + 0.27061 * normal_tail)
                / (1.0 + (0.99229 + 0.04481 * normal_tail) * normal_tail);
        let correction = (normal_quantile * normal_quantile - 3.0) / 6.0;
        let reciprocal_a = 1.0 / (2.0 * a - 1.0);
        let reciprocal_b = 1.0 / (2.0 * b - 1.0);
        let scale = 2.0 / (reciprocal_a + reciprocal_b);
        let w = normal_quantile * (scale + correction).sqrt() / scale
            - (reciprocal_b - reciprocal_a) * (correction + 5.0 / 6.0 - 2.0 / (3.0 * scale));
        let log_ratio = b.ln() - a.ln() + 2.0 * w;
        let initial = if log_ratio > 0.0 {
            let reciprocal = (-log_ratio).exp();
            reciprocal / (1.0 + reciprocal)
        } else {
            1.0 / (1.0 + log_ratio.exp())
        };
        if initial > 0.0 && initial < 1.0 {
            return (initial, f64::NAN);
        }
    }

    lower_tail_initial(a, b, probability, ln_beta)
}
