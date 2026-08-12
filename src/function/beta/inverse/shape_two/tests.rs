use super::*;

#[test]
fn shape_two_log_values_match_references() {
    let cases: [(f64, f64, f64, f64); 3] = [
        (200.0_f64, 2.0, 0.48970503636005447, -138.15510557964274),
        (200.0, 2.0, 0.14582246504394994, -379.9265403440175),
        (2.0, 5.0, 0.18180347131894917, -1.203972804325936),
    ];
    for (a, b, x, expected) in cases {
        let actual = log_cdf_parts(a, b, x);
        assert!(
            (actual.0 + actual.1).to_bits().abs_diff(expected.to_bits()) <= 2,
            "a={a} b={b} actual={:?} expected={expected:?}",
            actual.0 + actual.1
        );
    }
}

#[test]
fn shape_two_subnormal_probabilities_are_monotone() {
    for (a, b) in [(2.0, 200.0), (200.0, 2.0)] {
        let values =
            [1_u64, 2, 3, 4].map(|bits| inverse_beta_shape_two(a, b, f64::from_bits(bits)));
        assert!(values.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(values[0] > 0.0);
    }
}
