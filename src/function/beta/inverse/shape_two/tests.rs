use super::*;

#[test]
fn shape_two_scalar_phase_preserves_accurate_rounding() {
    let cases = [
        (200.0, 2.0, 1e-165, 0x3fc2aa4f7f316421_u64),
        (79.0, 2.0, 8.048559608467247e-58, 0x3fc6fca7645f9501_u64),
        (2.0, 732.0, 6.925147302269184e-72, 0x37fba96fc46c76d6_u64),
    ];
    for (a, b, probability, expected) in cases {
        assert_eq!(
            inverse_beta_shape_two(a, b, probability).to_bits(),
            expected
        );
    }
}

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

#[test]
fn shape_two_adjacent_selection_uses_one_error_scale() {
    let probability = f64::from_bits(0x3fb7_8ac0_9e9f_630f);
    assert_eq!(
        inverse_beta_shape_two(2.0, 65.0, probability).to_bits(),
        0x3f7f_81f8_1f81_f820
    );
}

#[test]
fn shape_two_large_shape_rounds_at_unit_boundary() {
    let shape = 18_446_744_073_709_551_616.0;
    assert_eq!(inverse_beta_shape_two(shape, 2.0, 0.5), 1.0);
    assert_eq!(
        inverse_beta_shape_two(2.0, shape, 0.5).to_bits(),
        0x3bfa_da82_5f97_62b2
    );
}

#[test]
fn shape_two_large_second_shape_does_not_overflow() {
    assert_eq!(
        crate::function::beta::inv_beta_reg(2.0, 1e308, 0.5).to_bits(),
        0x000c_1190_8513_0dd9
    );
}
