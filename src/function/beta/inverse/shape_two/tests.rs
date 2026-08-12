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

#[test]
fn real_shape_two_inputs_match_500_digit_references() {
    let cases = [
        (2.0, 5.5, 0.3, 0x3fc5_7cb4_a4fa_6544_u64),
        (5.5, 2.0, 0.3, 0x3fe5_4170_6a98_75f7),
        (2.0, 65.5, 0.3, 0x3f90_e2ac_2408_d260),
        (65.5, 2.0, 0.3, 0x3fee_d6c0_af00_418b),
        (2.0, 200.5, 0.3, 0x3f76_4d12_9f5f_2879),
        (200.5, 2.0, 0.3, 0x3fef_9d2f_de46_a3e6),
        (2.0, 200.5, 0.999_999_999, 0x3fbc_bed1_fc37_f6e0),
        (200.5, 2.0, 0.999_999_999, 0x3fef_ffff_888c_10e7),
        (2.0, 1_000_000.5, 1e-60, 0x387e_13b4_0c5d_6bc5),
        (1_000_000.5, 2.0, 1e-60, 0x3fef_fed3_dd80_3e82),
        (2.0, 1e308, 0.5, 0x000c_1190_8513_0dd9),
        (1e308, 2.0, 0.5, 0x3ff0_0000_0000_0000),
    ];
    for (a, b, probability, expected) in cases {
        assert_eq!(
            crate::function::beta::inv_beta_reg(a, b, probability).to_bits(),
            expected,
            "a={a} b={b} probability={probability}"
        );
    }
}

#[test]
fn real_shape_two_subnormal_quantiles_match_500_digit_references() {
    let cases = [
        (0x2007_9905_9deb_7818, 0x000f_77d8_b988_842e),
        (0x2007_9905_9deb_7819, 0x000f_77d8_b988_842f),
        (0x2007_9905_9deb_781a, 0x000f_77d8_b988_8431),
    ];
    let values = cases.map(|(probability, expected)| {
        let actual = crate::function::beta::inv_beta_reg(0.5, 2.0, f64::from_bits(probability));
        assert_eq!(actual.to_bits(), expected);
        actual
    });
    assert!(values.windows(2).all(|pair| pair[0] <= pair[1]));
}

#[test]
fn real_shape_two_zero_cell_is_rounded_and_monotone() {
    let cases = [(0x1668_7e92_154e_f7ac, 0_u64), (0x1e6c_cb05_3660_8d61, 1)];
    let values = cases.map(|(probability, expected)| {
        let actual = crate::function::beta::inv_beta_reg(0.5, 2.0, f64::from_bits(probability));
        assert_eq!(actual.to_bits(), expected);
        actual
    });
    assert!(values[0] <= values[1]);

    let tiny_shape = f64::from_bits(1);
    for probability in [0.1, 0.5, 0.9, 0.999_999_999] {
        assert_eq!(
            crate::function::beta::inv_beta_reg(tiny_shape, 2.0, probability),
            0.0
        );
    }
}

#[test]
fn real_shape_two_subnormal_cells_match_550_digit_references() {
    let cases = [
        (0x3fdf_ffff_ffff_ffff, 0x1e6d_64d5_1e0d_b31c, 0x2_u64),
        (0x3fdf_ffff_ffff_ffff, 0x1e87_9f3c_f0a9_fc5f, 0x10),
        (0x3fdf_ffff_ffff_ffff, 0x1e91_1a46_0cc5_8ef7, 0x21),
        (0x3fe0_0000_0000_0000, 0x1e72_f942_2c23_c47c, 0x2),
        (0x3fe0_0000_0000_0000, 0x1e88_5f42_f152_1410, 0x10),
        (0x3fe0_0000_0000_0000, 0x1e90_d663_a9ca_a39a, 0x20),
        (0x3fe0_0000_0000_0001, 0x1e6d_64d5_1e0d_af1b, 0x2),
        (0x3fe0_0000_0000_0001, 0x1e89_198c_8b83_0581, 0x11),
        (0x3fe0_0000_0000_0001, 0x1e90_916b_2b5f_fe37, 0x1f),
    ];
    for (a, probability, expected) in cases {
        assert_eq!(
            crate::function::beta::inv_beta_reg(
                f64::from_bits(a),
                2.0,
                f64::from_bits(probability),
            )
            .to_bits(),
            expected,
            "a={a:#018x} probability={probability:#018x}"
        );
    }
}

#[test]
fn subnormal_first_shape_rounds_to_zero_above_specialization_gate() {
    let probability = f64::from_bits(1.0_f64.to_bits() - 1);
    for shape_bits in [2_u64, 3, 0x10, 0x100, 0x0010_0000_0000_0000] {
        assert_eq!(
            crate::function::beta::inv_beta_reg(f64::from_bits(shape_bits), 2.0, probability),
            0.0,
            "shape_bits={shape_bits:#018x}"
        );
    }
}

#[test]
fn unit_first_shape_rounds_subnormal_quantiles_upward() {
    for probability_bits in 1_u64..=1024 {
        assert_eq!(
            crate::function::beta::inv_beta_reg(1.0, 2.0, f64::from_bits(probability_bits),)
                .to_bits(),
            probability_bits.div_ceil(2),
            "probability_bits={probability_bits:#018x}"
        );
    }
}
