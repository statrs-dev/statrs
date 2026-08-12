use super::*;
use crate::prec;
use core::f64::consts as f64_consts;
const MODULE_RELATIVE_ACC: f64 = 1e-14;
const INVERSE_REFERENCE_MAX_ULPS: u64 = 4;

// Tests named `*_500_digit_*` use exact decimal encodings of the input `f64`s
// with `cpp_dec_float<500>` and one final round to binary64. Inverse references
// were independently checked with mpmath at 550 decimal digits.

fn beta_assert_relative_eq(a: f64, b: f64) {
    prec::assert_relative_eq!(
        a,
        b,
        epsilon = MODULE_EPS,
        max_relative = MODULE_RELATIVE_ACC
    );
}

fn beta_assert_abs_diff_eq(a: f64, b: f64) {
    prec::assert_abs_diff_eq!(a, b, epsilon = MODULE_EPS);
}

#[test]
fn test_ln_beta() {
    beta_assert_relative_eq(ln_beta(0.5, 0.5), 1.144729885849400174144);
    beta_assert_relative_eq(ln_beta(1.0, 0.5), f64_consts::LN_2);
    beta_assert_relative_eq(ln_beta(2.5, 0.5), 0.163900632837673937284);
    beta_assert_relative_eq(ln_beta(0.5, 1.0), f64_consts::LN_2);
    beta_assert_relative_eq(ln_beta(1.0, 1.0), 0.0);
    beta_assert_relative_eq(ln_beta(2.5, 1.0), -0.9162907318741550651835);
    beta_assert_relative_eq(ln_beta(0.5, 2.5), 0.163900632837673937284);
    beta_assert_relative_eq(ln_beta(1.0, 2.5), -0.9162907318741550651835);
    beta_assert_relative_eq(ln_beta(2.5, 2.5), -2.608688089402107300388);
}

#[test]
#[should_panic]
fn test_ln_beta_a_lte_0() {
    ln_beta(0.0, 0.5);
}

#[test]
#[should_panic]
fn test_ln_beta_b_lte_0() {
    ln_beta(0.5, 0.0);
}

#[test]
fn test_checked_ln_beta_a_lte_0() {
    assert!(checked_ln_beta(0.0, 0.5).is_err());
}

#[test]
fn test_checked_ln_beta_b_lte_0() {
    assert!(checked_ln_beta(0.5, 0.0).is_err());
}

#[test]
#[should_panic]
fn test_beta_a_lte_0() {
    beta(0.0, 0.5);
}

#[test]
#[should_panic]
fn test_beta_b_lte_0() {
    beta(0.5, 0.0);
}

#[test]
fn test_checked_beta_a_lte_0() {
    assert!(checked_beta(0.0, 0.5).is_err());
}

#[test]
fn test_checked_beta_b_lte_0() {
    assert!(checked_beta(0.5, 0.0).is_err());
}

#[test]
fn test_beta() {
    beta_assert_relative_eq(beta(0.5, 0.5), f64_consts::PI);
    beta_assert_relative_eq(beta(1.0, 0.5), 2.0);
    beta_assert_relative_eq(beta(2.5, 0.5), 1.17809724509617246442);
    beta_assert_relative_eq(beta(0.5, 1.0), 2.0);
    beta_assert_relative_eq(beta(1.0, 1.0), 1.0);
    beta_assert_relative_eq(beta(2.5, 1.0), 0.4);
    beta_assert_relative_eq(beta(0.5, 2.5), 1.17809724509617246442);
    beta_assert_relative_eq(beta(1.0, 2.5), 0.4);
    beta_assert_relative_eq(beta(2.5, 2.5), 0.073631077818510779026);
}

#[test]
fn test_beta_inc() {
    beta_assert_relative_eq(beta_inc(0.5, 0.5, 0.5), f64_consts::FRAC_PI_2);
    beta_assert_relative_eq(beta_inc(0.5, 0.5, 1.0), f64_consts::PI);
    beta_assert_relative_eq(beta_inc(1.0, 0.5, 0.5), 0.5857864376269049511983);
    beta_assert_relative_eq(beta_inc(1.0, 0.5, 1.0), 2.0);
    beta_assert_relative_eq(beta_inc(2.5, 0.5, 0.5), 0.0890486225480862322117);
    beta_assert_relative_eq(beta_inc(2.5, 0.5, 1.0), 1.17809724509617246442);
    beta_assert_relative_eq(beta_inc(0.5, 1.0, 0.5), f64_consts::SQRT_2);
    beta_assert_relative_eq(beta_inc(0.5, 1.0, 1.0), 2.0);
    beta_assert_relative_eq(beta_inc(1.0, 1.0, 0.5), 0.5);
    beta_assert_relative_eq(beta_inc(1.0, 1.0, 1.0), 1.0);
    beta_assert_relative_eq(beta_inc(2.5, 1.0, 0.5), 0.0707106781186547524401);
    beta_assert_relative_eq(beta_inc(2.5, 1.0, 1.0), 0.4);
    beta_assert_relative_eq(beta_inc(0.5, 2.5, 0.5), 1.08904862254808623221);
    beta_assert_relative_eq(beta_inc(0.5, 2.5, 1.0), 1.17809724509617246442);
    beta_assert_relative_eq(beta_inc(1.0, 2.5, 0.5), 0.32928932188134524756);
    beta_assert_relative_eq(beta_inc(1.0, 2.5, 1.0), 0.4);
    beta_assert_relative_eq(beta_inc(2.5, 2.5, 0.5), 0.03681553890925538951323);
    beta_assert_relative_eq(beta_inc(2.5, 2.5, 1.0), 0.073631077818510779026);
}

#[test]
#[should_panic]
fn test_beta_inc_a_lte_0() {
    beta_inc(0.0, 1.0, 1.0);
}

#[test]
#[should_panic]
fn test_beta_inc_b_lte_0() {
    beta_inc(1.0, 0.0, 1.0);
}

#[test]
#[should_panic]
fn test_beta_inc_x_lt_0() {
    beta_inc(1.0, 1.0, -1.0);
}

#[test]
#[should_panic]
fn test_beta_inc_x_gt_1() {
    beta_inc(1.0, 1.0, 2.0);
}

#[test]
fn test_checked_beta_inc_a_lte_0() {
    assert!(checked_beta_inc(0.0, 1.0, 1.0).is_err());
}

#[test]
fn test_checked_beta_inc_b_lte_0() {
    assert!(checked_beta_inc(1.0, 0.0, 1.0).is_err());
}

#[test]
fn test_checked_beta_inc_x_lt_0() {
    assert!(checked_beta_inc(1.0, 1.0, -1.0).is_err());
}

#[test]
fn test_checked_beta_inc_x_gt_1() {
    assert!(checked_beta_inc(1.0, 1.0, 2.0).is_err());
}

#[test]
fn test_beta_reg() {
    beta_assert_abs_diff_eq(beta_reg(0.5, 0.5, 0.5), 0.5);
    assert_eq!(beta_reg(0.5, 0.5, 1.0), 1.0);
    beta_assert_abs_diff_eq(beta_reg(1.0, 0.5, 0.5), 0.292893218813452475599);
    assert_eq!(beta_reg(1.0, 0.5, 1.0), 1.0);
    beta_assert_abs_diff_eq(beta_reg(2.5, 0.5, 0.5), 0.07558681842161243795);
    assert_eq!(beta_reg(2.5, 0.5, 1.0), 1.0);
    beta_assert_abs_diff_eq(beta_reg(0.5, 1.0, 0.5), f64_consts::FRAC_1_SQRT_2);
    assert_eq!(beta_reg(0.5, 1.0, 1.0), 1.0);
    beta_assert_abs_diff_eq(beta_reg(1.0, 1.0, 0.5), 0.5);
    assert_eq!(beta_reg(1.0, 1.0, 1.0), 1.0);
    beta_assert_abs_diff_eq(beta_reg(2.5, 1.0, 0.5), 0.1767766952966368811);
    assert_eq!(beta_reg(2.5, 1.0, 1.0), 1.0);
    beta_assert_abs_diff_eq(beta_reg(0.5, 2.5, 0.5), 0.92441318157838756205);
    assert_eq!(beta_reg(0.5, 2.5, 1.0), 1.0);
    beta_assert_abs_diff_eq(beta_reg(1.0, 2.5, 0.5), 0.8232233047033631189);
    assert_eq!(beta_reg(1.0, 2.5, 1.0), 1.0);
    beta_assert_abs_diff_eq(beta_reg(2.5, 2.5, 0.5), 0.5);
    assert_eq!(beta_reg(2.5, 2.5, 1.0), 1.0);
}

#[test]
fn test_beta_reg_large_parameters_against_reference() {
    let cases = [
        (1e6, 2e6, 0.333, 0.11032283951664962),
        (1e6, 2e6, 1.0 / 3.0, 0.5000542891707268),
        (1e6, 2e6, 0.334, 0.9928335645421132),
        (1e8, 2e8, 0.3333, 0.11033439854811466),
        (1e8, 2e8, 1.0 / 3.0, 0.5000054289165304),
        (1e8, 2e8, 0.3334, 0.992845709515461),
        (1e5, 1e5, 0.49, 1.8571347290404196e-19),
        (1e5, 1e5, 0.499, 0.18554674455755675),
        (1e5, 1e5, 0.501, 0.8144532554424433),
        (40.0, 32.0, 1e-8, 1.2676414050441584e-300),
        (32.0, 40.0, 1e-8, 1.5845516362868252e-236),
        (0.1, 1e8, 1e-8, 0.9758726562930068),
        (0.1, 1e8, 1e-9, 0.8275517592836537),
        (2.0, 1e8, 1e-8, 0.2642411213359098),
        (10.0, 1e8, 1e-7, 0.5420704043826821),
        (1e13, 9.9e14, 0.01, 0.5000000414451727),
        (
            1.098252731340299,
            1.780042655540735e17,
            5.235783704840033e-17,
            0.999881646675342,
        ),
        (
            7_627_209.761,
            11.3319,
            0.9999105965110135,
            1.6790000011611638e-274,
        ),
        (99_999.0, 11.3319, 0.9998, 0.013667998876668642),
        (100_001.0, 11.3319, 0.9998, 0.013665136770782414),
        (100_000.0, 10.0, 0.992653308338289, 1.0000000000029653e-300),
    ];

    for (a, b, x, expected) in cases {
        let actual = beta_reg(a, b, x);
        let error = (actual - expected).abs();
        let tolerance = 5e-12 * expected.max(1e-300);
        assert!(
            error <= tolerance,
            "beta_reg({a}, {b}, {x}) = {actual}, expected {expected}, error {error}"
        );
    }
}

#[test]
fn test_beta_reg_large_symmetric_adjacent_against_500_digit_references() {
    // cpp_dec_float<500>, with each f64 input converted from its exact binary ratio.
    let lower = f64::from_bits(0.5_f64.to_bits() - 1);
    let upper = f64::from_bits(0.5_f64.to_bits() + 1);
    let cases = [
        (1e2, 0x3fdffffffffffff5_u64, 0x3fe000000000000b_u64),
        (1e3, 0x3fdfffffffffffdc, 0x3fe0000000000024),
        (1e4, 0x3fdfffffffffff8f, 0x3fe0000000000071),
        (1e5, 0x3fdffffffffffe9b, 0x3fe0000000000165),
        (1e6, 0x3fdffffffffffb98, 0x3fe0000000000468),
        (1e7, 0x3fdffffffffff210, 0x3fe0000000000df0),
        (1e8, 0x3fdfffffffffd3ec, 0x3fe0000000002c14),
    ];
    for (shape, lower_expected, upper_expected) in cases {
        for (x, expected) in [(lower, lower_expected), (upper, upper_expected)] {
            let actual = beta_reg(shape, shape, x).to_bits();
            assert!(
                actual.abs_diff(expected) <= 4,
                "shape={shape:?}, x={x:?}, actual={actual:#018x}, expected={expected:#018x}"
            );
        }
    }
}

#[test]
fn test_beta_reg_symmetric_central_boundary_against_500_digit_references() {
    // mpmath at 550 digits using exact f64 ratios and the equivalent gamma/hyp2f1 identity.
    let cases = [
        (
            100.0,
            0x3fddbcbcf5c0139f_u64,
            0x3fc44eca5b83b728_u64,
            0x3fe121a1851ff630_u64,
            0x3feaec4d691f1233_u64,
        ),
        (
            100.125,
            0x3fddbd198e18a036,
            0x3fc44eca5f98d22f,
            0x3fe1217338f3afe5,
            0x3feaec4d6819cb74,
        ),
        (
            1e6,
            0x3fdffa3516f00033,
            0x3fc44ed0bb7cb51c,
            0x3fe002e57487ffe6,
            0x3feaec4bd120d163,
        ),
        (
            1e12,
            0x3fdffffe845ffbe8,
            0x3fc44ed0bb87ad45,
            0x3fe00000bdd0020c,
            0x3feaec4bd11e14af,
        ),
    ];
    for (shape, lower_x, lower_expected, upper_x, upper_expected) in cases {
        assert_eq!(beta_reg(shape, shape, 0.5), 0.5);
        for (x, expected) in [(lower_x, lower_expected), (upper_x, upper_expected)] {
            let actual = beta_reg(shape, shape, f64::from_bits(x)).to_bits();
            assert!(
                actual.abs_diff(expected) <= 4,
                "shape={shape:?}, x={x:#018x}, actual={actual:#018x}, expected={expected:#018x}"
            );
        }
    }
}

#[test]
fn test_beta_reg_extreme_ratio_central_value_against_reference() {
    let cases: [(f64, f64, f64, f64); 2] = [
        (
            1.2e7,
            1.2000000000000001e307,
            9.999999999999999e-301,
            0.50003838823874907,
        ),
        (1.2e7, 1e308, 1.2e-301, 0.50003838823881181),
    ];
    for (a, b, x, expected) in cases {
        let actual = beta_reg(a, b, x);
        assert!(
            actual.to_bits().abs_diff(expected.to_bits()) <= 1024,
            "beta_reg({a}, {b}, {x}) = {actual}, expected {expected}"
        );
    }
}

#[test]
fn test_beta_reg_overflowing_shape_sum() {
    let lower = f64::from_bits(0.5_f64.to_bits() - 1);
    let upper = f64::from_bits(0.5_f64.to_bits() + 1);
    assert_eq!(beta_reg(1e308, 1e308, lower), 0.0);
    assert_eq!(beta_reg(1e308, 1e308, 0.5), 0.5);
    assert_eq!(beta_reg(1e308, 1e308, upper), 1.0);
    let actual = checked_ln_beta(1e308, 1e308).unwrap();
    assert!(actual.is_finite());
    assert!((actual / 1e308 + 2.0 * core::f64::consts::LN_2).abs() <= 2e-15);
    let expected = -2.0007184997951635e301;
    let actual = checked_ln_beta(f64::MAX, 1e300).unwrap();
    assert!(((actual - expected) / expected).abs() <= 3e-10);

    let mean = f64::from_bits(0x3fe5555555555555);
    assert_eq!(beta_reg(1e308, 5e307, mean), 0.0);
    assert_eq!(
        beta_reg(1e308, 5e307, f64::from_bits(mean.to_bits() + 1)),
        1.0
    );
}

#[test]
fn test_beta_reg_algorithm_boundaries_against_reference() {
    let cases = [
        (39_999_999.0, 79_999_999.0, 0.33335, 0.6507629787874431),
        (40_000_001.0, 80_000_001.0, 0.33335, 0.6507151999304125),
        (29_999_999.0, 270_000_001.0, 0.10001, 0.7182251069092127),
        (30_000_001.0, 269_999_999.0, 0.10001, 0.7180951316317142),
        (1e8, 2e8, 0.33328635138267637, 0.042150859881784875),
        (1e8, 2e8, 0.33328603712606697, 0.04112293252416181),
    ];

    for (a, b, x, expected) in cases {
        let actual = beta_reg(a, b, x);
        let relative_error = ((actual - expected) / expected).abs();
        assert!(
            relative_error <= 2e-12,
            "beta_reg({a}, {b}, {x}) = {actual}, expected {expected}, relative error {relative_error}"
        );
    }
}

#[test]
fn test_beta_reg_large_a_small_b_subnormal() {
    let cases = [
        (
            1e18,
            39.9,
            f64::from_bits(0x3feffffffffffff8),
            f64::from_bits(0x1520b9),
        ),
        (1e8, 0.9, 0.99999284, f64::from_bits(0xfce148c723)),
    ];
    for (a, b, x, expected) in cases {
        let actual = beta_reg(a, b, x);
        assert!(
            actual.to_bits().abs_diff(expected.to_bits()) <= 4,
            "beta_reg({a}, {b}, {x}) = {actual:e} ({:#x}), expected {expected:e} ({:#x})",
            actual.to_bits(),
            expected.to_bits()
        );
    }
}

#[test]
fn test_beta_reg_large_a_tiny_b_rounded_complement() {
    let x = f64::from_bits(1.0_f64.to_bits() - 1);
    let cases = [
        (
            1.7492718718060828e16,
            1.7529350052864036e-11,
            f64::from_bits(0x3d7057be8b9ff83b),
        ),
        (
            2.6496319847741348e16,
            3.8997923472821135e-12,
            f64::from_bits(0x3d2edb1e5cecbc3f),
        ),
        (
            1.3443603650606364e16,
            3.8682302848162155e-10,
            f64::from_bits(0x3dc581e85bf535df),
        ),
        (
            1.4398454548018444e16,
            1.1381500822684144e-10,
            f64::from_bits(0x3da5a5b386b28adf),
        ),
        (
            9_288_475_808_954_264.0,
            5.299156768316511e-9,
            f64::from_bits(0x3e12f55b03b79471),
        ),
        (
            1.6977806187270128e16,
            5.491909396055591e-12,
            f64::from_bits(0x3d562f56c473937b),
        ),
    ];
    for (a, b, expected) in cases {
        let actual = beta_reg(a, b, x);
        assert!(
            actual.to_bits().abs_diff(expected.to_bits()) <= 64,
            "beta_reg({a}, {b}, {x}) = {actual:e}, expected {expected:e}"
        );
    }
}

#[test]
fn test_beta_reg_small_shape_upper_gamma_against_reference() {
    let cases = [
        (
            112_176_097_488.593_9,
            1.3959752253898728e-12,
            f64::from_bits(0x3fefffffffff851d),
            [
                9.999569151432288e-13,
                9.999869047052781e-13,
                1.000016895594139e-12,
            ],
        ),
        (
            238_641_107_383.443_27,
            1.799146819367202e-12,
            f64::from_bits(0x3fefffffffffb5cc),
            [
                9.999141915967447e-13,
                9.999714463464230e-13,
                1.000028705627258e-12,
            ],
        ),
        (
            246.932962952654,
            1.1953991131275682e-12,
            f64::from_bits(0x3feffffee33a9e66),
            [
                9.999999999706979e-12,
                9.999999999957152e-12,
                1.000000000020733e-11,
            ],
        ),
    ];
    for (a, b, x, expected) in cases {
        for (offset, expected) in [-1_i64, 0, 1].into_iter().zip(expected) {
            let x = f64::from_bits(x.to_bits().wrapping_add_signed(offset));
            let actual = beta_reg(a, b, x);
            let relative_error = ((actual - expected) / expected).abs();
            assert!(
                relative_error <= 5e-13,
                "beta_reg({a}, {b}, {x}) = {actual:e}, expected {expected:e}, relative error {relative_error}"
            );
        }
    }
}

#[test]
fn test_ln_beta_reg_tiny_shape_scaled_gamma_against_reference() {
    let cases = [
        (
            f64::from_bits(0x3feffffbce423b02),
            [
                f64::from_bits(0xc085ae5914154dec),
                f64::from_bits(0xc085ae59141548a5),
                f64::from_bits(0xc085ae591415435d),
            ],
        ),
        (
            f64::from_bits(0x3fefffeb0750a667),
            [
                f64::from_bits(0xc085f9547c11ffd4),
                f64::from_bits(0xc085f9547c11fbaa),
                f64::from_bits(0xc085f9547c11f77f),
            ],
        ),
        (
            f64::from_bits(0x3fefff7be22e5816),
            [
                f64::from_bits(0xc087af793037cd6d),
                f64::from_bits(0xc087af793037c98d),
                f64::from_bits(0xc087af793037c5ad),
            ],
        ),
    ];
    for (x, expected) in cases {
        for (offset, expected) in [-1_i64, 0, 1].into_iter().zip(expected) {
            let x = f64::from_bits(x.to_bits().wrapping_add_signed(offset));
            let actual = checked_ln_beta_reg(1e6, 1e-300, x).unwrap();
            assert!((actual - expected).abs() <= 2e-13);
        }
    }
}

#[test]
fn test_ln_beta_reg_power_series_is_locally_monotone() {
    let cases = [
        (
            9.11327743985456,
            133_525_174_076_797.34,
            f64::from_bits(0x3cf3d7e149ac36dd),
        ),
        (
            6.078046923216118,
            31_131_628_187_944.344,
            f64::from_bits(0x3cf592225b607c93),
        ),
    ];
    for (a, b, root) in cases {
        let mut previous = f64::NEG_INFINITY;
        for offset in -100_i64..=100 {
            let x = f64::from_bits(root.to_bits().wrapping_add_signed(offset));
            let value = checked_ln_beta_reg(a, b, x).unwrap();
            assert!(value >= previous, "a={a}, b={b}, x={x}");
            previous = value;
        }
    }
}

#[test]
fn test_beta_reg_power_series_is_locally_monotone() {
    let cases: [(f64, f64, f64); 7] = [
        (
            0.47937889777569664,
            390_713_368_494_940.25,
            5.842150555453333e-16,
        ),
        (
            0.20713927131052443,
            1_264_447_072_006_281.8,
            7.355559632987759e-17,
        ),
        (
            0.5883286844875396,
            53_930_034_336_347.77,
            2.7619798816617607e-15,
        ),
        (
            0.3047929367901273,
            258_195_370_359_324.8,
            1.5384576649827977e-15,
        ),
        (
            0.21280081734067854,
            54_626_561.16286868,
            4.363878090733803e-9,
        ),
        (
            42.51394493556042,
            2_256_890_178_438.929,
            1.0526514336858459e-13,
        ),
        (
            77.54913939933753,
            14_481_621_713.827797,
            2.8605493321691776e-11,
        ),
    ];
    for (a, b, center) in cases {
        let mut previous = 0.0;
        for offset in -64_i64..=64 {
            let x = f64::from_bits(center.to_bits().wrapping_add_signed(offset));
            let value = beta_reg(a, b, x);
            assert!(value >= previous, "a={a}, b={b}, x={x}");
            previous = value;
        }
    }
}

#[test]
fn test_beta_reg_power_series_subnormal_result_against_reference() {
    let actual = beta_reg(
        147.13149557601173,
        1.6465152935404156e16,
        f64::from_bits(0x3c78ef1d912aaa46),
    );
    assert_eq!(actual.to_bits(), 4);
}

#[test]
fn test_beta_reg_power_series_boundary_against_reference() {
    let (log_beta, log_beta_error) = ln_beta_accurate_parts(10.0, 32.0);
    assert_eq!(log_beta.to_bits(), 0xc03723e193251f2a);
    assert!((log_beta_error - f64::from_bits(0xbcd496eeab49e82c)).abs() <= 2e-19);
    let cases = [
        (0x3fcfffffffffff7f, 0x3fe30d694d7fb0f1),
        (0x3fcfffffffffff80, 0x3fe30d694d7fb0f2),
        (0x3fcfffffffffff81, 0x3fe30d694d7fb0f4),
        (0x3fcfffffffffff82, 0x3fe30d694d7fb0f5),
    ];
    let mut previous = 0;
    for (x, expected) in cases {
        let actual = beta_reg(10.0, 32.0, f64::from_bits(x)).to_bits();
        assert!(
            actual.abs_diff(expected) <= 2,
            "x={x:#018x}, actual={actual:#018x}, expected={expected:#018x}"
        );
        assert!(
            actual > previous,
            "x={x:#018x}, actual={actual:#018x}, previous={previous:#018x}"
        );
        previous = actual;
    }
}

#[test]
fn test_beta_reg_near_one_moderate_shapes_converges() {
    let x = f64::from_bits(1.0_f64.to_bits() - 1);
    for (a, b) in [(39.9, 40.0), (40.0, 40.0), (40.0, 41.0)] {
        let actual = checked_beta_reg(a, b, x).unwrap();
        assert!(
            (0.0..=1.0).contains(&actual),
            "a={a}, b={b}, actual={actual:?}"
        );
    }
}

#[test]
fn test_beta_reg_near_one_uses_convergent_power_series() {
    let x = f64::from_bits(1.0_f64.to_bits() - 1);
    let cases = [
        (217348.9453342118, 7.083729216298346e17),
        (74.50754210941346, 4.6813710928374765e17),
        (13.940004463756644, 5.294575065065153e17),
    ];
    for (a, b) in cases {
        let actual = checked_beta_reg(a, b, x).unwrap();
        assert!(
            (0.0..=1.0).contains(&actual),
            "a={a}, b={b}, actual={actual:?}"
        );
    }
}

#[test]
fn test_beta_reg_tiny_first_shape_remains_monotone_below_split() {
    let a = 2.1856409177373306e-11;
    let b = 18.619031676940928;
    let references = [
        (0x3ea669742f6d91e9_u64, 0x3fefffffffdfb936_u64),
        (0x3fa7d0724ba189c0_u64, 0x3fefffffffff2a9e_u64),
    ];
    let mut previous = 0_u64;
    for (x, expected) in references {
        let actual = checked_beta_reg(a, b, f64::from_bits(x)).unwrap().to_bits();
        assert!(
            actual.abs_diff(expected) <= 4,
            "x={x:#018x}, actual={actual:#018x}, expected={expected:#018x}"
        );
        assert!(actual >= previous);
        previous = actual;
    }
}

#[test]
fn test_beta_reg_exact_complement_fraction_against_reference() {
    let center = 0x3ee7118258b21dd3_u64;
    let references = [
        (-128_i64, 0x3fe51a846b074d53_u64),
        (-64, 0x3fe51a846b074dbd),
        (-1, 0x3fe51a846b074e25),
        (0, 0x3fe51a846b074e27),
        (1, 0x3fe51a846b074e29),
        (64, 0x3fe51a846b074e91),
        (128, 0x3fe51a846b074efb),
    ];
    for (offset, expected) in references {
        let x = f64::from_bits(center.wrapping_add_signed(offset));
        let actual = checked_beta_reg(10.0, 1e6, x).unwrap().to_bits();
        assert!(
            actual.abs_diff(expected) <= 3,
            "offset={offset}, actual={actual:#018x}, expected={expected:#018x}"
        );
    }
    let mut previous = 0.0;
    for bits in center - 128..=center + 128 {
        let actual = checked_beta_reg(10.0, 1e6, f64::from_bits(bits)).unwrap();
        assert!(
            actual >= previous,
            "bits={bits:#018x}, previous={previous:?}, actual={actual:?}"
        );
        previous = actual;
    }
}

#[test]
fn test_beta_reg_continued_fraction_adjacent_reference() {
    let a = 1833.469197457969;
    let b = 648975.2550258434;
    let cases = [
        (0x3f63feb8f2cd8c97, 0x3e112e0be826bc4b, 0xc034b927f32c0140),
        (0x3f63feb8f2cd8c98, 0x3e112e0be826bd23, 0xc034b927f32c0133),
    ];
    let mut previous = 0;
    for (x, expected, expected_log) in cases {
        let x = f64::from_bits(x);
        assert_eq!(
            checked_ln_beta_reg(a, b, x).unwrap().to_bits(),
            expected_log
        );
        let actual = beta_reg(a, b, x).to_bits();
        let log_power = beta_reg_log_power_parts(a, b, x);
        let fraction = beta_continued_fraction(a, b, x).unwrap();
        let direct = ((log_power.0 + log_power.1).exp() / fraction).to_bits();
        assert!(
            actual.abs_diff(expected) <= 2,
            "actual={actual:#018x}, direct={direct:#018x}, expected={expected:#018x}"
        );
        assert!(actual > previous);
        previous = actual;
    }
}

#[test]
fn test_beta_reg_moderate_fraction_against_500_digit_reference() {
    let a = 25.32628846940565;
    let b = 3.1028101710805442;
    let x = 0.9276950604606229;
    let expected = 0x3fe69562e02877e6_u64;
    let actual = beta_reg(a, b, x).to_bits();
    assert!(
        actual.abs_diff(expected) <= 4,
        "actual={actual:#018x}, expected={expected:#018x}"
    );
}

#[test]
fn test_inv_beta_reg_typical_against_500_digit_reference() {
    let actual = inv_beta_reg(2.0, 5.0, 0.3).to_bits();
    let expected = 0x3fc745560dce9cd1_u64;
    assert!(
        actual.abs_diff(expected) <= 2,
        "actual={actual:#018x}, expected={expected:#018x}"
    );
}

#[test]
fn test_beta_reg_tiny_b_against_500_digit_references() {
    let cases = [
        (
            0.8144818117006096,
            1.250857626649459e-12,
            0.9669920517519052,
            0x3d94af09e6a6b751_u64,
        ),
        (
            0.2623971057030866,
            5.23256841817563e-12,
            0.9924817752047999,
            0x3dc7f760fcea90cd,
        ),
    ];
    for (a, b, x, expected) in cases {
        let actual = beta_reg(a, b, x).to_bits();
        assert!(
            actual.abs_diff(expected) <= 1,
            "a={a:?}, b={b:?}, x={x:?}, actual={actual:#018x}, expected={expected:#018x}"
        );
    }
}

#[test]
fn test_beta_reg_tiny_b_boundary_against_500_digit_references() {
    let cases = [
        (
            0.015778004354037867,
            3.91414134306449e-9,
            0.9138081692744422,
            0x3e91430c6cd6e778_u64,
        ),
        (0.5, 5e-5, 0.95, 0x3f2c8c230a2377e9),
        (0.1, 1e-5, 0.99999, 0x3f2bfe1c7f2d26f5),
    ];
    for (a, b, x, expected) in cases {
        let actual = beta_reg(a, b, x).to_bits();
        assert!(
            actual.abs_diff(expected) <= 2,
            "a={a:?}, b={b:?}, x={x:?}, actual={actual:#018x}, expected={expected:#018x}"
        );
    }
}

#[test]
fn test_beta_reg_tiny_x_large_b_against_reference() {
    let cases: [(f64, f64, f64, u64); 2] = [
        (100.0, 1e308, 1.01e-306, 0x3fe1b153914c2fe1_u64),
        (1e6, 1e308, 1.000001e-302, 0x3fe0045b85d90000_u64),
    ];
    for (a, b, center, expected) in cases {
        let center_bits = center.to_bits();
        let mut previous = 0.0;
        for bits in center_bits - 128..=center_bits + 128 {
            let actual = checked_beta_reg(a, b, f64::from_bits(bits)).unwrap();
            assert!(
                actual >= previous,
                "a={a}, b={b}, bits={bits:#018x}, previous={previous:?}, actual={actual:?}"
            );
            previous = actual;
        }
        let actual = checked_beta_reg(a, b, center).unwrap().to_bits();
        assert!(
            actual.abs_diff(expected) <= 4,
            "a={a}, b={b}, actual={actual:#018x}, expected={expected:#018x}"
        );
    }
}

#[test]
fn test_beta_reg_tiny_x_continued_fraction_singularity() {
    let references = [
        (0x3c9d1c7c0f1fd2c9_u64, 0x3fe1b153914c2fde_u64),
        (0x3c9d1c7c0f1fd2ca_u64, 0x3fe1b153914c2fe2_u64),
        (0x3c9d1c7c0f1fd2cb_u64, 0x3fe1b153914c2fe7_u64),
    ];
    let mut previous = 0_u64;
    for (x, expected) in references {
        let actual = checked_beta_reg(100.0, 1e18, f64::from_bits(x))
            .unwrap()
            .to_bits();
        assert!(
            actual.abs_diff(expected) <= 8,
            "x={x:#018x}, actual={actual:#018x}, expected={expected:#018x}"
        );
        assert!(actual >= previous);
        previous = actual;
    }
}

#[test]
fn test_beta_reg_tiny_x_does_not_lose_complement() {
    let a = 40.0;
    let b = 1e18;
    let center = 0x3c87a28834d566b4_u64;
    let mut previous = 0.0;
    for bits in center - 128..=center + 128 {
        let actual = checked_beta_reg(a, b, f64::from_bits(bits)).unwrap();
        assert!(
            actual >= previous,
            "bits={bits:#018x}, previous={previous:?}, actual={actual:?}"
        );
        previous = actual;
    }
    let actual = checked_beta_reg(a, b, f64::from_bits(center)).unwrap();
    assert_eq!(actual.to_bits(), 0x3fe2a783c7380c04);
}

#[test]
fn test_beta_reg_power_series_tiny_shape_boundary() {
    let a = f64::from_bits(0x00000000000007e8);
    let b = f64::from_bits(0x4040000000000000);
    let x = f64::from_bits(0x01556e1fc2f8f359);
    assert!(beta_power_series_log_parts(a, b, x).is_ok());
    for offset in -3_i64..=3 {
        let x = f64::from_bits(x.to_bits().wrapping_add_signed(offset));
        assert_eq!(checked_beta_reg(a, b, x).unwrap(), 1.0);
        assert_eq!(
            checked_ln_beta_reg(a, b, x).unwrap().to_bits(),
            0x8000000000155101
        );
    }
}

#[test]
fn test_beta_reg_power_series_tiny_shape_is_locally_monotone() {
    let a = f64::from_bits(0x3d719799812dea11);
    let b = f64::from_bits(0x43abc16d674ec800);
    let x = f64::from_bits(0x3c32725dd1d243ac);
    for offset in -2_i64..=3 {
        let x = f64::from_bits(x.to_bits().wrapping_add_signed(offset));
        assert_eq!(beta_reg(a, b, x).to_bits(), 0x3feffffffffff848);
    }
}

#[test]
fn test_accurate_ln_against_multiprecision_reference() {
    let cases = [
        (0x0000000000000001, 0xc0874385446d71c3, 0xbd28e569fa8ee781),
        (0x0010000000000000, 0xc086232bdd7abcd2, 0xbd1eef3fec1be37f),
        (0x39b0000000000000, 0xc051542457337d43, 0x3cde3948c376279d),
        (0x3fe8000000000000, 0xbfd269621134db92, 0xbc7e0efadd9db02b),
        (0x3ff6a09e667f3bcc, 0x3fd62e42fefa39ee, 0xbc78d6e518e495a3),
        (0x3ff6a09e667f3bcd, 0x3fd62e42fefa39f0, 0x3c7c2e0e1b1548c2),
        (0x3ff6a09e667f3bce, 0x3fd62e42fefa39f3, 0x3c7133014f0f271f),
        (0x3ff8000000000000, 0x3fd9f323ecbf984c, 0xbc4a92e513217f5c),
        (0x4000000000000000, 0x3fe62e42fefa39ef, 0x3c7abc9e3b39803f),
        (0x4630000000000000, 0x4051542457337d43, 0xbcde3948c376279d),
        (0x7fefffffffffffff, 0x40862e42fefa39ef, 0x3d1a9c9e3b39803f),
    ];
    for (input, expected_high, expected_low) in cases {
        let (high, low) = accurate_ln(f64::from_bits(input));
        let expected_low = f64::from_bits(expected_low);
        let magnitude = expected_low.abs();
        let spacing = f64::from_bits(magnitude.to_bits() + 1) - magnitude;
        assert_eq!(high.to_bits(), expected_high);
        assert!(
            (low - expected_low).abs() <= 8.0 * spacing,
            "input={input:#018x}, low={low:?}, expected={expected_low:?}"
        );
    }
}

#[test]
fn test_beta_reg_bgrat_lower_shape_boundary() {
    let cases = [
        (31.999, 0.5, 0.9, f64::from_bits(0x3f83d8d11db5fecb)),
        (32.001, 0.5, 0.9, f64::from_bits(0x3f83d79daec1916d)),
    ];
    for (a, b, x, expected) in cases {
        let actual = beta_reg(a, b, x);
        let relative_error = ((actual - expected) / expected).abs();
        assert!(
            relative_error <= 1e-12,
            "beta_reg({a}, {b}, {x}) = {actual}, expected {expected}, relative error {relative_error}"
        );
    }
}

#[test]
fn test_beta_reg_scaled_gamma_boundary_against_reference() {
    let cases = [
        (100_000.0, 10.1, 0.9996800497549934, 1.904358612390508e-6),
        (100_000.0, 10.1, 0.9996200704814975, 2.132915725768903e-8),
        (100_000.0, 10.1, 0.9996000781900461, 4.537230484132134e-9),
        (1e8, 0.1, 0.9999993610002013, 4.358202373741317e-31),
        (1e8, 0.1, 0.999999360000202, 3.9380016482795125e-31),
        (1e8, 0.9, 0.9999928600254862, 3.9763309919351194e-311),
        (1e8, 0.9, 0.9999928400256292, 5.37987584721e-312),
    ];
    for (a, b, x, expected) in cases {
        let actual = beta_reg(a, b, x);
        let relative_error = ((actual - expected) / expected).abs();
        assert!(
            relative_error <= 5e-12,
            "beta_reg({a}, {b}, {x}) = {actual:e}, expected {expected:e}, relative error {relative_error}"
        );
    }
}

#[test]
fn test_beta_reg_small_shapes_stays_in_range() {
    let cases = [
        (
            0.1350095402068847,
            2.522023373459552e-11,
            0.858047569045879,
            2.2760966295231215e-10,
        ),
        (
            1.6182184909371272e-12,
            0.8611154417262772,
            0.2090095742796264,
            0.9999999999971043,
        ),
    ];
    for (a, b, x, expected) in cases {
        let actual = beta_reg(a, b, x);
        assert!((0.0..=1.0).contains(&actual));
        assert!(
            (actual - expected).abs() <= 5e-15 * expected.max(1e-10),
            "beta_reg({a}, {b}, {x}) = {actual}, expected {expected}"
        );
    }
}

#[test]
#[should_panic]
fn test_beta_reg_a_lte_0() {
    beta_reg(0.0, 1.0, 1.0);
}

#[test]
#[should_panic]
fn test_beta_reg_b_lte_0() {
    beta_reg(1.0, 0.0, 1.0);
}

#[test]
#[should_panic]
fn test_beta_reg_x_lt_0() {
    beta_reg(1.0, 1.0, -1.0);
}

#[test]
#[should_panic]
fn test_beta_reg_x_gt_1() {
    beta_reg(1.0, 1.0, 2.0);
}

#[test]
fn test_checked_beta_reg_a_lte_0() {
    assert!(checked_beta_reg(0.0, 1.0, 1.0).is_err());
}

#[test]
fn test_checked_beta_reg_b_lte_0() {
    assert!(checked_beta_reg(1.0, 0.0, 1.0).is_err());
}

#[test]
fn test_checked_beta_reg_x_lt_0() {
    assert!(checked_beta_reg(1.0, 1.0, -1.0).is_err());
}

#[test]
fn test_checked_beta_reg_x_gt_1() {
    assert!(checked_beta_reg(1.0, 1.0, 2.0).is_err());
}

#[test]
fn test_inv_beta_reg_extreme_probability_does_not_panic() {
    let actual = inv_beta_reg(200.0, 2.0, 1e-165);
    let expected = 0.14582246504394993;
    let relative_error = ((actual - expected) / expected).abs();
    assert!(
        relative_error <= 5e-13,
        "actual {actual}, expected {expected}"
    );
}

#[test]
fn test_inv_beta_reg_extreme_probability_terminates() {
    let actual = inv_beta_reg(200.0, 2.0, 1e-60);
    let expected = 0.4897050363600545;
    let relative_error = ((actual - expected) / expected).abs();
    assert!(
        relative_error <= 5e-13,
        "actual {actual}, expected {expected}"
    );
}

#[test]
fn test_inv_beta_reg_small_shape_lower_tail() {
    let cases = [
        (1e-33, 0.0),
        (1e-32, f64::from_bits(2)),
        (1e-31, 1.215703604971242e-313),
        (1e-30, 1.2157036049544172e-303),
        (1e-20, 1.2157036049544e-203),
        (1e-10, 1.2157036049543856e-103),
        (1e-4, 1.2157036049543764e-43),
        (1e-2, 1.215703604954373e-23),
    ];
    let mut previous = 0.0;

    for (probability, expected) in cases {
        let actual = inv_beta_reg(0.1, 500.0, probability);
        if expected == 0.0 {
            assert_eq!(actual, expected);
            continue;
        }
        let relative_error = ((actual - expected) / expected).abs();
        assert!(
            relative_error <= 5e-14,
            "inv_beta_reg(0.1, 500, {probability}) = {actual}, expected {expected}, relative error {relative_error}"
        );
        assert!(actual >= previous);
        previous = actual;
    }
}

#[test]
fn test_inv_beta_reg_small_shape_rounds_extreme_tail() {
    let cases = [
        (1e-30, 0x0010aad919ea62cfa),
        (1e-31, 0x00000005baa38454),
        (1e-32, 0x0000000000000002),
    ];
    for (probability, expected) in cases {
        assert_eq!(inv_beta_reg(0.1, 500.0, probability).to_bits(), expected);
    }
}

#[test]
fn test_inv_beta_reg_early_tail_correction_against_reference() {
    assert_eq!(
        inv_beta_reg(10.0, 1e18, f64::from_bits(0x206b45a31ae6c90e),).to_bits(),
        0x392f275e33972f0c
    );
}

#[test]
fn test_inv_beta_reg_large_a_tiny_b_lower_tail() {
    let cases = [
        (
            27.229198855436444,
            3.192251825919222e-12,
            1e-12,
            0x3fef0fdff94fb881,
        ),
        (
            10.741694769633645,
            2.057645959850482e-10,
            5e-9,
            0x3fefffffffffca0f,
        ),
        (
            3.791228906881053,
            3.2160853621997853e-9,
            5e-9,
            0x3feeb7bc46a5108f,
        ),
        (
            0.07111267420172858,
            2.459402818189203e-11,
            1e-9,
            0x3fefffffffffa790,
        ),
        (
            0.0715388852036888,
            3.187243980970482e-9,
            1e-7,
            0x3feffffff2a24e82,
        ),
        (
            0.14934701587929067,
            4.564473364066682e-9,
            1e-7,
            0x3fefffff95a64b04,
        ),
    ];
    for (a, b, probability, expected) in cases {
        let actual = inv_beta_reg(a, b, probability).to_bits();
        assert!(
            actual.abs_diff(expected) <= 2,
            "a={a}, b={b}, actual={actual:#x}, expected={expected:#x}"
        );
    }
}

#[test]
fn test_beta_reg_moderate_a_tiny_b_against_reference() {
    let actual = beta_reg(
        6.333131463399467,
        1.3323977213610329e-11,
        0.9137396220685055,
    )
    .to_bits();
    let expected = 0x3d9ef22640629504_u64;
    assert!(
        actual.abs_diff(expected) <= 4,
        "actual={actual:#018x}, expected={expected:#018x}"
    );
}

#[test]
fn test_beta_reg_small_shapes_near_one_against_reference() {
    let actual = checked_beta_reg(0.8593272045160161, 0.9835139781033098, 0.9999999999999999)
        .unwrap()
        .to_bits();
    assert!(actual.abs_diff(0x3feffffffffffffe) <= 1);
}

#[test]
fn test_ln_beta_accurate_parts_reference() {
    let cases = [
        (0.1, 32.0, 0x3ffe85545aa95cd9, 0xbc8fef9442e0fba4),
        (0.3, 1000.0, 0xbfef3edcaae7008a, 0xbc8237c135557682),
        (10.0, 32.0, 0xc03723e193251f2a, 0xbcd496eeab49e82c),
    ];
    for (a, b, high, low) in cases {
        let actual = ln_beta_accurate_parts(a, b);
        assert_eq!(actual.0.to_bits(), high);
        let expected = f64::from_bits(low);
        let high_value = f64::from_bits(high).abs();
        let spacing = f64::from_bits(high_value.to_bits() + 1) - high_value;
        assert!(
            (actual.1 - expected).abs() <= 0.01 * spacing,
            "a={a}, b={b}, actual={:?}, expected={expected:?}",
            actual.1
        );
    }
    let gamma = ln_gamma_accurate_parts(0.1);
    assert_eq!(gamma.0.to_bits(), 0x4002058e35f3deee);
    assert!(
        (gamma.1 - f64::from_bits(0xbc97ad885b23066b)).abs() <= 5e-19,
        "gamma={gamma:?}"
    );
    let gamma_cases = [
        (0.125, 0x400027c4cfd515b0, 0x3c91baac8949b315),
        (0.03125, 0x400b968177c407c6, 0x3ca22ff84657c0bd),
        (0.01, 0x401265de0d9b33c4, 0xbca5ae6a9ccd75b9),
        (1e-4, 0x40226baa2b2b7f63, 0xbcc49a20e6676b4a),
        (1e-8, 0x40326bb1bb9c8a88, 0xbcc0a09c9d154d84),
        (1e-12, 0x403ba18a998ffefe, 0xbcc9835734ab358b),
        (1e-16, 0x40426bb1bbb55516, 0xbcef9d9398a70762),
        (f64::MIN_POSITIVE, 0x4086232bdd7abcd2, 0x3d1eef3fec1be37f),
        (f64::from_bits(1), 0x40874385446d71c3, 0x3d28e569fa8ee781),
    ];
    for (x, expected_high, expected_low) in gamma_cases {
        let expected_low_value = f64::from_bits(expected_low);
        let recurrence = ln_gamma_accurate_parts(x);
        assert_eq!(recurrence.0.to_bits(), expected_high, "x={x:?}");
        assert!(
            (recurrence.1 - expected_low_value).abs() <= 5e-19,
            "x={x:?}, recurrence={recurrence:?}, expected_low={expected_low_value:?}"
        );
        let series = ln_gamma_small_accurate_parts(x);
        assert_eq!(series.0.to_bits(), expected_high, "x={x:?}");
        assert!(
            series.1.to_bits().abs_diff(expected_low) <= 8,
            "x={x:?}, series={series:?}, expected_low={expected_low_value:?}"
        );
    }
    let delta = ln_gamma_delta_parts(32.0, 0.1);
    assert_eq!(delta.0.to_bits(), 0x3fd6172044f9840c);
    assert!((delta.1 - f64::from_bits(0xbc7ed6f8e6ca2265)).abs() <= 5e-19);
}

#[test]
fn test_inv_beta_reg_regular_shape_lower_tail() {
    let cases = [
        (1e-300, 7.053456158585983e-153),
        (1e-100, 7.053456158585983e-53),
        (1e-40, 7.053456158585983e-23),
        (1e-30, 7.053456158585999e-18),
        (1e-20, 7.053456158916007e-13),
    ];
    let mut previous = 0.0;

    for (probability, expected) in cases {
        let actual = inv_beta_reg(2.0, 200.0, probability);
        let relative_error = ((actual - expected) / expected).abs();
        assert!(
            relative_error <= 5e-12,
            "inv_beta_reg(2, 200, {probability}) = {actual}, expected {expected}, relative error {relative_error}"
        );
        assert!(actual > previous);
        previous = actual;
    }
}

#[test]
fn test_inv_beta_reg_large_parameters() {
    let cases = [(0.1, 0.3332984541555588), (0.9, 0.3333682129869408)];

    for (probability, expected) in cases {
        let actual = inv_beta_reg(1e8, 2e8, probability);
        let relative_error = ((actual - expected) / expected).abs();
        assert!(
            relative_error <= 5e-12,
            "inv_beta_reg(1e8, 2e8, {probability}) = {actual}, expected {expected}, relative error {relative_error}"
        );
    }
}

#[test]
fn test_inv_beta_reg_overflowing_shape_sum() {
    for shape in [1e307, 1e308] {
        assert_eq!(inv_beta_reg(shape, shape, 0.1), 0.5);
        assert_eq!(inv_beta_reg(shape, shape, 0.9), 0.5);
    }
    let expected = f64::from_bits(0x3fe5555555555555);
    for probability in [0.1, 0.5, 0.9] {
        assert_eq!(inv_beta_reg(1e308, 5e307, probability), expected);
    }
}

#[test]
fn test_inv_beta_reg_min_subnormal_large_a_tiny_b() {
    let cases = [
        (
            1.418970410722184e16,
            0.0001029663852090984,
            f64::from_bits(0x3feffffffffffe31),
        ),
        (
            4.674866848491979e16,
            1.8053488701439817e-11,
            f64::from_bits(0x3fefffffffffff77),
        ),
        (
            3.2111418342313892e16,
            0.004499324538510611,
            f64::from_bits(0x3fefffffffffff33),
        ),
        (
            3.117388966777583e17,
            0.00105319319351692,
            f64::from_bits(0x3fefffffffffffeb),
        ),
        (
            9.629243664883278e17,
            3.208469262232818e-5,
            f64::from_bits(0x3feffffffffffff9),
        ),
        (
            7.351984375091425e17,
            2.6812348495943197e-11,
            f64::from_bits(0x3feffffffffffff7),
        ),
        (
            1.7012222411445178e17,
            7.129120396546662e-6,
            f64::from_bits(0x3fefffffffffffda),
        ),
        (
            1.9543788953358486e17,
            1.1304448170316649e-12,
            f64::from_bits(0x3fefffffffffffdf),
        ),
        (
            9.996829742803416e17,
            1.410942501012109e-8,
            f64::from_bits(0x3feffffffffffffa),
        ),
    ];
    for (a, b, expected) in cases {
        let actual = inv_beta_reg(a, b, f64::from_bits(1));
        assert_eq!(actual, expected, "a={a}, b={b}");
    }
}

#[test]
fn test_inv_beta_reg_small_shape_upper_gamma() {
    let cases = [
        (
            112_176_097_488.593_9,
            1.3959752253898728e-12,
            1e-12,
            f64::from_bits(0x3fefffffffff851d),
        ),
        (
            238_641_107_383.443_27,
            1.799146819367202e-12,
            1e-12,
            f64::from_bits(0x3fefffffffffb5cc),
        ),
        (
            246.932962952654,
            1.1953991131275682e-12,
            1e-11,
            f64::from_bits(0x3feffffee33a9e66),
        ),
    ];
    for (a, b, probability, expected) in cases {
        assert_eq!(inv_beta_reg(a, b, probability), expected);
    }
}

#[test]
fn test_inv_beta_reg_large_a_tiny_b_is_monotone() {
    let cases = [
        (5.034263241208714e17, 1.8917307295846354e-5),
        (7.663354755004902e17, 0.06629881964843289),
        (9.703110430017175e17, 1.3592520602121614e-6),
        (7.633216846220836e17, 0.04203941489807821),
        (9.846275348488209e17, 7.919461066109182e-7),
        (8.324653375999025e17, 5.050727538603147e-11),
        (6.519274800253329e17, 1.3080952792915084e-9),
        (9.600975622510844e17, 3.1549066745793863e-7),
        (5.0359005294126995e17, 4.282989132250602e-6),
        (8.523009112110578e17, 2.1697803811832315e-7),
    ];
    for (a, b) in cases {
        let lower = inv_beta_reg(a, b, 1e-310);
        let upper = inv_beta_reg(a, b, 1e-300);
        assert!(lower <= upper, "a={a}, b={b}, lower={lower}, upper={upper}");
    }
}

#[test]
fn test_inv_beta_reg_log_solver_boundary_is_monotone() {
    let probability = 1e-8_f64;
    let probabilities = [
        f64::from_bits(probability.to_bits() - 1),
        probability,
        f64::from_bits(probability.to_bits() + 1),
    ];
    let cases = [
        (
            2.0,
            200.0,
            [
                f64::from_bits(0x3ea7ab27fd13660a),
                f64::from_bits(0x3ea7ab27fd13660b),
                f64::from_bits(0x3ea7ab27fd13660b),
            ],
        ),
        (
            0.1,
            500.0,
            [
                f64::from_bits(0x2eb79df9fcc6b8b8),
                f64::from_bits(0x2eb79df9fcc6b8c3),
                f64::from_bits(0x2eb79df9fcc6b8ce),
            ],
        ),
        (3.508179849994976e17, 0.8360747930277879, [1.0; 3]),
    ];
    for (a, b, expected) in cases {
        let actual = probabilities.map(|p| inv_beta_reg(a, b, p));
        assert!(
            actual[0] <= actual[1] && actual[1] <= actual[2],
            "a={a}, b={b}, actual={actual:?}"
        );
        for ((value, reference), probability) in actual.into_iter().zip(expected).zip(probabilities)
        {
            let ulp_error = value.to_bits().abs_diff(reference.to_bits());
            assert!(
                ulp_error <= INVERSE_REFERENCE_MAX_ULPS,
                "a={a}, b={b}, probability={probability}, value={value}, reference={reference}, ulp_error={ulp_error}"
            );
            let quantile_relative_error = ((value - reference) / reference).abs();
            assert!(
                quantile_relative_error <= 4e-14,
                "a={a}, b={b}, probability={probability}, value={value}, reference={reference}, quantile_relative_error={quantile_relative_error}"
            );
            if value > 0.0 && value < 1.0 {
                let relative_error = ((beta_reg(a, b, value) - probability) / probability).abs();
                assert!(
                    relative_error <= 1e-14,
                    "a={a}, b={b}, probability={probability}, value={value}, relative_error={relative_error}"
                );
            }
        }
    }
}

#[test]
fn test_inv_beta_reg_adjacent_probability_is_monotone() {
    let probability = 1e-8_f64;
    let probabilities = [
        f64::from_bits(probability.to_bits() - 1),
        probability,
        f64::from_bits(probability.to_bits() + 1),
    ];
    let cases = [
        (
            9.11327743985456,
            133_525_174_076_797.34,
            f64::from_bits(0x3cf3d7e149ac36dd),
        ),
        (
            6.078046923216118,
            31_131_628_187_944.344,
            f64::from_bits(0x3cf592225b607c93),
        ),
    ];
    for (a, b, expected) in cases {
        let actual = probabilities.map(|p| inv_beta_reg(a, b, p));
        assert!(
            actual[0] <= actual[1] && actual[1] <= actual[2],
            "a={a}, b={b}, actual={actual:?}"
        );
        for value in actual {
            assert!(value.to_bits().abs_diff(expected.to_bits()) <= INVERSE_REFERENCE_MAX_ULPS);
        }
    }
}

#[test]
fn test_inv_beta_reg_upper_adjacent_probability_is_monotone() {
    let cases = [
        (
            100.0,
            1e6,
            [0x3feffffffffffff9, 0x3feffffffffffffa],
            [0x3f2a6e8528d3e729, 0x3f2a78942066b3b0],
        ),
        (
            1000.0,
            1e6,
            [0x3feffffffffffff7, 0x3feffffffffffff8],
            [0x3f54d1ec0e95e0f5, 0x3f54d42ffc3c17aa],
        ),
        (
            1000.0,
            1e6,
            [0x3feffffffffffffb, 0x3feffffffffffffc],
            [0x3f54dd318598d8ed, 0x3f54e1735a4b5c03],
        ),
        (
            1000.0,
            1e6,
            [0x3feffffffffffffd, 0x3feffffffffffffe],
            [0x3f54e6ebec74e0ca, 0x3f54ee997db90e85],
        ),
        (
            1000.0,
            1e8,
            [0x3feffffffffffff3, 0x3feffffffffffff4],
            [0x3eeaa4df95604c33, 0x3eeaa6db7106f8eb],
        ),
    ];
    for (a, b, probability_bits, expected_bits) in cases {
        let actual = probability_bits.map(|bits| inv_beta_reg(a, b, f64::from_bits(bits)));
        assert!(actual[0] <= actual[1]);
        for (value, expected) in actual.into_iter().zip(expected_bits.map(f64::from_bits)) {
            let ulp_error = value.to_bits().abs_diff(expected.to_bits());
            assert!(
                ulp_error <= INVERSE_REFERENCE_MAX_ULPS,
                "a={a}, b={b}, value={value}, expected={expected}, ulp_error={ulp_error}"
            );
        }
    }
}

#[test]
fn test_inv_beta_reg_orientation_preserves_tiny_quantiles() {
    let cases = [
        (0.49, f64::from_bits(0x083429b7deb4de35)),
        (0.5, f64::from_bits(0x0a0650cbd0bac729)),
        (0.51, f64::from_bits(0x0bd08de62d4b3d17)),
        (0.9, f64::from_bits(0x3f064452047719b0)),
        (0.99, 1.0),
    ];
    let mut previous = 0.0;
    for (probability, expected) in cases {
        let actual = inv_beta_reg(0.001, 0.01, probability);
        assert!(actual >= previous);
        if expected == 1.0 {
            assert_eq!(actual, expected);
        } else {
            assert!(((actual - expected) / expected).abs() <= 1e-12);
        }
        previous = actual;
    }
    let actual = inv_beta_reg(0.01, 1e8, 0.51);
    let expected = f64::from_bits(0x38260460ad60f7d3);
    assert!(((actual - expected) / expected).abs() <= 1e-12);
}

#[test]
fn test_inv_beta_reg_concentrated_quantiles_round_correctly() {
    let cases = [
        (
            5.6337457945398355e35,
            3.4148653071385907e36,
            0.1,
            f64::from_bits(0x3fc2206894075924),
        ),
        (
            5.6337457945398355e35,
            3.4148653071385907e36,
            0.9,
            f64::from_bits(0x3fc2206894075924),
        ),
        (
            7.778370008599511e35,
            3.99094171205976e36,
            f64::from_bits(1),
            f64::from_bits(0x3fc4e0cc7f8ea39f),
        ),
        (
            7.778370008599511e35,
            3.99094171205976e36,
            0.1,
            f64::from_bits(0x3fc4e0cc7f8ea3a0),
        ),
        (
            7.778370008599511e35,
            3.99094171205976e36,
            0.9,
            f64::from_bits(0x3fc4e0cc7f8ea3a0),
        ),
    ];
    for (a, b, probability, expected) in cases {
        assert_eq!(inv_beta_reg(a, b, probability), expected);
    }
}

#[test]
fn test_inv_beta_reg_extreme_tail_balanced_shapes() {
    let cases = [
        (f64::from_bits(1), 0.1384383837250825),
        (1e-300, 0.14764444133469024),
    ];
    for (probability, expected) in cases {
        let actual = inv_beta_reg(1000.0, 1000.0, probability);
        let relative_error = ((actual - expected) / expected).abs();
        assert!(
            relative_error <= 5e-13,
            "probability {probability}, actual {actual}, expected {expected}, relative error {relative_error}"
        );
    }
}

#[test]
fn test_inv_beta_reg_extreme_tail_imbalanced_shapes() {
    let cases = [
        (200.0, 2.0, 1e-192, 0.10683857283574616),
        (1000.0, 2.0, f64::from_bits(1), 0.47203081850113066),
        (1000.0, 2.0, 1e-303, 0.49464719057284383),
        (1000.0, 2.0, 1e-200, 0.627230829476228),
        (1000.0, 2.0, 1e-100, 0.7900887907081466),
        (1000.0, 10.0, f64::from_bits(1), 0.454569346824437),
        (1000.0, 10.0, 1e-303, 0.47650393899531424),
        (1000.0, 10.0, 1e-200, 0.6055787273511661),
        (1000.0, 10.0, 1e-100, 0.7659557362087095),
        (1000.0, 100.0, f64::from_bits(1), 0.356892489498544),
        (1000.0, 100.0, 1e-303, 0.3750351205470552),
        (1000.0, 100.0, 1e-200, 0.48455098775995836),
        (1000.0, 100.0, 1e-100, 0.6303764215497716),
        (7_627_209.761, 11.3319, 1.679e-274, 0.9999105965110135),
    ];
    for (a, b, probability, expected) in cases {
        let actual = inv_beta_reg(a, b, probability);
        let relative_error = ((actual - expected) / expected).abs();
        assert!(
            relative_error <= 5e-13,
            "inv_beta_reg({a}, {b}, {probability}) = {actual}, expected {expected}, relative error {relative_error}"
        );
    }
}

#[test]
fn test_inv_beta_reg_subnormal_power_series_boundary() {
    let a = f64::from_bits(0x4024000000000000);
    let b = f64::from_bits(0x7e37e43c8800759c);
    let probability = f64::from_bits(0x2df5ed8667733d64);
    for offset in -2_i64..=2 {
        let probability = f64::from_bits(probability.to_bits().wrapping_add_signed(offset));
        assert_eq!(
            inv_beta_reg(a, b, probability).to_bits(),
            0x000730d67819e860,
            "offset={offset}"
        );
    }
}

#[test]
fn test_error_is_sync_send() {
    fn assert_sync_send<T: Sync + Send>() {}
    assert_sync_send::<BetaFuncError>();
}
