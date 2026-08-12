use super::*;

fn exact_integer(value: u64) -> Interval {
    Interval::exact(Fixed::integer(value).unwrap(), 0).unwrap()
}

#[test]
fn alternating_remainder_uses_the_correct_parity() {
    let positive = exact_integer(10);
    let negative = exact_integer(3);
    let term = exact_integer(1);
    let even = remainder_interval(positive, negative, term, 8).unwrap();
    let odd = remainder_interval(positive, negative, term, 9).unwrap();

    assert!(even.lower == Fixed::integer(7).unwrap());
    assert!(even.upper == Fixed::integer(8).unwrap());
    assert!(odd.lower == Fixed::integer(6).unwrap());
    assert!(odd.upper == Fixed::integer(7).unwrap());
}

#[test]
fn full_precision_reaches_an_enclosure_at_domain_extremes() {
    let shapes = [
        f64::from_bits(MINIMUM_SHAPE_BITS),
        f64::from_bits(0x7fef_ffff_ffff_ffff),
    ];
    let probabilities = [
        f64::from_bits(1),
        0.5,
        f64::from_bits(1.0_f64.to_bits() - 1),
    ];
    let lower_bits = [0, 1, ENDPOINT_LIMIT_BITS - 1];

    for shape in shapes {
        for probability in probabilities {
            for lower in lower_bits {
                assert!(!matches!(
                    order_at_precision(shape, probability, lower, FRACTION_LIMBS),
                    Stage::Refine
                ));
            }
        }
    }
}

#[test]
fn invalid_inputs_are_rejected() {
    for shape in [f64::NAN, f64::INFINITY, -1.0, 1.0] {
        assert_eq!(
            midpoint_certificate(shape, 0.5, 0),
            Err(CertifierError::InvalidInput)
        );
    }
    for probability in [f64::NAN, 0.0, 1.0] {
        assert_eq!(
            midpoint_certificate(f64::from_bits(MINIMUM_SHAPE_BITS), probability, 0),
            Err(CertifierError::InvalidInput)
        );
    }
}
