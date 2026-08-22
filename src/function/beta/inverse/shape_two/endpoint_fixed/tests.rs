use super::Fixed;

#[test]
fn multiplication_bounds_contain_the_full_precision_product() {
    let cases = [(0.1, 0.3), (1.5, 2.25), (7.75, 7.875)];
    for cutoff in [0, 8, 16, 24, 31] {
        for (left, right) in cases {
            let left = Fixed::from_f64(left).unwrap();
            let right = Fixed::from_f64(right).unwrap();
            let exact = left.mul_floor(right, 0).unwrap();
            let lower = left.mul_floor(right, cutoff).unwrap();
            let upper = left.mul_ceil(right, cutoff).unwrap();
            assert!(lower <= exact && exact <= upper);
        }
    }
}

#[test]
fn division_bounds_are_outward_and_detect_discarded_bits() {
    let value = Fixed::from_f64(1.0).unwrap();
    for cutoff in [0, 8, 16, 24, 31] {
        let fine_lower = value.div_small_floor(3, 0);
        let fine_upper = value.div_small_ceil(3, 0).unwrap();
        let lower = value.div_small_floor(3, cutoff);
        let upper = value.div_small_ceil(3, cutoff).unwrap();
        assert!(lower <= fine_lower && fine_upper <= upper);
        assert!(lower < upper);
    }

    let exact = Fixed::from_f64(1.5).unwrap();
    assert!(exact.div_small_floor(2, 0) == exact.div_small_ceil(2, 0).unwrap());
}
