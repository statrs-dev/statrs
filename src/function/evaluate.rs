//! Provides functions that don't have a numerical solution and must
//! be solved computationally (e.g. evaluation of a polynomial)

/// evaluates a polynomial at `z` where `coeff` are the coeffecients
/// to a polynomial of order `k` where `k` is the length of `coeff` and the
/// coeffecient
/// to the `k`th power is the `k`th element in coeff. E.g. [3,-1,2] equates to
/// `2z^2 - z + 3`
///
/// # Remarks
///
/// Returns 0 for a 0 length coefficient slice
pub fn polynomial(z: f64, coeff: &[f64]) -> f64 {
    let n = coeff.len();
    if n == 0 {
        return 0.0;
    }

    let mut sum = *coeff.last().unwrap();
    for c in coeff[0..n - 1].iter().rev() {
        sum = *c + z * sum;
    }
    sum
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use core::f64;

    // these tests probably could be more robust
    #[test]
    fn test_polynomial() {
        let empty: [f64; 0] = [];
        assert_eq!(super::polynomial(2.0, &empty), 0.0);

        let zero = [0.0];
        assert_eq!(super::polynomial(2.0, &zero), 0.0);

        let mut coeff = [1.0, 0.0, 5.0];
        assert_eq!(super::polynomial(2.0, &coeff), 21.0);

        coeff = [-5.0, -2.0, 3.0];
        assert_eq!(super::polynomial(2.0, &coeff), 3.0);
        assert_eq!(super::polynomial(-2.0, &coeff), 11.0);

        let large_coeff = [-1.35e3, 2.5e2, 8.0, -4.0, 1e2, 3.0];
        assert_eq!(super::polynomial(5.0, &large_coeff), 71475.0);
        assert_eq!(super::polynomial(-5.0, &large_coeff), 51225.0);

        coeff = [f64::INFINITY, -2.0, 3.0];
        assert_eq!(super::polynomial(2.0, &coeff), f64::INFINITY);
        assert_eq!(super::polynomial(-2.0, &coeff), f64::INFINITY);

        coeff = [f64::NEG_INFINITY, -2.0, 3.0];
        assert_eq!(super::polynomial(2.0, &coeff), f64::NEG_INFINITY);
        assert_eq!(super::polynomial(-2.0, &coeff), f64::NEG_INFINITY);

        coeff = [f64::NAN, -2.0, 3.0];
        assert!(super::polynomial(2.0, &coeff).is_nan());
        assert!(super::polynomial(-2.0, &coeff).is_nan());
    }
}
