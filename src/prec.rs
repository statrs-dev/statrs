//! Provides utility functions for working with floating point precision

/// Standard epsilon, maximum relative precision of IEEE 754 double-precision
/// floating point numbers (64 bit) e.g. `2^-53`
pub const F64_PREC: f64 = 0.00000000000000011102230246251565;

/// Default accuracy for `f64`, equivalent to `0.0 * F64_PREC`
pub const DEFAULT_F64_ACC: f64 = 0.0000000000000011102230246251565;

/// Returns true if `a` and `b `are within `acc` of each other.
/// If `a` or `b` are infinite, returns `true` only if both are
/// infinite and similarly signed. Always returns `false` if
/// either number is a `NAN`.
pub fn almost_eq(a: f64, b: f64, acc: f64) -> bool {
    // only true if a and b are infinite with same
    // sign
    if a.is_infinite() || b.is_infinite() {
        return a == b;
    }

    // NANs are never equal
    if a.is_nan() && b.is_nan() {
        return false;
    }

    (a - b).abs() < acc
}

/// Compares if two floats are close via `approx::relative_eq!`
/// and `crate::consts::ACC` relative precision.
/// Updates first argument to value of second argument
pub fn convergence(x: &mut f64, x_new: f64) -> bool {
    let res = approx::relative_eq!(*x, x_new, max_relative = crate::consts::ACC);
    *x = x_new;
    res
}
