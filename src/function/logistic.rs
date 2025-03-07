//! Provides the [logistic](http://en.wikipedia.org/wiki/Logistic_function) and
//! related functions

/// Computes the logistic function
pub fn logistic(p: f64) -> f64 {
    1.0 / ((-p).exp() + 1.0)
}

/// Computes the logit function
///
/// # Panics
///
/// If `p < 0.0` or `p > 1.0`
pub fn logit(p: f64) -> f64 {
    checked_logit(p).unwrap()
}

/// Computes the logit function, returning `None` if `p < 0.0` or `p > 1.0`.
pub fn checked_logit(p: f64) -> Option<f64> {
    if (0.0..=1.0).contains(&p) {
        Some((p / (1.0 - p)).ln())
    } else {
        None
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use core::f64;
    use crate::prec;
    use super::*;

    #[test]
    fn test_logistic() {
        assert_eq!(logistic(f64::NEG_INFINITY), 0.0);
        assert_eq!(logistic(-11.512915464920228103874353849992239636376994324587), 0.00001);
        prec::assert_abs_diff_eq!(logistic(-6.9067547786485535272274487616830597875179908939086), 0.001, epsilon = 1e-18);
        prec::assert_abs_diff_eq!(logistic(-2.1972245773362193134015514347727700402304323440139), 0.1, epsilon = 1e-16);
        assert_eq!(logistic(0.0), 0.5);
        prec::assert_abs_diff_eq!(logistic(2.1972245773362195801634726294284168954491240598975), 0.9, epsilon = 1e-15);
        prec::assert_abs_diff_eq!(logistic(6.9067547786485526081487245019905638981131702804661), 0.999, epsilon = 1e-15);
        assert_eq!(logistic(11.512915464924779098232747799811946290419057060965), 0.99999);
        assert_eq!(logistic(f64::INFINITY), 1.0);
    }

    #[test]
    fn test_logit() {
        assert_eq!(logit(0.0), f64::NEG_INFINITY);
        assert_eq!(logit(0.00001), -11.512915464920228103874353849992239636376994324587);
        assert_eq!(logit(0.001), -6.9067547786485535272274487616830597875179908939086);
        assert_eq!(logit(0.1), -2.1972245773362193134015514347727700402304323440139);
        assert_eq!(logit(0.5), 0.0);
        assert_eq!(logit(0.9), 2.1972245773362195801634726294284168954491240598975);
        assert_eq!(logit(0.999), 6.9067547786485526081487245019905638981131702804661);
        assert_eq!(logit(0.99999), 11.512915464924779098232747799811946290419057060965);
        assert_eq!(logit(1.0), f64::INFINITY);
    }

    #[test]
    #[should_panic]
    fn test_logit_p_lt_0() {
        logit(-1.0);
    }

    #[test]
    #[should_panic]
    fn test_logit_p_gt_1() {
        logit(2.0);
    }

    #[test]
    fn test_checked_logit_p_lt_0() {
        assert!(checked_logit(-1.0).is_none());
    }

    #[test]
    fn test_checked_logit_p_gt_1() {
        assert!(checked_logit(2.0).is_none());
    }
}
