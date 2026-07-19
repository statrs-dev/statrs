/// A type that accumulates `f64` observations one at a time.
///
/// Tuples up to arity 5 fan each observation into each accumulator, so
/// multiple statistics can share a single pass - the pattern is more
/// useful when the online stats aren't all moments, but example below:
///
/// ```
/// use statrs::statistics::{Accumulate, OnlineMean, OnlineVariance};
///
/// let data = [3.0_f64, -1.0, 4.0, 1.0, -5.0];
/// let (mean, var): (OnlineMean, OnlineVariance) =
///     data.iter().copied().fold(Default::default(), Accumulate::push);
///
/// assert!(mean.mean().is_some());
/// assert!(var.variance().is_some());
/// ```
pub trait Accumulate: Default + Sized {
    fn push(self, x: f64) -> Self;
}

impl<A: Accumulate> Accumulate for (A,) {
    fn push(self, x: f64) -> Self {
        (self.0.push(x),)
    }
}

impl<A: Accumulate, B: Accumulate> Accumulate for (A, B) {
    fn push(self, x: f64) -> Self {
        (self.0.push(x), self.1.push(x))
    }
}

impl<A: Accumulate, B: Accumulate, C: Accumulate> Accumulate for (A, B, C) {
    fn push(self, x: f64) -> Self {
        (self.0.push(x), self.1.push(x), self.2.push(x))
    }
}

impl<A: Accumulate, B: Accumulate, C: Accumulate, D: Accumulate> Accumulate for (A, B, C, D) {
    fn push(self, x: f64) -> Self {
        (
            self.0.push(x),
            self.1.push(x),
            self.2.push(x),
            self.3.push(x),
        )
    }
}

impl<A: Accumulate, B: Accumulate, C: Accumulate, D: Accumulate, E: Accumulate> Accumulate
    for (A, B, C, D, E)
{
    fn push(self, x: f64) -> Self {
        (
            self.0.push(x),
            self.1.push(x),
            self.2.push(x),
            self.3.push(x),
            self.4.push(x),
        )
    }
}
