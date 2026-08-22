use crate::statistics::OnlineMoments;

/// A type that accumulates `f64` observations one at a time, which is
/// a single-pass accumulator for central moments via Welford's online algorithm.
///
/// Tuples up to arity 8 fan each observation into each accumulator, so
/// multiple statistics can share a single pass - the pattern is more
/// useful when the online stats aren't all moments, but example below:
///
/// ```
/// use statrs::statistics::{Accumulate, OnlineMean, OnlineVariance};
///
/// let data = [3.0_f64, -1.0, 4.0, 1.0, -5.0];
/// let (OnlineMean(mean), OnlineVariance(variance)) =
///     data.iter()
///         .copied()
///         .fold(Accumulate::default(), Accumulate::push)
///         .get();
/// // Both `mean` and `variance` are of type `Option<f64>`
/// assert!(mean.is_some());
/// assert!(variance.is_some());
/// ```
///
/// Moments are accumulated for `x - offset`, where `offset` is the first value
/// pushed. Central moments are invariant under that shift, and it is what makes
/// the accumulator usable on data with a large offset.
pub struct Accumulate<MS: OnlineMoments> {
    pub count: u64,
    // `m` holds the moments of `x - offset`. Welford's update can become
    // insensitive reducing unscaled data. See statrs-dev/statrs#376.
    //
    // Some type magics can be done here to make the array length equal
    // to MS::order(). But since we only need up to three-order moment,
    // a simple [_; 3] is enough.
    offset: f64,
    pub(super) m: [f64; 3],
    phantom: core::marker::PhantomData<MS>,
}

impl<MS: OnlineMoments> Default for Accumulate<MS> {
    fn default() -> Self {
        Self {
            count: 0,
            offset: 0.0,
            m: [0.0; 3],
            phantom: core::marker::PhantomData,
        }
    }
}

impl<MS: OnlineMoments> Accumulate<MS> {
    /// Merges two accumulators as if all observations had been pushed into
    /// one, using the pairwise update of Chan, Golub & LeVeque (extended to
    /// the third moment by Pébay, 2008).
    /// In addition to API after computing on parallel streams, merging
    /// at the end is also slightly *better* conditioned than one long Welford
    /// chain.
    ///
    /// ```
    /// use statrs::statistics::OnlineVariance;
    /// use statrs::statistics::Accumulate;
    /// let a = [1.0_f64, 2.0].iter().copied().fold(Accumulate::default(), Accumulate::push);
    /// let b = [3.0_f64, 4.0].iter().copied().fold(Accumulate::default(), Accumulate::push);
    /// let all = [1.0_f64, 2.0, 3.0, 4.0].iter().copied().fold(Accumulate::default(), Accumulate::push);
    ///
    /// let (OnlineVariance(merged_variance),) = a.merge(b).get();
    /// let (OnlineVariance(all_variance),) = all.get();
    ///
    /// assert_eq!(merged_variance, all_variance);
    /// ```
    ///
    /// # Precision
    /// Consider breaking apart large streams and merging for precision.
    /// Recursively merging in a binary tree accumulates roughly O(log N)
    /// rounding error against O(N) for a single chain, the same effect as
    /// pairwise vs. naive summation.
    ///
    /// ```
    /// use statrs::statistics::{OnlineVariance, Accumulate};
    /// use approx::assert_relative_eq;
    ///
    /// // Repeating a block leaves its variance unchanged, so this stream has
    /// // an exactly known variance to check against: mean 5.0, M2 = 32 per
    /// // block.
    /// let block = [2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    /// let blocks = 1 << 16;
    /// let exact_variance = (32.0 * blocks as f64) / (8.0 * blocks as f64 - 1.0);
    ///
    /// let chained = (0..blocks)
    ///     .flat_map(|_| block.iter().copied())
    ///     .fold(Accumulate::default(), Accumulate::push);
    /// let (OnlineVariance(chained_variance),) = chained.get();
    ///
    /// let stream = |count| {
    ///     (0..count)
    ///         .flat_map(|_| block.iter().copied())
    ///         .fold(Accumulate::default(), Accumulate::push)
    /// };
    /// let per_stream = blocks / 4;
    /// let merged = stream(per_stream)
    ///     .merge(stream(per_stream))
    ///     .merge(stream(per_stream).merge(stream(per_stream)));
    /// let (OnlineVariance(merged_variance),) = merged.get();
    ///
    /// let chained_err = (chained_variance.unwrap() - exact_variance).abs();
    /// let merged_err = (merged_variance.unwrap() - exact_variance).abs();
    /// assert!(merged_err < chained_err);
    /// assert_relative_eq!(merged_variance.unwrap(), exact_variance, max_relative = 1e-13);
    /// ```
    pub fn merge(self, other: Self) -> Self {
        if other.count == 0 {
            return self;
        }
        if self.count == 0 {
            return other;
        }
        let na = self.count as f64;
        let nb = other.count as f64;
        let n = na + nb;
        // The two accumulators generally have different offsets, so re-express
        // `other`'s mean in `self`'s frame. Grouping the two differences
        // separately keeps this accurate when the offsets are close, which is
        // the common case (both are data values).
        let delta = (other.offset - self.offset) + (other.m[0] - self.m[0]);

        let mut m = [0.0; 3];
        m[0] = self.m[0] + delta * nb / n;
        if MS::order() >= 2 {
            let m2a = self.m[1];
            let m2b = other.m[1];
            if MS::order() >= 3 {
                let m3a = self.m[2];
                let m3b = other.m[2];
                m[2] = m3a
                    + m3b
                    + delta * delta * delta * na * nb * (na - nb) / (n * n)
                    + 3.0 * delta * (na * m2b - nb * m2a) / n;
            }
            m[1] = m2a + m2b + delta * delta * na * nb / n;
        }

        Self {
            count: self.count + other.count,
            offset: self.offset,
            m,
            phantom: self.phantom,
        }
    }

    /// Folds one observation into the moments.
    ///
    /// ```
    /// use statrs::statistics::OnlineVariance;
    /// use statrs::statistics::Accumulate;
    /// let (OnlineVariance(variance),) = [1.0_f64, 2.0, 3.0].iter().copied()
    ///     .fold(Accumulate::default(), Accumulate::push).get();
    /// ```
    ///
    /// # Precision
    /// Sensitive to data ordering, especially with regard to scale of initial item.
    /// If consuming very large streams, see [`merge`][Accumulate::merge]
    pub fn push(mut self, x: f64) -> Self {
        if self.count == 0 {
            self.offset = x;
        }
        self.count += 1;
        let n = self.count as f64;
        // work relative to the first observation; see the type-level docs
        let x = x - self.offset;

        // Welford / Pebay (2008) central moment update. Update order: M3
        // before M2 before mean; each step uses the previous observation's
        // lower-order accumulators.
        let delta = x - self.m[0];
        let delta_n = delta / n;
        let new_mean = self.m[0] + delta_n;
        let delta2 = x - new_mean;

        if MS::order() >= 2 {
            let old_m2 = self.m[1];
            if MS::order() >= 3 {
                let inc =
                    delta * (delta_n * delta_n) * (n - 1.0) * (n - 2.0) - 3.0 * delta_n * old_m2;
                self.m[2] += inc;
            }
            self.m[1] += delta * delta2;
        }

        self.m[0] = new_mean;
        self
    }

    /// Get any online stats you want.
    ///
    /// You can combine any stats that implements
    /// [`OnlineMoment`][crate::statistics::OnlineMoment] with any order
    /// into a tuple, and the Rust compiler will automatically infer the
    /// return type of this method.
    ///
    /// ```
    /// use statrs::statistics::{OnlineVariance, OnlineSkewness, OnlineStdDev};
    /// use statrs::statistics::Accumulate;
    /// let (OnlineVariance(variance), OnlineSkewness(skewness), OnlineStdDev(std_dev)) =
    ///     [1.0_f64, 2.0, 3.0].iter().copied()
    ///         .fold(Accumulate::default(), Accumulate::push).get();
    /// ```
    pub fn get(&self) -> MS {
        MS::from_acc(self)
    }
}
