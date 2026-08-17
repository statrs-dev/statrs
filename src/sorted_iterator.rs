use std::marker::PhantomData;

use crate::stats_tests::NaNPolicy;

pub trait SortedIterator {
    fn sorted_iter(&self, policy: NaNPolicy) -> Sorted;
}

impl SortedIterator for Vec<f64> {
    fn sorted_iter(&self, policy: NaNPolicy) -> Sorted {
        Sorted::new(self, policy)
    }
}

/// TODO iron out implementation details later because this is not optimal retrieval of sorted data
pub struct Sorted {
    sorted_iter: std::vec::IntoIter<f64>,
    policy: NaNPolicy,
}

impl Sorted {
    pub fn new(data: &[f64], policy: NaNPolicy) -> Self {
        let mut cloned = Vec::from(data);
        cloned.sort_by(|a, b| a.total_cmp(b));
        Self {
            sorted_iter: cloned.into_iter(),
            policy,
        }
    }
}

impl Iterator for Sorted {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        self.sorted_iter.next()
    }
}
