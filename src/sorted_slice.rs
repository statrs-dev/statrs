use std::{
    cmp::Ordering,
    ops::{Deref, Index},
    slice::SliceIndex,
};

#[derive(Debug)]
pub enum SortError {
    NotSorted,
}

pub struct SortedSlice<'a, T>(&'a [T]);

impl<'a, T> SortedSlice<'a, T> {
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.0.iter()
    }
    pub fn into_iter(self) -> std::slice::Iter<'a, T> {
        self.0.into_iter()
    }
}

impl<'a> TryInto<SortedSlice<'a, f64>> for &'a [f64] {
    type Error = SortError;

    fn try_into(self) -> Result<SortedSlice<'a, f64>, Self::Error> {
        match (
            self.is_sorted_by(|a, b| a.total_cmp(b) != Ordering::Greater),
            self.is_empty(),
        ) {
            (_, true) => Ok(SortedSlice(self)),
            (true, false) => Ok(SortedSlice(self)),
            (false, false) => Err(SortError::NotSorted),
        }
    }
}

impl<'a, T, I> Index<I> for SortedSlice<'a, T>
where
    T: Ord,
    I: SliceIndex<[T]>,
{
    type Output = <I as SliceIndex<[T]>>::Output;

    fn index(&self, index: I) -> &Self::Output {
        self.0.index(index)
    }
}
impl<'a, T> Deref for SortedSlice<'a, T>
where
    T: Ord,
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.0
    }
}
