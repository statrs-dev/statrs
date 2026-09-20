use std::{
    ops::{Deref, Index},
    slice::SliceIndex,
};

pub enum SortError {
    NotSorted,
}

pub struct SortedSlice<'a, T>(&'a [T]);

impl<'a, T> TryInto<SortedSlice<'a, T>> for &'a [T]
where
    T: Ord,
{
    type Error = SortError;

    fn try_into(self) -> Result<SortedSlice<'a, T>, Self::Error> {
        match self.is_sorted() {
            true => Ok(SortedSlice(self)),
            false => Err(SortError::NotSorted),
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
