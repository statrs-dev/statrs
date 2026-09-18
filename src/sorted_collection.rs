use std::{
    cmp::Ordering,
    ops::{Deref, Index},
    slice::SliceIndex,
};

pub enum SortError {
    NotSorted,
}

#[derive(Clone, Debug)]
pub enum Collection<'a, T> {
    Ref(&'a [T]),
    Owned(Vec<T>),
}

#[derive(Clone, Debug)]
pub struct SortedCollection<'a, T> {
    sorted: Collection<'a, T>,
}

impl<'a, T> SortedCollection<'a, T>
where
    T: Ord,
{
    pub fn from_slice(coll: &'a [T]) -> Result<Self, SortError> {
        match coll
            .windows(2)
            .all(|sl| sl[0].cmp(&sl[1]) == Ordering::Less)
        {
            true => Ok(Self {
                sorted: Collection::Ref(coll),
            }),
            false => Err(SortError::NotSorted),
        }
    }
    pub fn from_mut_vec(mut coll: Vec<T>) -> Self {
        coll.sort();
        Self {
            sorted: Collection::Owned(coll),
        }
    }
    pub fn iter(&'a self) -> std::slice::Iter<'a, T> {
        match self.sorted {
            Collection::Ref(items) => items.iter(),
            Collection::Owned(ref items) => items.iter(),
        }
    }
}

impl<'a, T> TryFrom<&'a [T]> for SortedCollection<'a, T>
where
    T: Ord,
{
    type Error = SortError;

    fn try_from(value: &'a [T]) -> Result<Self, Self::Error> {
        Self::from_slice(value)
    }
}

impl<'a, T> TryFrom<&'a Box<[T]>> for SortedCollection<'a, T>
where
    T: Ord,
{
    type Error = SortError;

    fn try_from(value: &'a Box<[T]>) -> Result<Self, Self::Error> {
        Self::from_slice(value.as_ref())
    }
}

impl<'a, T> AsRef<[T]> for SortedCollection<'a, T> {
    fn as_ref(&self) -> &[T] {
        match self.sorted {
            Collection::Ref(items) => items,
            Collection::Owned(ref items) => items.as_ref(),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum SortMarker {
    NotSorted,
    Sorted,
}

#[derive(Debug)]
pub struct SortedVec<T: Ord>(SortMarker, Vec<T>);

impl<T> SortedVec<T>
where
    T: Ord + Clone,
{
    pub fn sort(&mut self) {
        self.1.sort();
        self.0 = SortMarker::Sorted;
    }
    pub fn is_sorted(&self) -> bool {
        self.0 == SortMarker::Sorted
    }
}

impl<F, T> From<F> for SortedVec<T>
where
    F: Into<Vec<T>> + Iterator<Item = T> + Copy,
    T: Ord + Clone,
{
    fn from(value: F) -> Self {
        let marker = match value.is_sorted() {
            true => SortMarker::Sorted,
            false => SortMarker::NotSorted,
        };

        Self(marker, value.into())
    }
}

impl<T> Deref for SortedVec<T>
where
    T: Ord,
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.1.deref()
    }
}

impl<T, I> Index<I> for SortedVec<T>
where
    T: Ord,
    I: SliceIndex<[T]>,
{
    type Output = <I as SliceIndex<[T]>>::Output;

    fn index(&self, index: I) -> &Self::Output {
        self.1.index(index)
    }
}

/// Represents a slice of a SortedVec
/// Stores a reference to a slice and a reference to its corresponding `SortedVec`'s `SortMaker`.
pub struct SortedSlice<'a, T>(&'a SortMarker, &'a [T]);

impl<'a, T> SortedSlice<'a, T> {
    /// Creates a SortedSlice with the provided SortMarker  without checking whether the slice is
    /// adheres to it.
    pub fn sorted_from_unchecked(marker: &'a SortMarker, vals: &'a [T]) -> Self {
        Self(marker, vals)
    }
    pub fn len(&self) -> usize {
        self.1.len()
    }
    pub fn is_empty(&self) -> bool {
        self.1.is_empty()
    }
    pub fn is_sorted(&self) -> bool {
        *self.0 == SortMarker::Sorted
    }
}

impl<'a, T, I> Index<I> for SortedSlice<'a, T>
where
    T: Ord,
    I: SliceIndex<[T]>,
{
    type Output = <I as SliceIndex<[T]>>::Output;

    fn index(&self, index: I) -> &Self::Output {
        self.1.index(index)
    }
}
impl<'a, T> Deref for SortedSlice<'a, T>
where
    T: Ord,
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.1
    }
}
