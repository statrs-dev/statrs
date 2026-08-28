use std::cmp::Ordering;

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
