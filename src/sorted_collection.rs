use std::cmp::Ordering;

pub enum SortError {
    NotSorted,
}

pub enum Collection<'a, T> {
    Ref(&'a [T]),
    Owned(Vec<T>),
}

pub struct SortedCollection<'a, T> {
    sorted: Collection<'a, T>,
}

impl<'a, T> SortedCollection<'a, T>
where
    T: PartialOrd + Ord,
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
    pub fn from_mut_vec(mut coll: Vec<T>) -> Result<Self, SortError> {
        coll.sort();
        Ok(Self {
            sorted: Collection::Owned(coll),
        })
    }
    pub fn iter(&'a self) -> std::slice::Iter<'a, T> {
        match self.sorted {
            Collection::Ref(items) => items.iter(),
            Collection::Owned(ref items) => items.iter(),
        }
    }
}
