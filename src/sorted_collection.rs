pub enum SortError {
    NotSorted,
}

pub struct SortedCollection<'a, T> {
    sorted: &'a [T],
}

impl<'a, T> SortedCollection<'a, T>
where
    T: PartialOrd,
{
    pub fn new(coll: &'a [T]) -> Result<Self, SortError> {
        match coll.windows(2).all(|sl| sl[0] < sl[1]) {
            true => Ok(Self { sorted: coll }),
            false => Err(SortError::NotSorted),
        }
    }
}
