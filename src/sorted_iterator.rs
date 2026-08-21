pub enum SortedIteratorError {
    TestError,
}
pub trait IntoSortedIterator {
    fn into_sorted_iter(&self) -> Sorted;
}

impl IntoSortedIterator for Vec<f64> {
    fn into_sorted_iter(&self) -> Sorted {
        Sorted::new(self)
    }
}

/// TODO iron out implementation details later because this is not optimal retrieval of sorted data
#[derive(Clone)]
pub struct Sorted {
    sorted_iter: std::vec::IntoIter<f64>,
}

impl Sorted {
    pub fn new(data: &[f64]) -> Self {
        let mut cloned = Vec::from(data);
        cloned.sort_by(|a, b| a.total_cmp(b));
        Self {
            sorted_iter: cloned.into_iter(),
        }
    }
}

impl Iterator for Sorted {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        ///TODO potential algorithm:
        ///use bloom filter to test for untraveled indices, I'm not sure how to implement traversal
        ///yet
        self.sorted_iter.next()
    }
}
