use either::Either;

use crate::TwoVec;

pub struct TwoVecIter<'a, A: Copy, B: Copy> {
    inner: &'a TwoVec<A, B>,
    curr: usize,
}

impl<'a, A: Copy, B: Copy> TwoVecIter<'a, A, B> {
    pub fn new(twovec: &'a TwoVec<A, B>) -> Self {
        Self {
            inner: twovec,
            curr: 0,
        }
    }
}

impl<'a, A: Copy, B: Copy> Iterator for TwoVecIter<'a, A, B> {
    type Item = Either<A, B>;

    fn next(&mut self) -> Option<Self::Item> {
        let val: Option<Either<A, B>> = self.inner.get(self.curr);
        self.curr += 1;
        val
    }
}

impl<'a, A: Copy, B: Copy> IntoIterator for &'a TwoVec<A, B> {
    type Item = Either<A, B>;

    type IntoIter = TwoVecIter<'a, A, B>;

    fn into_iter(self) -> Self::IntoIter {
        TwoVecIter {
            inner: self,
            curr: 0,
        }
    }
}

pub struct TwoVecIterA<'a, A: Copy, B: Copy> {
    inner: &'a TwoVec<A, B>,
    curr: usize,
}

impl<'a, A: Copy, B: Copy> TwoVecIterA<'a, A, B> {
    pub fn new(twovec: &'a TwoVec<A, B>) -> Self {
        Self {
            inner: twovec,
            curr: 0,
        }
    }
}

impl<'a, A: Copy, B: Copy> Iterator for TwoVecIterA<'a, A, B> {
    type Item = A;

    fn next(&mut self) -> Option<Self::Item> {
        let mut val = None;

        while val.is_none() && self.curr < self.inner.len {
            val = self.inner.get(self.curr);
            self.curr += 1;
        }

        val
    }
}

pub struct TwoVecIterB<'a, A: Copy, B: Copy> {
    inner: &'a TwoVec<A, B>,
    curr: usize,
}

impl<'a, A: Copy, B: Copy> TwoVecIterB<'a, A, B> {
    pub fn new(twovec: &'a TwoVec<A, B>) -> Self {
        Self {
            inner: twovec,
            curr: 0,
        }
    }
}

impl<'a, A: Copy, B: Copy> Iterator for TwoVecIterB<'a, A, B> {
    type Item = B;

    fn next(&mut self) -> Option<Self::Item> {
        let mut val = None;

        while val.is_none() && self.curr < self.inner.len {
            val = self.inner.get(self.curr);
            self.curr += 1;
        }

        val
    }
}
