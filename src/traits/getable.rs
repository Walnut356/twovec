use either::Either;

use crate::TwoVec;

pub trait Getable<A, B, const Z: bool>
where
    Self: Sized,
    A: Copy,
    B: Copy,
{
    fn get(tv: &TwoVec<A, B>, idx: usize) -> Option<Self>;
}

impl<A: Copy, B: Copy> Getable<A, B, false> for A {
    fn get(tv: &TwoVec<A, B>, idx: usize) -> Option<Self> {
        tv.get_a(idx)
    }
}

impl<A: Copy, B: Copy> Getable<A, B, true> for B {
    fn get(tv: &TwoVec<A, B>, idx: usize) -> Option<Self> {
        tv.get_b(idx)
    }
}

impl<A: Copy, B: Copy> Getable<A, B, false> for Either<A, B> {
    fn get(tv: &TwoVec<A, B>, idx: usize) -> Option<Self> {
        tv.get_either(idx)
    }
}
