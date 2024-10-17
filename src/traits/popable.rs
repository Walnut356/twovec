use either::Either;

use crate::TwoVec;

pub trait Popable<A, B, const Z: bool>
where
    Self: Sized,
    A: Copy,
    B: Copy,
{
    fn pop(tv: &mut TwoVec<A, B>) -> Option<Self>;
}

impl<A: Copy, B: Copy> Popable<A, B, false> for A {
    fn pop(tv: &mut TwoVec<A, B>) -> Option<Self> {
        tv.pop_a()
    }
}

impl<A: Copy, B: Copy> Popable<A, B, true> for B {
    fn pop(tv: &mut TwoVec<A, B>) -> Option<Self> {
        tv.pop_b()
    }
}

impl<A: Copy, B: Copy> Popable<A, B, false> for Either<A, B> {
    fn pop(tv: &mut TwoVec<A, B>) -> Option<Self> {
        tv.pop_either()
    }
}
