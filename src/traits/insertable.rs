use either::Either;

use crate::TwoVec;

pub trait Insertable<A, B, const Z: bool>
where
    Self: Sized,
    A: Copy,
    B: Copy,
{
    fn insert(tv: &mut TwoVec<A, B>, idx: usize, val: Self);
}

impl<A: Copy, B: Copy> Insertable<A, B, false> for A {
    fn insert(tv: &mut TwoVec<A, B>, idx: usize, val: Self) {
        tv.insert_a(idx, val);
    }
}

impl<A: Copy, B: Copy> Insertable<A, B, true> for B {
    fn insert(tv: &mut TwoVec<A, B>, idx: usize, val: Self) {
        tv.insert_b(idx, val);
    }
}

impl<A: Copy, B: Copy> Insertable<A, B, false> for Either<A, B> {
    fn insert(tv: &mut TwoVec<A, B>, idx: usize, val: Self) {
        tv.insert_either(idx, val);
    }
}
