use either::Either;

use crate::TwoVec;

pub trait Pushable<A, B, const Z: bool>
where
    Self: Sized,
    A: Copy,
    B: Copy
{
    fn push(tv: &mut TwoVec<A, B>, val: Self);
}

impl<A: Copy, B: Copy> Pushable<A, B, false> for A {
    fn push(tv: &mut TwoVec<A, B>, val: Self) {
        tv.push_a(val);
    }
}

impl<A: Copy, B: Copy> Pushable<A, B, true> for B {
    fn push(tv: &mut TwoVec<A, B>, val: Self) {
        tv.push_b(val);
    }
}

impl<A: Copy, B: Copy> Pushable<A, B, true> for Either<A, B> {
    fn push(tv: &mut TwoVec<A, B>, val: Self) {
        tv.push_either(val);
    }
}