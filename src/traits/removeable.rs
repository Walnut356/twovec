use either::Either;

use crate::TwoVec;

pub trait Removable<A, B, const Z: bool>
where
    Self: Sized,
    A: Copy,
    B: Copy,
{
    type Output;

    fn remove(tv: &mut TwoVec<A, B>, idx: usize) -> Self::Output;
}

impl<A: Copy, B: Copy> Removable<A, B, false> for A {
    type Output = Option<A>;

    fn remove(tv: &mut TwoVec<A, B>, idx: usize) -> Self::Output {
        tv.remove_a(idx)
    }
}

impl<A: Copy, B: Copy> Removable<A, B, true> for B {
    type Output = Option<B>;

    fn remove(tv: &mut TwoVec<A, B>, idx: usize) -> Self::Output {
        tv.remove_b(idx)
    }
}

impl<A: Copy, B: Copy> Removable<A, B, true> for Either<A, B> {
    type Output = Self;
    
    fn remove(tv: &mut TwoVec<A, B>, idx: usize) -> Self::Output {
        tv.remove_either(idx)
    }
}
