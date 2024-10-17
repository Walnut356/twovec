mod raw_mem;
mod traits;
use either::Either;
pub use traits::*;

use std::ptr::{slice_from_raw_parts, slice_from_raw_parts_mut, NonNull};
use std::{fmt::Debug, marker::PhantomData};

pub struct TwoVec<A: Copy, B: Copy> {
    /// number of meaningful bits in `self.bitfield`
    len: usize,
    /// Allocation size of `self.data`
    capacity: usize,
    /// pointer to the start of the bitfield. contained within the same allocation as `self.data`
    bitfield: NonNull<u8>,
    /// pointer to the start of the contained data. contained within the same allocation as `self.data`, and occurs just after the end of the area pointed to by `self.bitfield`
    data: NonNull<u8>,
    a: PhantomData<A>,
    b: PhantomData<B>,
}

impl<A: Copy, B: Copy> TwoVec<A, B> {
    /// size_of::<A>()
    pub const A_SIZE: usize = size_of::<A>();
    /// size_of::<B>()
    pub const B_SIZE: usize = size_of::<B>();

    // no const traits so no const Ord so no const .max and .min functions =(

    /// The size of the largest type between `A` and `B`
    const MAX_A_B: usize = if Self::A_SIZE > Self::B_SIZE {
        Self::A_SIZE
    } else {
        Self::B_SIZE
    };

    /// The size of the smallest type between `A` and `B`
    const MIN_A_B: usize = if Self::A_SIZE < Self::B_SIZE {
        Self::A_SIZE
    } else {
        Self::B_SIZE
    };

    /// Creates a new `TwoVec`. Does not allocate until an element is added.
    pub fn new() -> Self {
        Self {
            len: 0,
            capacity: 0,
            bitfield: NonNull::dangling(),
            data: NonNull::dangling(),
            a: PhantomData,
            b: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the total capacity of the current allocation in bytes
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the max number of `A` objects that can be stored in the current allocation
    pub fn a_capacity(&self) -> usize {
        (self.capacity - self.bitfield_size()) / Self::A_SIZE
    }

    /// Returns the max number of `B` objects that can be stored in the current allocation
    pub fn b_capacity(&self) -> usize {
        (self.capacity - self.bitfield_size()) / Self::B_SIZE
    }

    pub(crate) fn bitfield_size(&self) -> usize {
        unsafe { self.data.offset_from(self.bitfield) as usize }
    }

    /// Returns the entire allocated memory block as a byte array. Mostly used for debugging
    #[allow(dead_code)]
    fn mem_block(&self) -> &[u8] {
        unsafe {
            slice_from_raw_parts(self.bitfield.as_ptr(), self.capacity)
                .as_ref()
                .unwrap()
        }
    }

    /// Returns the offset from `self.data` to the value at `idx`. Walks `self` by individually checking each bit and
    /// adding the appropriate `size_of` to the offset. Significantly (approx 100x) slower than
    /// `self.idx_to_offset`. Mostly used for debugging purposes since the algorithm is so simple.
    #[allow(dead_code)]
    pub(crate) fn walk_to_idx(&self, idx: usize) -> usize {
        let mut ptr = self.data;

        for i in 0..idx {
            if self.is_a(i) {
                unsafe { ptr = ptr.add(size_of::<A>()) };
            } else {
                unsafe { ptr = ptr.add(size_of::<B>()) };
            }
        }

        unsafe { ptr.offset_from(self.data) as usize }
    }

    /// Returns the offset from `self.data` to the value at `idx`.
    pub(crate) fn idx_to_offset(&self, idx: usize) -> usize {
        let extra_byte = (idx % 8 != 0) as usize;
        let len = (idx / 8) + extra_byte;
        let bitfield = unsafe {
            slice_from_raw_parts(self.bitfield.as_ptr(), len)
                .as_ref()
                .unwrap()
        };

        let mut result = 0;

        for (i, chunk) in bitfield.chunks(8).enumerate() {
            let val = match chunk.len() {
                8 => u64::from_be_bytes(chunk.try_into().unwrap()),
                x => {
                    let mut slice = [0u8; 8];
                    slice[..x].copy_from_slice(&chunk[..x]);

                    u64::from_be_bytes(slice)
                }
            };

            let important_bits = idx % 64;
            if (i + 1) * 8 < len || important_bits == 0 {
                let a_count = val.count_zeros() as usize;
                result += (a_count * Self::A_SIZE) + ((64 - a_count) * Self::B_SIZE);
                continue;
            }

            let shift_bits = 64 - important_bits;

            let b_count = ((val >> shift_bits) << shift_bits).count_ones() as usize;

            let a_bytes = important_bits.saturating_sub(b_count) * Self::A_SIZE;
            let b_bytes = b_count * Self::B_SIZE;

            result += a_bytes + b_bytes
        }

        result
    }

    /// Sets the bit corresponding to the element at `idx` to 1. Used to indicate that the type of
    /// that object is `B`.
    fn set_bit(&mut self, idx: usize) {
        let byte_idx = idx / 8;
        let bit_idx = idx % 8;
        let bf = self.bitfield_mut();
        bf[byte_idx] |= 1 << (7 - bit_idx);
    }

    /// Sets the bit corresponding to the element at `idx` to 0. Used to indicate that the type of
    /// that object is `A`.
    fn clear_bit(&mut self, idx: usize) {
        let byte_idx = idx / 8;
        let bit_idx = idx % 8;
        let bf = self.bitfield_mut();
        bf[byte_idx] &= !(1 << (7 - bit_idx));
    }

    /// Returns the value of the bit corresponding to the element at `idx`. 0 == type `A`, 1 == type `B`
    fn get_bit(&self, idx: usize) -> u8 {
        let byte_idx = idx / 8;
        let bit_idx = idx % 8;
        ((self.bitfield()[byte_idx]) & (1 << (7 - bit_idx)) != 0) as u8
    }

    /// Returns `true` if the value at the given index is of type `A`
    pub fn is_a(&self, idx: usize) -> bool {
        self.get_bit(idx) == 0
    }

    /// Returns `true` if the value at the given index is of type `B`
    pub fn is_b(&self, idx: usize) -> bool {
        !self.is_a(idx)
    }

    pub(crate) fn bitfield(&self) -> &[u8] {
        let byte_idx = self.len / 8;
        let bit_idx = self.len % 8;

        let add = (bit_idx != 0) as usize;
        unsafe {
            slice_from_raw_parts(self.bitfield.as_ptr(), byte_idx + add)
                .as_ref()
                .unwrap()
        }
    }

    pub(crate) fn bitfield_mut(&mut self) -> &mut [u8] {
        let byte_idx = self.len / 8;
        let bit_idx = self.len % 8;

        let add = (bit_idx != 0) as usize;
        unsafe {
            slice_from_raw_parts_mut(self.bitfield.as_ptr(), byte_idx + add)
                .as_mut()
                .unwrap()
        }
    }

    /// Adds the given value to the end of the `TwoVec`, reallocating if necessary.
    pub fn push<T, const Z: bool>(&mut self, val: T)
    where
        T: Pushable<A, B, Z>,
    {
        T::push(self, val);
    }

    /// Adds a value of type `A` to the end of the `TwoVec`, reallocating if necessary.
    pub fn push_a(&mut self, val: A) {
        if self.capacity == 0
            || self.idx_to_offset(self.len) + Self::A_SIZE + self.bitfield_size() > self.capacity
        {
            self.grow();
        }
        let end = unsafe { self.data.add(self.idx_to_offset(self.len)).cast::<A>() };
        // unsafe {dbg!(self.data.offset_from(self.bitfield));
        // dbg!(end.offset_from(self.bitfield.cast()));}
        unsafe { end.write_unaligned(val) };

        self.len += 1;

        self.clear_bit(self.len - 1);
    }

    /// Adds a value of type `B` to the end of the `TwoVec`, reallocating if necessary.
    pub fn push_b(&mut self, val: B) {
        if self.capacity == 0
            || self.idx_to_offset(self.len) + Self::B_SIZE + self.bitfield_size() > self.capacity
        {
            self.grow();
        }
        let end = unsafe { self.data.add(self.idx_to_offset(self.len)).cast::<B>() };
        unsafe { end.write_unaligned(val) };

        self.len += 1;

        self.set_bit(self.len - 1);
    }

    /// Adds the given value to the end of the `TwoVec`, reallocating if necessary.
    pub fn push_either(&mut self, val: Either<A, B>) {
        match val {
            Either::Left(x) => self.push_a(x),
            Either::Right(x) => self.push_b(x),
        }
    }

    /// Returns the value at the given index. If `index >= self.len()` returns None.
    pub fn get<T, const Z: bool>(&self, idx: usize) -> Option<T>
    where
        T: Getable<A, B, Z>,
    {
        T::get(self, idx)
    }

    /// Returns the value at the given index. If `index >= self.len()` or the type of the object at
    /// `index` is `B` returns None.
    pub fn get_a(&self, idx: usize) -> Option<A> {
        if idx >= self.len || !self.is_a(idx) {
            return None;
        }
        let offset = self.idx_to_offset(idx);
        let ptr = unsafe { self.data.byte_add(offset) };
        unsafe { Some(ptr.cast::<A>().read_unaligned()) }
    }

    /// Returns the value at the given index. If `index >= self.len()` or the type of the object at
    /// `index` is `A` returns None.
    pub fn get_b(&self, idx: usize) -> Option<B> {
        if idx >= self.len || !self.is_b(idx) {
            return None;
        }
        let offset = self.idx_to_offset(idx);
        let ptr = unsafe { self.data.byte_add(offset) };
        unsafe { Some(ptr.cast::<B>().read_unaligned()) }
    }

    /// Returns the value at the given index. If `index >= self.len()`, returns None.
    pub fn get_either(&self, idx: usize) -> Option<Either<A, B>> {
        if idx >= self.len() {
            return None;
        }

        if self.is_a(idx) {
            self.get_a(idx).map(|x| Either::Left(x))
        } else {
            self.get_b(idx).map(|x| Either::Right(x))
        }
    }

    /// If `!self.is_empty()` removes the value from the end of `self` and returns it.
    /// Otherwise does nothing and returns None.
    pub fn pop<T, const Z: bool>(&mut self) -> Option<T>
    where
        T: Popable<A, B, Z>,
    {
        T::pop(self)
    }

    /// If `!self.is_empty()` and the last element of the array is of type `A`, removes the value
    /// from the end of `self` and returns it. Otherwise does nothing and returns None
    pub fn pop_a(&mut self) -> Option<A> {
        let res = self.get_a(self.len - 1);
        if res.is_some() {
            self.len -= 1;
        }

        res
    }

    /// If `!self.is_empty()` and the last element of the array is of type `A`, removes the value
    /// from the end of `self` and returns it. Otherwise does nothing and returns None
    pub fn pop_b(&mut self) -> Option<B> {
        let res = self.get_b(self.len - 1);
        if res.is_some() {
            self.len -= 1;
        }

        res
    }

    /// If `!self.is_empty()`, removes the value from the end of `self` and returns it.
    pub fn pop_either(&mut self) -> Option<Either<A, B>> {
        let res = self.get_either(self.len - 1);
        if res.is_some() {
            self.len -= 1;
        }

        res
    }

    /// Returns an iterator that yields only elements of type `A`
    pub fn iter_only_a(&self) -> impl Iterator<Item = A> + '_ {
        TwoVecIterA::new(self)
    }

    /// Returns an iterator that yields only elements of type `B`
    pub fn iter_only_b(&self) -> impl Iterator<Item = B> + '_ {
        TwoVecIterB::new(self)
    }

    /// Creates 2 new vecs, one for each type, and copies elements from `self` to their respective
    /// `Vec`s. Since `A` and `B` are required to be `Copy`, this does not consume `self`.
    pub fn to_two_vecs(&self) -> (Vec<A>, Vec<B>) {
        let mut a = Vec::new();
        let mut b = Vec::new();

        for v in self {
            match v {
                Either::Left(x) => a.push(x),
                Either::Right(x) => b.push(x),
            }
        }

        (a, b)
    }

    /// "Removes" all values by setting length to 0.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Inserts the given value of type `A` into the TwoVec at the given index. All values at or above the given
    /// index are shifted to make space.
    pub fn insert_a(&mut self, idx: usize, element: A) {
        if idx > self.len {
            panic!("Cannot insert value at index {idx}, length is {}", self.len);
        }
        if self.idx_to_offset(self.len) + Self::A_SIZE + self.bitfield_size() > self.capacity {
            self.grow();
        }

        unsafe {
            let end_offset = self.idx_to_offset(self.len);
            let val_offset = self.idx_to_offset(idx);
            let ptr = self.data.add(val_offset);

            ptr.copy_to(ptr.add(Self::A_SIZE), end_offset - val_offset);

            ptr.cast::<A>().write(element);
        }

        let bf = &mut self.bitfield_mut()[idx / 8..];
        let rem = idx % 8;

        let mut x = bf[0];
        let mut carry = x & 1;
        // clear the top bits, shift the value 1 further to the right, then set the bits accordingly
        let bottom_bits = (x << rem) >> (rem + 1).min(7);
        let top_bits = (x >> (7 - rem)) << (7 - rem);
        x = top_bits | bottom_bits;
        // the above leaves a gap, which we fill with the appropriate bit
        x &= !(1 << (7 - rem));
        bf[0] = x;

        // now we do the same for the rest of the bitfield bytes
        for x in &mut bf[1..] {
            let next_carry = *x & 1;
            *x = (*x >> 1) | (carry << 7);
            carry = next_carry;
        }

        self.len += 1;
    }

    /// Inserts the given value of type `B` into the TwoVec at the given index. All values at or above the given
    /// index are shifted to make space.
    pub fn insert_b(&mut self, idx: usize, element: B) {
        if idx > self.len {
            panic!("Cannot insert value at index {idx}, length is {}", self.len);
        }
        if self.idx_to_offset(self.len) + Self::B_SIZE + self.bitfield_size() > self.capacity {
            self.grow();
        }

        unsafe {
            let end_offset = self.idx_to_offset(self.len);
            let val_offset = self.idx_to_offset(idx);
            let ptr = self.data.add(val_offset);

            ptr.copy_to(ptr.add(Self::B_SIZE), end_offset - val_offset);

            ptr.cast::<B>().write(element);
        }

        let bf = &mut self.bitfield_mut()[idx / 8..];
        let rem = idx % 8;

        let mut x = bf[0];
        let mut carry = x & 1;
        // clear the top bits, shift the value 1 further to the right, then set the bits accordingly
        let bottom_bits = (x << rem) >> (rem + 1).min(7);
        let top_bits = (x >> (7 - rem)) << (7 - rem);
        x = top_bits | bottom_bits;
        // the above leaves a gap, which we fill with the appropriate bit
        x |= 1 << (7 - rem);
        bf[0] = x;

        // now we do the same for the rest of the bitfield bytes
        for x in &mut bf[1..] {
            let next_carry = *x & 1;
            *x = (*x >> 1) | (carry << 7);
            carry = next_carry;
        }

        self.len += 1;
    }

    /// Inserts the given value into the TwoVec at the given index. All values at or above the given
    /// index are shifted to make space.
    pub fn insert_either(&mut self, idx: usize, element: Either<A, B>) {
        match element {
            Either::Left(x) => self.insert_a(idx, x),
            Either::Right(x) => self.insert_b(idx, x),
        }
    }

    /// If the value at the given index is of type `A`, the value is removed and returned.
    /// Otherwise, `None` is returned and the `TwoVec` **is not modified**
    ///
    /// All values after the given index are shifted down to fill newly empty space
    pub fn remove_a(&mut self, idx: usize) -> Option<A> {
        if idx > self.len {
            panic!("Cannot remove value at index {idx}, length is {}", self.len);
        }
        if !self.is_a(idx) {
            return None;
        }

        let val_offset = self.idx_to_offset(idx);
        let end_offset = self.idx_to_offset(self.len);
        let out: Option<A>;

        unsafe {
            let val_ptr = self.data.add(val_offset);
            out = Some(val_ptr.cast::<A>().read_unaligned());

            let after_val_ptr = val_ptr.add(Self::A_SIZE);

            let end_ptr = self.data.add(end_offset);

            after_val_ptr.copy_to(val_ptr, end_ptr.byte_offset_from(after_val_ptr) as usize);
        }

        self.shift_bits_down_after(idx);

        self.len -= 1;
        out
    }

    /// If the value at the given index is of type `B`, the value is removed and returned.
    /// Otherwise, `None` is returned and the `TwoVec` **is not modified**.
    ///
    /// All values after the given index are shifted down to fill newly empty space
    pub fn remove_b(&mut self, idx: usize) -> Option<B> {
        if idx > self.len {
            panic!("Cannot remove value at index {idx}, length is {}", self.len);
        }
        if !self.is_b(idx) {
            return None;
        }

        let val_offset = self.idx_to_offset(idx);
        let end_offset = self.idx_to_offset(self.len);
        let out: Option<B>;

        unsafe {
            let val_ptr = self.data.add(val_offset);
            out = Some(val_ptr.cast::<B>().read_unaligned());

            let after_val_ptr = val_ptr.add(Self::B_SIZE);
            let end_ptr = self.data.add(end_offset);

            after_val_ptr.copy_to(val_ptr, end_ptr.byte_offset_from(after_val_ptr) as usize);
        }

        self.shift_bits_down_after(idx);

        self.len -= 1;
        out
    }

    /// Removes the value at the given index and returns it. All values after the given index are
    /// shifted down to fill newly empty space
    pub fn remove_either(&mut self, idx: usize) -> Either<A, B> {
        if self.is_a(idx) {
            Either::Left(self.remove_a(idx).unwrap())
        } else {
            Either::Right(self.remove_b(idx).unwrap())
        }
    }

    #[allow(dead_code)]
    fn shift_bits(&mut self, idx: usize) {
        for i in idx..self.len - 1 {
            let b = self.get_bit(i + 1);
            if b == 0 {
                self.clear_bit(i);
            } else {
                self.set_bit(i);
            }
        }
    }

    fn shift_bits_down_after(&mut self, idx: usize) {
        let bf = self.bitfield_mut();

        let start_byte_idx = idx / 8;
        let start_bit_idx = idx % 8;

        let val = bf[start_byte_idx];

        // clears all the bits including and after idx
        let shift_amt = 8 - start_bit_idx;
        let pre_idx_bits = ((val as u32 >> shift_amt) << shift_amt) as u8;

        // clears all the bits up to and including idx. Requires a +1 because bits are
        // "0-indexed" when shifting. When shifting back, we drop the +1 so that we can overwrite the
        // bit that's being removed
        let post_idx_bits = (((val as u32) << (start_bit_idx + 1)) as u8) >> start_bit_idx;

        // if there isn't a next byte, it'll set to 0. If there is, we shift the top bit to the bottom
        // so we can mask it onto the bottom of `val`
        let mut carry = bf.get(start_byte_idx + 1).cloned().unwrap_or_default() >> 7;

        bf[start_byte_idx] = pre_idx_bits | post_idx_bits | carry;

        // now we do the same for the rest of the bitfield bytes
        for i in start_byte_idx + 1..bf.len() {
            let mut x = bf[i];
            carry = bf.get(i + 1).cloned().unwrap_or_default() >> 7;
            x = (x << 1) | carry;
            bf[i] = x;
        }
    }
}

impl<A: Copy, B: Copy> Default for TwoVec<A, B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Debug + Copy, B: Debug + Copy> Debug for TwoVec<A, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TwoVec[")?;
        let mut ptr = self.data;

        for i in 0..self.len {
            if self.is_a(i) {
                write!(f, "{:?}", unsafe { (ptr.cast::<A>()).read_unaligned() })?;

                unsafe { ptr = ptr.add(size_of::<A>()) };
            } else {
                // dbg!(unsafe { (ptr as *const B).as_ref().unwrap() });
                write!(f, "{:?}", unsafe { (ptr.cast::<B>()).read_unaligned() })?;
                unsafe { ptr = ptr.add(size_of::<B>()) };
            }

            if i != self.len - 1 {
                write!(f, ", ")?;
            }
        }

        write!(f, "]")
    }
}

impl<A: Copy + PartialEq, B: Copy + PartialEq> PartialEq for TwoVec<A, B> {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }

        let lhs = self.bitfield();
        let rhs = other.bitfield();
        let bytes = self.len / 8;
        let bits = self.len % 8;

        if lhs[..bytes] != rhs[..bytes] {
            return false;
        }

        if bits != 0 {
            let x = lhs[bytes];
            let y = rhs[bytes];

            let b = bits - 1;

            if (x >> (7 - b)) << (7 - b) != (y >> (7 - b)) << (7 - b) {
                dbg!(x);
                dbg!(y);
                return false;
            }
        }

        for (lhs, rhs) in std::iter::zip(self.into_iter(), other.into_iter()) {
            if lhs != rhs {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn index_offset() {
        let b = 1u8;
        let a = -1i64;

        let mut tv = TwoVec::new();
        for i in (0..2000).step_by(2) {
            tv.push_a(a);

            unsafe {
                let walk_offset = tv.walk_to_idx(i);
                let walk = tv.data.add(walk_offset);

                let calc_offset = tv.idx_to_offset(i);
                let calc = tv.data.add(calc_offset);

                // dbg!(walk.read());
                // dbg!(calc.read());

                assert_eq!(
                    walk,
                    calc,
                    "indexes are {} bytes apart on step {} (len: {})",
                    walk.offset_from(calc),
                    i,
                    tv.len
                );

                tv.push_b(b);

                let walk_offset = tv.walk_to_idx(i + 1);
                let walk = tv.data.add(walk_offset);

                let calc_offset = tv.idx_to_offset(i + 1);
                let calc = tv.data.add(calc_offset);

                assert_eq!(
                    walk,
                    calc,
                    "indexes are {} bytes apart on step {} (len: {})",
                    walk.offset_from(calc),
                    i,
                    tv.len
                );
            }
        }
    }

    #[test]
    fn insert() {
        let mut rng = SmallRng::from_entropy();
        let seed = rng.gen();
        let mut rng = SmallRng::seed_from_u64(seed);

        let mut tv = TwoVec::<u16, f32>::new();
        let mut either = Vec::<Either<u16, f32>>::new();
        for _ in 0..100 {
            if rng.gen_bool(0.5) {
                let val = rng.gen();
                tv.push_a(val);
                either.push(Either::Left(val));
            } else {
                let val = rng.gen();
                tv.push_b(val);
                either.push(Either::Right(val));
            }
        }

        for _ in 0..10000 {
            if rng.gen_bool(0.5) {
                let val = rng.gen();
                let idx = rng.gen_range(0..tv.len());
                tv.insert_a(idx, val);

                either.insert(idx, Either::Left(val));
            } else {
                let val = rng.gen();
                let idx = rng.gen_range(0..tv.len());
                tv.insert_b(idx, val);

                either.insert(idx, Either::Right(val));
            }
        }

        for i in 0..10000 {
            let l = tv.get_either(i).unwrap();
            let r = *either.get(i).unwrap();

            assert_eq!(
                l, r,
                "Values do not match at index {i}. TwoVec: {l}, Vec: {r} [Seed: {seed}]"
            );
        }
    }

    #[test]
    fn remove_a() {
        let mut rng = SmallRng::from_entropy();
        let seed = rng.gen();
        let mut rng = SmallRng::seed_from_u64(seed);

        let mut tv = TwoVec::<u16, f32>::new();
        let mut either = Vec::<Either<u16, f32>>::new();
        for _ in 0..10000 {
            if rng.gen_bool(0.5) {
                let val = rng.gen();
                tv.push_a(val);
                either.push(Either::Left(val));
            } else {
                let val = rng.gen();
                tv.push_b(val);
                either.push(Either::Right(val));
            }
        }

        for _ in 0..10000 {
            let i = rng.gen_range(0..tv.len());
            let l = tv.remove_a(i);
            if l.is_none() {
                continue;
            }
            let r = either.remove(i).left();

            assert_eq!(
                l, r,
                "Values not equal. TwoVec: {l:?}, Vec: {r:?} [Seed: {seed}]"
            );
        }
    }

    #[test]
    fn remove_b() {
        let mut rng = SmallRng::from_entropy();
        let seed = rng.gen();
        let mut rng = SmallRng::seed_from_u64(seed);

        let mut tv = TwoVec::<u16, f32>::new();
        let mut either = Vec::<Either<u16, f32>>::new();
        for _ in 0..10000 {
            if rng.gen_bool(0.5) {
                let val = rng.gen();
                tv.push_a(val);
                either.push(Either::Left(val));
            } else {
                let val = rng.gen();
                tv.push_b(val);
                either.push(Either::Right(val));
            }
        }

        for _ in 0..10000 {
            let i = rng.gen_range(0..tv.len());
            let l = tv.remove_b(i);
            if l.is_none() {
                continue;
            }
            let r = either.remove(i).right();

            assert_eq!(
                l, r,
                "Values not equal. TwoVec: {l:?}, Vec: {r:?} [Seed: {seed}]"
            );
        }
    }

    #[test]
    pub fn remove_either() {
        let mut rng = SmallRng::from_entropy();
        let seed: u64 = rng.gen();
        let mut rng = SmallRng::seed_from_u64(5846794401380124296);
        let mut tv = TwoVec::<u16, f32>::new();
        let mut either = Vec::<Either<u16, f32>>::new();
        for _ in 0..10 {
            if rng.gen_bool(0.5) {
                let val = rng.gen();
                tv.push_a(val);
                either.push(Either::Left(val));
            } else {
                let val = rng.gen();
                tv.push_b(val);
                either.push(Either::Right(val));
            }
        }

        while !tv.is_empty() {
            let i = rng.gen_range(0..tv.len());
            let l = tv.remove_either(i);
            let r = either.remove(i);

            assert_eq!(
                l, r,
                "Values not equal. TwoVec: {l:?}, Vec: {r:?} [Seed: {seed}]"
            );
        }
    }
}
