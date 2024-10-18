//! Contains the TwoVec methods that deal with the base memory allocation

use crate::TwoVec;

use std::{
    alloc::{alloc_zeroed, dealloc, handle_alloc_error, Layout},
    ptr::NonNull,
};

impl<A: Copy, B: Copy> Clone for TwoVec<A, B> {
    fn clone(&self) -> Self {
        let layout = Layout::array::<u8>(self.capacity).unwrap();
        let bitfield;
        let data;
        unsafe {
            let ptr = NonNull::new(alloc_zeroed(layout)).unwrap();
            bitfield = ptr;

            let offset = self.bitfield_size();
            data = ptr.byte_add(offset);
        }


        let mut result = Self {
            len: self.len,
            capacity: self.capacity,
            bitfield,
            data,
            a: self.a,
            b: self.b,
        };

        for val in self {
            result.push(val);
        }

        result
    }
}

impl<A: Copy, B: Copy> TwoVec<A, B> {
    // see: std library `RawVec`, prevents excrutiatingly tiny vecs
    const MIN_NON_ZER0_CAPACITY: usize = if Self::MAX_A_B == 1 {
        8
    } else if Self::MAX_A_B <= 1024 {
        4
    } else {
        1
    };

    pub(crate) fn grow(&mut self) {
        let (new_capacity, new_layout, bitfield_size) = if self.capacity == 0 {
            // "base" size of the allocation
            let base_size = Self::MAX_A_B * Self::MIN_NON_ZER0_CAPACITY;
            // number of bits needed to represent the "worst case" (i.e. 100% of capacity filled with the smaller element)
            let max_elements = base_size / Self::MIN_A_B;
            let bitfield_size = max_elements / 8;

            let new_capacity = base_size + bitfield_size;
            let new_layout = Layout::array::<u8>(new_capacity).unwrap();

            (new_capacity, new_layout, bitfield_size)
        } else {
            let new_capacity = self.capacity * 2;
            let new_layout = Layout::array::<u8>(new_capacity).unwrap();
            let bitfield_size = unsafe { self.data.byte_offset_from(self.bitfield) } as usize * 2;
            (new_capacity, new_layout, bitfield_size)
        };

        assert!(
            new_layout.size() <= isize::MAX as usize,
            "Allocation too large"
        );

        let allocation = unsafe { alloc_zeroed(new_layout) };

        let new_ptr = match NonNull::new(allocation) {
            Some(p) => p,
            None => handle_alloc_error(new_layout),
        };

        if self.capacity != 0 {
            unsafe {
                let offset = self.data.offset_from(self.bitfield) as usize;

                // necessary to do this in 2 steps instead of just calling `realloc` because the
                // offset between self.bitfield and self.data increases when the vec grows
                self.bitfield.copy_to_nonoverlapping(new_ptr, offset);

                // let bf = slice_from_raw_parts(new_ptr.as_ptr(), bitfield_size).as_ref().unwrap();
                // dbg!(bf);

                // let temp = slice_from_raw_parts(new_ptr.as_ptr(), new_capacity).as_ref().unwrap();
                // dbg!(temp);

                let new_data = new_ptr.add(bitfield_size);

                self.data
                    .copy_to_nonoverlapping(new_data, self.capacity - offset);

                // let temp = slice_from_raw_parts(self.bitfield.as_ptr(), self.capacity).as_ref().unwrap();
                //     dbg!(temp);

                // let temp = slice_from_raw_parts(new_ptr.as_ptr(), new_capacity).as_ref().unwrap();
                //     dbg!(temp);

                dealloc(
                    self.bitfield.as_ptr(),
                    Layout::array::<u8>(self.capacity).unwrap(),
                );
            }
        }

        self.bitfield = new_ptr;
        self.data = unsafe { self.bitfield.byte_add(bitfield_size) };
        self.capacity = new_capacity;
        assert_ne!(self.bitfield, self.data);
    }
}

impl<A: Copy, B: Copy> Drop for TwoVec<A, B> {
    // since all elements are `Copy`, they can't have drop implementations in the first place
    // so there's no need to call drop on them. We can just deallocate the whole memory region and
    // call it good
    fn drop(&mut self) {
        let layout = Layout::array::<u8>(self.capacity).unwrap();
        unsafe { dealloc(self.bitfield.as_ptr(), layout) }

    }
}