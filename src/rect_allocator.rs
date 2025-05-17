use std::{
    alloc::{self, Layout},
    mem,
    ops::Range,
    ptr,
};

#[derive(Clone, Copy, Debug)]
struct Hole {
    idx: u32,
    length: u32,
}

pub struct Allocator<T> {
    slots: *mut T,
    size: usize,
    holes: Vec<Hole>,
}

impl<T> Allocator<T> {
    const _ASSERT_SIZED: () = assert!(mem::size_of::<T>() != 0);

    pub fn new() -> Self {
        let size = 20_000;
        let ptr = unsafe { alloc::alloc(Layout::array::<T>(size).unwrap()) };

        Self {
            slots: ptr as *mut T,
            size,
            holes: vec![Hole {
                idx: 0,
                length: size as u32,
            }],
        }
    }

    pub fn alloc(&mut self, length: u32) -> u32 {
        let idx = self._alloc(length);
        idx
    }

    pub fn _alloc(&mut self, length: u32) -> u32 {
        for (i, hole) in self.holes.iter_mut().enumerate() {
            if hole.length > length {
                hole.length -= length;
                return hole.idx + hole.length;
            }
            if hole.length == length {
                let hole_idx = hole.idx;
                self.holes.remove(i);
                return hole_idx;
            }
        }
        panic!("No space left");
    }

    pub unsafe fn free(&mut self, idx: u32, length: u32) {
        assert!(length > 0);

        let next_hole_idx = self.holes.partition_point(|h| h.idx < idx);
        let before_hole =
            next_hole_idx < self.holes.len() && idx + length == self.holes[next_hole_idx].idx;
        let after_hole = next_hole_idx != 0 && {
            let hole = self.holes[next_hole_idx - 1];
            hole.idx + hole.length == idx
        };

        match (before_hole, after_hole) {
            (true, true) => {
                let right_length = self.holes.remove(next_hole_idx).length;
                self.holes[next_hole_idx - 1].length += length + right_length;
            }
            (true, false) => {
                let hole = &mut self.holes[next_hole_idx];
                hole.length += length;
                hole.idx = idx;
            }
            (false, true) => {
                self.holes[next_hole_idx - 1].length += length;
            }
            (false, false) => {
                self.holes.insert(next_hole_idx, Hole { idx, length });
            }
        }
    }

    pub unsafe fn resolve(&self, idx: u32) -> *mut T {
        self.slots.add(idx as usize)
    }

    pub unsafe fn resolve_range(&self, range: Range<u32>) -> *mut [T] {
        ptr::slice_from_raw_parts_mut(self.resolve(range.start), range.len())
    }
}
