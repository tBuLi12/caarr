use std::{
    alloc::{self, Layout},
    ops::Range,
    ptr, slice,
};

use crate::RectData;

struct RectAllocator {
    items: *mut RectData,
    size: usize,
    holes: Vec<Hole>,
}

impl RectAllocator {
    fn new() -> Self {
        let size = 20_000;
        let ptr = unsafe { alloc::alloc(Layout::array::<RectData>(size).unwrap()) };

        Self {
            items: ptr as *mut RectData,
            size,
            holes: vec![Hole {
                idx: 0,
                length: size as u32,
            }],
        }
    }

    fn alloc(&mut self, length: u32) -> RectIdx {
        for (i, hole) in self.holes.iter_mut().enumerate() {
            if hole.length > length {
                let idx = hole.idx;
                hole.idx += length;
                hole.length -= length;

                if hole.length == 0 {
                    self.holes.remove(i);
                }

                return RectIdx { inner: idx };
            }
        }
        panic!("No space left");
    }

    unsafe fn free(&mut self, rect_idx: RectIdx, length: u32) {
        assert!(length > 0);

        let idx = rect_idx.inner;
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

    unsafe fn resolve(&self, rect_idx: RectIdx) -> *mut RectData {
        self.items.add(rect_idx.inner as usize)
    }

    fn allocated_span(&self) -> &[RectData] {
        let allocated_size = self.size - self.holes.last().map_or(0, |hole| hole.length as usize);
        unsafe { slice::from_raw_parts(self.items, allocated_size) }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
struct RectIdx {
    inner: u32,
}

#[derive(Clone, Copy, Debug)]
struct Hole {
    idx: u32,
    length: u32,
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct RefIdx {
    inner: u32,
}

struct RefAllocator {
    items: Vec<RefData>,
    next_free: u32,
}

impl RefAllocator {
    fn new() -> Self {
        Self {
            items: vec![],
            next_free: 0,
        }
    }

    fn alloc(&mut self, ref_data: RefData) -> RefIdx {
        let idx = if let Some(idx) = self.next_free.checked_sub(1) {
            self.next_free = self.items[idx as usize].ref_count;
            self.items[idx as usize] = ref_data;
            idx
        } else {
            let idx = self.items.len() as u32;
            self.items.push(ref_data);
            idx
        };

        RefIdx { inner: idx }
    }

    fn free(&mut self, ref_idx: RefIdx) {
        self.items[ref_idx.inner as usize].ref_count = self.next_free;
        self.next_free = ref_idx.inner + 1;
    }

    fn resolve(&mut self, ref_idx: RefIdx) -> &mut RefData {
        &mut self.items[ref_idx.inner as usize]
    }

    fn allocated_span(&self) -> &[RefData] {
        &self.items
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RefData {
    ref_count: u32,
    rect_idx: RectIdx,
}

pub struct RectStorage {
    ref_allocator: RefAllocator,
    rect_allocator: RectAllocator,
}

impl RectStorage {
    pub fn new() -> Self {
        Self {
            ref_allocator: RefAllocator::new(),
            rect_allocator: RectAllocator::new(),
        }
    }

    pub fn new_rect(&mut self) -> RefIdx {
        let rect_idx = self.rect_allocator.alloc(1);
        let ref_idx = self.ref_allocator.alloc(RefData {
            rect_idx,
            ref_count: 1,
        });

        unsafe {
            self.rect_allocator.resolve(rect_idx).write(RectData {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
                bg_color: [0, 0, 0, 0],
                tex_position: [0, 0],
                parent_idx: 0,
                children_start: 0,
                children_end: 0,
                children_capacity: 0,
                ref_idx: ref_idx.inner + 1,
                fill_kind: 0,
            });
        }

        ref_idx
    }

    pub fn new_child(&mut self, ref_idx: RefIdx) -> RefIdx {
        self.ensure_not_full(ref_idx);

        unsafe {
            let rect_idx = self.ref_allocator.resolve(ref_idx).rect_idx;
            let rect = &mut *self.rect_allocator.resolve(rect_idx);

            let child_rect_idx = RectIdx {
                inner: rect.children_end,
            };

            rect.children_end += 1;

            let child_ref_idx = self.ref_allocator.alloc(RefData {
                ref_count: 1,
                rect_idx: child_rect_idx,
            });

            self.rect_allocator.resolve(child_rect_idx).write(RectData {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
                bg_color: [0, 0, 0, 0],
                tex_position: [0, 0],
                parent_idx: rect_idx.inner + 1,
                children_start: 0,
                children_end: 0,
                children_capacity: 0,
                ref_idx: child_ref_idx.inner + 1,
                fill_kind: 0,
            });

            child_ref_idx
        }
    }

    pub fn append_child(&mut self, ref_idx: RefIdx, child_ref_idx: RefIdx) {
        self.ensure_not_full(ref_idx);
        unsafe {
            let ref_data = *self.ref_allocator.resolve(ref_idx);
            let rect_data = &mut *self.rect_allocator.resolve(ref_data.rect_idx);
            let child_ref_data = *self.ref_allocator.resolve(child_ref_idx);
            let child_rect_data = &mut *self.rect_allocator.resolve(child_ref_data.rect_idx);
            child_rect_data.parent_idx = ref_data.rect_idx.inner + 1;
            rect_data.children_end += 1;
            let dst = rect_data.children_end - 1;

            self.move_rect(child_ref_data.rect_idx, RectIdx { inner: dst });
        }
    }

    pub fn clear_children(&mut self, ref_idx: RefIdx) {
        unsafe {
            let rect_data = &mut *self.resolve_rect(ref_idx);
            self.unlink_from_parent(rect_data.children_start..rect_data.children_end);
            rect_data.children_end = rect_data.children_start;
        }
    }

    pub fn increment_ref_count(&mut self, ref_idx: RefIdx) {
        self.ref_allocator.resolve(ref_idx).ref_count += 1;
    }

    pub fn decrement_ref_count(&mut self, ref_idx: RefIdx) {
        let ref_data = self.ref_allocator.resolve(ref_idx);
        // eprintln!("decrementing ref count of {:?}", ref_data.rect_idx);
        ref_data.ref_count -= 1;

        if ref_data.ref_count == 0 {
            let rect_idx = ref_data.rect_idx;
            self.ref_allocator.free(ref_idx);
            let rect_data = unsafe { &mut *self.rect_allocator.resolve(rect_idx) };
            rect_data.ref_idx = 0;
            if rect_data.parent_idx == 0 {
                drop(rect_data);
                self.free_rect(rect_idx);
            }
        }
    }

    unsafe fn move_rect(&mut self, src: RectIdx, dst: RectIdx) {
        debug_assert_ne!(src.inner, dst.inner);

        let source = self.rect_allocator.resolve(src);
        let src_data = source.read();
        self.rect_allocator.resolve(dst).write(src_data);

        for child_child_idx in src_data.children_start..src_data.children_end {
            let child_child = &mut *self.rect_allocator.resolve(RectIdx {
                inner: child_child_idx,
            });
            child_child.parent_idx = dst.inner + 1;
        }

        if let Some(ref_idx) = src_data.ref_idx.checked_sub(1) {
            let src_ref_idx = RefIdx { inner: ref_idx };
            let src_ref_data = self.ref_allocator.resolve(src_ref_idx);
            src_ref_data.rect_idx = dst;
        }
    }

    fn ensure_not_full(&mut self, ref_idx: RefIdx) {
        unsafe {
            let rect = self.resolve_rect(ref_idx).read();

            let child_count = rect.children_end - rect.children_start;

            if child_count == rect.children_capacity {
                // eprintln!("new child - realloc");
                let new_length = if child_count == 0 { 4 } else { child_count * 2 };
                // eprintln!(
                //     "reallocing children of {:?}: {} -> {}",
                //     self.ref_allocator.resolve(ref_idx).rect_idx,
                //     rect.children_capacity,
                //     new_length
                // );
                let new_children_start = self.rect_allocator.alloc(new_length);

                for offset in 0..child_count {
                    self.move_rect(
                        RectIdx {
                            inner: rect.children_start + offset,
                        },
                        RectIdx {
                            inner: new_children_start.inner + offset,
                        },
                    );
                }

                if child_count != 0 {
                    self.rect_allocator.free(
                        RectIdx {
                            inner: rect.children_start,
                        },
                        child_count,
                    );
                }

                {
                    let rect_data = &mut *self.resolve_rect(ref_idx);
                    rect_data.children_start = new_children_start.inner;
                    rect_data.children_end = new_children_start.inner + child_count;
                    rect_data.children_capacity = new_length;
                }
            }
        };
    }

    fn free_rect(&mut self, rect_idx: RectIdx) {
        let rect_data = unsafe { self.rect_allocator.resolve(rect_idx).read() };
        unsafe { self.rect_allocator.free(rect_idx, 1) };

        self.unlink_from_parent(rect_data.children_start..rect_data.children_end);
    }

    fn unlink_from_parent(&mut self, range: Range<u32>) {
        for child_idx in range {
            let child_data =
                unsafe { &mut *self.rect_allocator.resolve(RectIdx { inner: child_idx }) };
            child_data.parent_idx = 0;

            if child_data.ref_idx == 0 {
                let range = child_data.children_start..child_data.children_end;
                self.unlink_from_parent(range);
            } else {
                let new_idx = self.rect_allocator.alloc(1);
                unsafe {
                    self.move_rect(RectIdx { inner: child_idx }, new_idx);
                }
            }
        }
    }

    pub fn set_bg_color(&mut self, ref_idx: RefIdx, color: [u8; 4]) {
        unsafe {
            let rect = &mut *self.resolve_rect(ref_idx);
            rect.bg_color = color;
            rect.fill_kind = 0;
        }
    }

    pub fn set_tex_position(&mut self, ref_idx: RefIdx, tex_position: [u32; 2]) {
        unsafe {
            let rect = &mut *self.resolve_rect(ref_idx);
            rect.tex_position = tex_position;
            rect.fill_kind = 1;
        }
    }

    pub fn set_size(&mut self, ref_idx: RefIdx, width: u32, height: u32) {
        unsafe {
            let rect = &mut *self.resolve_rect(ref_idx);
            rect.width = width;
            rect.height = height;
        }
    }

    pub fn set_pos(&mut self, ref_idx: RefIdx, x: u32, y: u32) {
        unsafe {
            let rect = &mut *self.resolve_rect(ref_idx);
            rect.x = x;
            rect.y = y;
        }
    }

    pub fn allocated_span(&self) -> &[RectData] {
        self.rect_allocator.allocated_span()
    }

    pub fn allocated_ref_span(&self) -> &[RefData] {
        self.ref_allocator.allocated_span()
    }

    // pub fn debug(&mut self, rect: &Rect) {
    //     let rect_ref = self.ref_allocator.resolve(rect.ref_ref).rect_ref;
    //     self.debug_rec(rect_ref, 0);
    // }

    // fn debug_rec(&mut self, rect_idx: u32, depth: u32) {
    //     unsafe {
    //         let rect = self.rect_allocator.resolve(rect_idx).read();
    //         for _ in 0..depth {
    //             eprint!("  ");
    //         }
    //         // eprintln!("rect {}..{}", rect.children_start, rect.children_end);
    //         eprintln!("rect {rect_idx} in {:?}", rect.parent_idx.checked_sub(1));
    //         for child_idx in rect.children_start..rect.children_end {
    //             self.debug_rec(child_idx, depth + 1);
    //         }
    //     }
    // }

    fn resolve_rect(&mut self, ref_idx: RefIdx) -> *mut RectData {
        unsafe {
            self.rect_allocator
                .resolve(self.ref_allocator.resolve(ref_idx).rect_idx)
        }
    }
}
