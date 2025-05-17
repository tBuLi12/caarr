pub struct Ref {
    pub ref_count: u32,
    pub rect_idx: u32,
}

pub struct RefAllocator {
    refs: Vec<Ref>,
    next_free: u32,
}

impl RefAllocator {
    pub fn new() -> Self {
        Self {
            refs: vec![],
            next_free: 0,
        }
    }

    pub fn alloc(&mut self) -> u32 {
        if let Some(idx) = self.next_free.checked_sub(1) {
            self.next_free = self.refs[idx as usize].rect_idx;
            idx
        } else {
            let idx = self.refs.len() as u32;
            self.refs.push(Ref {
                ref_count: 0,
                rect_idx: 0,
            });
            idx
        }
    }

    pub fn free(&mut self, ref_idx: u32) {
        self.refs[ref_idx as usize] = Ref {
            rect_idx: self.next_free,
            ref_count: 0,
        };
        self.next_free = ref_idx + 1;
    }

    pub fn resolve(&mut self, ref_idx: u32) -> &mut Ref {
        &mut self.refs[ref_idx as usize]
    }
}
