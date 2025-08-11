use std::cell::RefCell;

#[derive(Clone)]
pub(crate) struct RectData {
    pub(crate) x: i32,
    pub(crate) y: i32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) bg_color: [u8; 4],
    pub(crate) tex_position: [u32; 2],
    pub(crate) fill_kind: u32,
    pub(crate) ref_count: u32,
    pub(crate) parent_idx: u32,
    pub(crate) children: Vec<Rect>,
}

pub struct Rect {
    inner: u32,
}

impl Rect {
    pub fn new() -> Self {
        STORAGE.with_borrow_mut(|this| {
            let rect_data = RectData {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
                bg_color: [0; 4],
                tex_position: [0; 2],
                fill_kind: 0,
                parent_idx: 0,
                ref_count: 1,
                children: vec![],
            };

            let idx = if let Some(idx) = this.next_free.checked_sub(1) {
                this.next_free = this.items[idx as usize].ref_count;
                this.items[idx as usize] = rect_data;
                idx
            } else {
                let idx = this.items.len() as u32;
                this.items.push(rect_data);
                idx
            };

            Rect { inner: idx }
        })
    }

    pub fn clear_children(&self) {
        STORAGE.with_borrow_mut(|this| {
            for child in &mut this.items[self.inner as usize].children {
                this.items[child.inner as usize].parent_idx = 0;
            }
            this.items[self.inner as usize].children.clear()
        })
    }

    pub fn set_bg_color(&self, color: [u8; 4]) {
        STORAGE.with_borrow_mut(|this| this.items[self.inner as usize].bg_color = color)
    }

    pub fn set_tex_position(&self, tex_position: [u32; 2]) {
        STORAGE.with_borrow_mut(|this| this.items[self.inner as usize].tex_position = tex_position)
    }

    pub fn set_size(&self, width: u32, height: u32) {
        STORAGE.with_borrow_mut(|this| {
            let data = &mut this.items[self.inner as usize];
            data.width = width;
            data.height = height;
        })
    }

    pub fn set_pos(&self, x: i32, y: i32) {
        STORAGE.with_borrow_mut(|this| {
            let data = &mut this.items[self.inner as usize];
            data.x = x;
            data.y = y;
        })
    }

    pub fn get_size(&self) -> (u32, u32) {
        STORAGE.with_borrow_mut(|this| {
            let data = &this.items[self.inner as usize];
            (data.width, data.height)
        })
    }

    pub fn append_child(&self, child: Rect) {
        STORAGE.with_borrow_mut(|this| {
            let parent_idx = &mut this.items[child.inner as usize].parent_idx;
            if *parent_idx != 0 {
                panic!("rect already has a parent");
            }
            *parent_idx = self.inner + 1;
            this.items[self.inner as usize].children.push(child)
        })
    }

    pub(crate) fn get_flat_idx_list(&self) -> Vec<u32> {
        let mut indices = vec![];

        STORAGE.with_borrow(|this| {
            fn push(indices: &mut Vec<u32>, rects: &[RectData], idx: u32) {
                indices.push(idx);
                for child in &rects[idx as usize].children {
                    push(indices, rects, child.inner);
                }
            }
            push(&mut indices, &this.items, self.inner);
        });

        indices
    }
}

impl Drop for Rect {
    fn drop(&mut self) {
        STORAGE.with_borrow_mut(|this| {
            let rc = &mut this.items[self.inner as usize].ref_count;
            *rc -= 1;

            if *rc == 0 {
                this.items[self.inner as usize] = RectData {
                    x: 0,
                    y: 0,
                    width: 0,
                    height: 0,
                    bg_color: [0; 4],
                    tex_position: [0; 2],
                    fill_kind: 0,
                    ref_count: this.next_free,
                    parent_idx: 0,
                    children: vec![],
                };
                this.next_free = self.inner + 1;
            }
        })
    }
}

impl Clone for Rect {
    fn clone(&self) -> Self {
        STORAGE.with_borrow_mut(|this| {
            this.items[self.inner as usize].ref_count += 1;
            Rect { inner: self.inner }
        })
    }
}

pub(crate) fn with_all_rects<T>(fun: impl FnOnce(&[RectData]) -> T) -> T {
    STORAGE.with_borrow(|this| fun(&this.items))
}

struct RectAllocator {
    items: Vec<RectData>,
    next_free: u32,
}

impl RectAllocator {
    fn new() -> Self {
        Self {
            items: vec![],
            next_free: 0,
        }
    }
}

thread_local! {
    static STORAGE: RefCell<RectAllocator> = RefCell::new(RectAllocator::new());
}
