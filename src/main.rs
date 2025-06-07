use std::{fs, ops::Range};

use rand::{rngs::StdRng, Rng, SeedableRng};

mod compute_tree;
mod graphics_pipeline;

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
struct BgColor([f32; 4]);

#[derive(Debug, Clone)]
#[repr(C)]
struct Rect {
    pos: [f32; 2],
    size: [f32; 2],
    bg_color: BgColor,
    children: Vec<Rect>,
}

fn main() {
    let line_lens: Vec<_> = fs::read_to_string("./src/graphics_pipeline.rs")
        .unwrap()
        .lines()
        .skip(100)
        .take(60)
        .map(|line| line.len())
        .collect();

    let width = 7680;
    let height = 4320 / 2;
    let width = 1920 / 2;
    let height = 1080;

    let get_page = |x: u32, width: u32| {
        let mut container = Rect {
            bg_color: BgColor([1.0, 0.0, 0.0, 1.0]),
            pos: [x as f32, 0.0],
            size: [width as f32, height as f32],
            children: vec![],
        };

        for (i, &len) in line_lens.iter().enumerate() {
            let mut line = Rect {
                bg_color: BgColor([0.0, 0.0, 0.0, 0.0]),
                pos: [0.0, i as f32 * 35.0],
                size: [len as f32 * 16.0, 20.0],
                children: vec![],
            };
            for i in 0..len {
                line.children.push(Rect {
                    bg_color: BgColor([0.0, 0.0, 1.0, 0.7]),
                    pos: [i as f32 * 16.0, 0.0],
                    size: [15.0, 22.0],
                    children: vec![],
                });
            }
            container.children.push(line);
        }

        container
    };

    let rectangles: Vec<_> = (0..1).map(|i| get_page(i * width / 1, width / 1)).collect();

    // let rectangles = get_rects(
    //     &mut StdRng::seed_from_u64(4827493030),
    //     0,
    //     0.0..(width as f32),
    //     0.0..(height as f32),
    //     (width as f32) * 3.0 / 5.0,
    //     (height as f32) * 3.0 / 5.0,
    // );

    let rectangles = vec![Rect {
        children: vec![
            Rect {
                children: vec![],
                bg_color: BgColor([0.0, 1.0, 1.0, 1.0]),
                pos: [20.0, 20.0],
                size: [100.0, 100.0],
            },
            Rect {
                children: vec![],
                bg_color: BgColor([1.0, 0.0, 1.0, 1.0]),
                pos: [200.0, 20.0],
                size: [100.0, 100.0],
            },
        ],
        bg_color: BgColor([0.0, 0.0, 0.0, 1.0]),
        pos: [10.0, 10.0],
        size: [500.0, 500.0],
    }];

    unsafe {
        compute_tree::unsafe_main(&rectangles, width, height);
        // graphics_pipeline::unsafe_main(&rectangles, width, height);
    }
}

fn get_rects(
    rng: &mut StdRng,
    depth: u32,
    x_bounds: Range<f32>,
    y_bounds: Range<f32>,
    max_width: f32,
    max_height: f32,
) -> Vec<Rect> {
    if depth == 4 {
        return vec![];
    }

    (0..10)
        .map(|_| {
            let width = rng.random_range((max_width * 0.8)..max_width);
            let height = rng.random_range((max_height * 0.8)..max_height);
            let x = rng.random_range(x_bounds.start..(x_bounds.end - width));
            let y = rng.random_range(y_bounds.start..(y_bounds.end - height));
            Rect {
                pos: [x, y],
                size: [width, height],
                bg_color: BgColor([
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    rng.random_range(0.0..1.0),
                    1.0,
                ]),
                children: get_rects(
                    rng,
                    depth + 1,
                    0.0..(width),
                    0.0..(height),
                    width / 1.3,
                    height / 1.3,
                ),
            }
        })
        .collect()
}
