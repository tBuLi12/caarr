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
    pos: [u32; 2],
    size: [u32; 2],
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
    // let width = 1920;
    // let height = 1080;

    let get_page = |x: u32, width: u32| {
        let mut container = Rect {
            bg_color: BgColor([1.0, 0.0, 0.0, 1.0]),
            pos: [x, 0],
            size: [width, height],
            children: vec![],
        };

        for (i, &len) in line_lens.iter().enumerate() {
            let mut line = Rect {
                bg_color: BgColor([0.0, 1.0, 0.0, 0.0]),
                pos: [0, i as u32 * 35],
                size: [len as u32 * 16, 22],
                children: vec![],
            };
            for i in 0..len {
                line.children.push(Rect {
                    bg_color: BgColor([0.0, 0.0, 1.0, 0.7]),
                    pos: [i as u32 * 16, 0],
                    size: [15, 22],
                    children: vec![],
                });
            }
            container.children.push(line);
        }

        container
    };

    let rectangles: Vec<_> = (0..5).map(|i| get_page(i * width / 5, width / 5)).collect();

    // let rectangles = get_rects(
    //     &mut StdRng::seed_from_u64(4827493030),
    //     0,
    //     0.0..(width as f32),
    //     0.0..(height as f32),
    //     (width as f32) * 3.0 / 5.0,
    //     (height as f32) * 3.0 / 5.0,
    // );

    // let rectangles = vec![Rect {
    //     children: vec![
    //         Rect {
    //             children: vec![],
    //             bg_color: BgColor([0.0, 1.0, 1.0, 1.0]),
    //             pos: [20.0, 20.0],
    //             size: [100.0, 100.0],
    //         },
    //         Rect {
    //             children: vec![],
    //             bg_color: BgColor([1.0, 0.0, 1.0, 1.0]),
    //             pos: [200.0, 20.0],
    //             size: [100.0, 100.0],
    //         },
    //     ],
    //     bg_color: BgColor([0.0, 0.0, 0.0, 1.0]),
    //     pos: [10.0, 10.0],
    //     size: [500.0, 500.0],
    // }];

    unsafe {
        compute_tree::unsafe_main(&rectangles, width, height);
        // graphics_pipeline::unsafe_main(&rectangles, width, height);
    }
}

fn get_rects(
    rng: &mut StdRng,
    depth: u32,
    x_bounds: Range<u32>,
    y_bounds: Range<u32>,
    max_width: u32,
    max_height: u32,
) -> Vec<Rect> {
    if depth == 4 {
        return vec![];
    }

    (0..10)
        .map(|_| {
            let width = rng.random_range((max_width / 2)..max_width);
            let height = rng.random_range((max_height / 2)..max_height);
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
                    0..(width),
                    0..(height),
                    width / 2,
                    height / 2,
                ),
            }
        })
        .collect()
}
