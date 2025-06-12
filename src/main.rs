use std::{fs, ops::Range, thread, time::Duration};

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
    inline: bool,
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
    let width = 1920;
    let height = 1080;

    let get_page = |x: u32, width: u32| {
        let mut container = Rect {
            bg_color: BgColor([1.0, 0.0, 0.0, 1.0]),
            pos: [x, 0],
            size: [width, height],
            children: vec![],
            inline: false,
        };

        for (i, &len) in line_lens.iter().enumerate() {
            let mut line = Rect {
                bg_color: BgColor([0.0, 1.0, 0.0, 1.0]),
                pos: [0, i as u32 * 35],
                size: [len as u32 * 16, 22],
                children: vec![],
                inline: false,
            };
            for i in 0..len {
                line.children.push(Rect {
                    bg_color: BgColor([0.0, 0.0, 1.0, 1.0]),
                    pos: [i as u32 * 16, 0],
                    size: [15, 22],
                    children: vec![],
                    inline: false,
                });
            }
            container.children.push(line);
        }

        let outline_width = width / 6;
        let outline_height = height / 2;

        let line_lens: Vec<_> = fs::read_to_string("./src/graphics_pipeline.rs")
            .unwrap()
            .lines()
            .take(200)
            .map(|line| line.len())
            .collect();

        for i in 0..5 {
            let mut outline = Rect {
                pos: [width - outline_width - (i as u32 * (outline_width + 20)), 0],
                size: [outline_width, outline_height],
                bg_color: BgColor([0.0, 1.0, 1.0, 1.0]),
                children: vec![],
                inline: false,
            };

            for (i, &len) in line_lens.iter().enumerate() {
                let mut line = Rect {
                    bg_color: BgColor([0.0, 1.0, 0.0, 1.0]),
                    pos: [0, i as u32 * 2],
                    size: [len as u32 * 2, 2],
                    children: vec![],
                    inline: false,
                };

                for i in 0..len {
                    line.children.push(Rect {
                        bg_color: BgColor([0.0, 0.0, 1.0, 1.0]),
                        pos: [i as u32 * 2, 0],
                        size: [2, 2],
                        children: vec![],
                        inline: false,
                    });
                }

                outline.children.push(line);
            }

            // container.children.push(outline);
        }

        container
    };

    let mut rectangles: Vec<_> = (0..2).map(|i| get_page(i * width / 2, width / 2)).collect();

    split_children(&mut rectangles);

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
    //             bg_color: BgColor([0.0, 1.0, 1.0, 0.9]),
    //             pos: [10, 10],
    //             size: [100, 100],
    //             inline: false,
    //         },
    //         Rect {
    //             children: vec![Rect {
    //                 children: vec![],
    //                 bg_color: BgColor([1.0, 1.0, 1.0, 1.0]),
    //                 pos: [10, 10],
    //                 size: [20, 20],
    //                 inline: false,
    //             }],
    //             bg_color: BgColor([1.0, 0.0, 1.0, 1.0]),
    //             pos: [50, 20],
    //             size: [100, 100],
    //             inline: false,
    //         },
    //     ],
    //     bg_color: BgColor([0.0, 0.0, 0.0, 1.0]),
    //     pos: [10, 10],
    //     size: [500, 500],
    //     inline: false,
    // }];

    unsafe {
        compute_tree::unsafe_main(&rectangles, width, height);
        // graphics_pipeline::unsafe_main(&rectangles, width, height);
    }
}

fn split_children(rects: &mut Vec<Rect>) {
    if rects.len() > 10 {
        let new_rects: Vec<_> = rects
            .chunks(10)
            .map(|chunk| {
                let x_min = chunk.iter().map(|rect| rect.pos[0]).min().unwrap();
                let y_min = chunk.iter().map(|rect| rect.pos[1]).min().unwrap();
                let x_max = chunk
                    .iter()
                    .map(|rect| rect.pos[0] + rect.size[0])
                    .max()
                    .unwrap();
                let y_max = chunk
                    .iter()
                    .map(|rect| rect.pos[1] + rect.size[1])
                    .max()
                    .unwrap();

                Rect {
                    pos: [x_min, y_min],
                    size: [x_max - x_min, y_max - y_min],
                    bg_color: BgColor([0.0, 0.0, 0.0, 0.0]),
                    children: chunk.to_vec(),
                    inline: true,
                }
            })
            .collect();

        *rects = new_rects;
    }

    for rect in rects.iter_mut() {
        split_children(&mut rect.children);
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
                inline: false,
            }
        })
        .collect()
}
