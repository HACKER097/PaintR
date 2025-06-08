use image::{open, Rgba, RgbaImage};
use rayon::prelude::*;
use imageproc::{drawing::draw_polygon_mut, point::Point, drawing::Blend};
use rand::Rng;

use std::collections::BinaryHeap;
use std::cmp::Reverse;
use std::cmp::Ordering;

pub struct BottomNHeap<T> {
    heap: BinaryHeap<Reverse<T>>,
}

impl<T: Ord> BottomNHeap<T> {
    pub fn new() -> Self {
        Self { heap: BinaryHeap::new() }
    }

    pub fn insert(&mut self, elem: T) {
        self.heap.push(Reverse(elem));
    }

    pub fn remove_bottom_n(&mut self, n: usize) -> Vec<T> {
        let mut removed = Vec::with_capacity(n);
        for _ in 0..n {
            if let Some(Reverse(val)) = self.heap.pop() {
                removed.push(val);
            } else {
                break;
            }
        }
        removed
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.heap.iter().map(|r| &r.0)
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

struct Triangle{
    points: [Point<i32>; 3],
    color: Rgba<u8>,
    score: Option<u64>,
}

impl PartialEq for Triangle {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for Triangle {}

impl PartialOrd for Triangle {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Triangle {
    fn cmp(&self, other: &Self) -> Ordering {
        // None is considered greater than any Some, so it goes last
        match (&self.score, &other.score) {
            (Some(a), Some(b)) => a.cmp(b),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => Ordering::Equal,
        }
    }
}

impl Triangle {
    fn get_random(max_x: u32, max_y: u32) -> Self{
        let mut rng = rand::rng();
        Self{
            points: [Point::new(
                rng.random_range(0..=max_x) as i32,
                rng.random_range(0..=max_y) as i32,
            ), Point::new(
                rng.random_range(0..=max_x) as i32,
                rng.random_range(0..=max_y) as i32,
            ), Point::new(
                rng.random_range(0..=max_x) as i32,
                rng.random_range(0..=max_y) as i32,
            )],
            color: Rgba([
                rng.random_range(0..=255),
                rng.random_range(0..=255),
                rng.random_range(0..=255),
                rng.random_range(0..=255),
            ]),
            score: None,
        }
    }

    fn get_mutated(&self, temperature: f32, max_x: u32, max_y: u32) -> Self {
        let mut rng = rand::rng();

        let dx = (temperature * max_x as f32).round() as i32;
        let dy = (temperature * max_y as f32).round() as i32;
        let dc = (temperature * 255.0).round() as i16;

        let mut points = [Point::new(0, 0); 3];
        for i in 0..3 {
            let p = self.points[i];
            let x = (p.x + rng.random_range(-dx..=dx)).clamp(0, max_x as i32);
            let y = (p.y + rng.random_range(-dy..=dy)).clamp(0, max_y as i32);
            points[i] = Point::new(x, y);
        }

        let mut color = [0u8; 4];
        for i in 0..4 {
            let c = self.color.0[i] as i16 + rng.random_range(-dc..=dc);
            color[i] = c.clamp(0, 255) as u8;
        }

        Self {
            points,
            color: Rgba(color),
            score: self.score,
        }
    }

    fn draw(&self, image: &RgbaImage) -> RgbaImage{
        if self.points[0] == self.points[1] || self.points[1] == self.points[2] || self.points[0] == self.points[2] {
            return image.clone(); // skip drawing invalid triangle
        }
        let mut blended_canvas = Blend(image.clone());
        draw_polygon_mut(&mut blended_canvas, &self.points, self.color);
        blended_canvas.0
    }
}

fn diff(img1: &RgbaImage, img2: &RgbaImage) -> u64 {
    assert_eq!(img1.dimensions(), img2.dimensions());

    let buf1 = img1.as_raw();
    let buf2 = img2.as_raw();
    assert_eq!(buf1.len(), buf2.len());

    buf1.par_chunks(1024)
        .zip(buf2.par_chunks(1024))
        .map(|(chunk1, chunk2)| {
            let mut sum = 0u64;
            for i in 0..chunk1.len() {
                let diff = chunk1[i] as i32 - chunk2[i] as i32;
                sum += (diff * diff) as u64;
            }
            sum
        })
        .sum()
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::fs;
    use std::io::{self, Write};
    use rayon::prelude::*;

    let target_image = open("mona.jpg")?.to_rgba8();
    let (w, h) = target_image.dimensions();
    fs::create_dir_all("images")?;

    // Get input
    let mut input = String::new();
    print!("Population size: ");
    io::stdout().flush()?;
    io::stdin().read_line(&mut input)?;
    let population_size: usize = input.trim().parse()?;
    input.clear();

    print!("Number of iterations per triangle: ");
    io::stdout().flush()?;
    io::stdin().read_line(&mut input)?;
    let num_iterations: usize = input.trim().parse()?;
    input.clear();

    print!("Number of triangles: ");
    io::stdout().flush()?;
    io::stdin().read_line(&mut input)?;
    let num_triangles: usize = input.trim().parse()?;
    input.clear();

    print!("Initial temperature (e.g., 0.1): ");
    io::stdout().flush()?;
    io::stdin().read_line(&mut input)?;
    let mut temperature: f32 = input.trim().parse()?;
    input.clear();


    print!("Save every Nth triangle (e.g., 10): ");
    io::stdout().flush()?;
    io::stdin().read_line(&mut input)?;
    let save_every_n: usize = input.trim().parse()?;
    input.clear();

    let mut canvas = RgbaImage::from_pixel(w, h, Rgba([0, 0, 0, 255]));

    for triangle_index in 0..num_triangles {
        // Step 1: generate population in parallel
        let initial_population: Vec<Triangle> = (0..population_size)
            .into_par_iter()
            .map(|_| {
                let mut tri = Triangle::get_random(w, h);
                tri.score = Some(diff(&target_image, &tri.draw(&canvas)));
                tri
            })
            .collect();

        let mut heap = BottomNHeap::new();
        for tri in initial_population {
            heap.insert(tri);
        }

        for iter in 0..num_iterations {
            // Step 3: take top 50%
            let survivors = heap.remove_bottom_n(population_size / 2);
            heap = BottomNHeap::new(); // clear old gen

            // Step 4: mutate in parallel and re-evaluate
            let mutated_population: Vec<Triangle> = survivors
                .into_par_iter()
                .map(|s| {
                    let mut mutated = s.get_mutated(temperature, w, h);
                    mutated.score = Some(diff(&target_image, &mutated.draw(&canvas)));
                    mutated
                })
                .collect();

            for m in mutated_population {
                heap.insert(m);
            }

            // Cooldown
            temperature *= 0.99;

            // Log progress
            if let Some(best) = heap.iter().min_by_key(|t| t.score.unwrap()) {
                println!(
                    "Triangle {:03}, Iteration {:03} — Best Score: {}",
                    triangle_index,
                    iter,
                    best.score.unwrap()
                );
            }
        }

        if let Some(best) = heap.iter().min_by_key(|t| t.score.unwrap()) {
            canvas = best.draw(&canvas);

            if triangle_index % save_every_n == 0 {
                let filename = format!("images/step_{:04}.png", triangle_index);
                canvas.save(&filename)?;
            }
        }
    }

    Ok(())
}
