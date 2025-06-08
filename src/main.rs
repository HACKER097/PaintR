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

impl<T: Send> IntoParallelIterator for BottomNHeap<T> {
    type Item = T;
    type Iter = rayon::vec::IntoIter<T>;

    fn into_par_iter(self) -> Self::Iter {
        self.heap
            .into_iter()
            .map(|r| r.0)
            .collect::<Vec<_>>()
            .into_par_iter()
    }
}

impl<T: Clone + Send + Sync> BottomNHeap<T> {
    pub fn par_iter_cloned(&self) -> impl ParallelIterator<Item = T> {
        self.heap
            .iter()
            .map(|r| r.0.clone())
            .collect::<Vec<_>>()
            .into_par_iter()
    }
}

#[derive(Clone)]
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
            return image.clone();
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
    let temperature: f32 = input.trim().parse()?;
    input.clear();

    print!("Emergency Iterations (e.g., 100): ");
    io::stdout().flush()?;
    io::stdin().read_line(&mut input)?;
    let emergency_iterations: usize = input.trim().parse()?;
    input.clear();

    print!("Save every Nth triangle (e.g., 10): ");
    io::stdout().flush()?;
    io::stdin().read_line(&mut input)?;
    let save_every_n: usize = input.trim().parse()?;
    input.clear();

    let mut canvas = RgbaImage::from_pixel(w, h, Rgba([0, 0, 0, 255]));

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

    for triangle_index in 0..num_triangles {
        println!("\n--- Evolving Triangle #{} (out of {}) ---", triangle_index + 1, num_triangles);
        println!("  Initial Heap Size: {}", heap.len());

        let score_before_this_triangle = diff(&target_image, &canvas);
        println!("  Score of canvas before this triangle: {}", score_before_this_triangle);

        let mut iter = 0;
        let mut best_candidate_score_ever_this_triangle_evolution = u64::MAX;

        loop {
            let num_survivors = population_size / 2;
            let num_mutants_to_generate = population_size - num_survivors;

            let mut current_triangles: Vec<Triangle> = heap.into_par_iter().collect();
            current_triangles.par_sort_unstable_by_key(|t| t.score.unwrap_or(u64::MAX));

            let survivors: Vec<Triangle> = current_triangles.drain(0..num_survivors).collect();

            let mutated_offspring: Vec<Triangle> = (0..num_mutants_to_generate)
                .into_par_iter()
                .map(|_| {
                    let mut rng = rand::rng();
                    let parent_index = rng.random_range(0..survivors.len());
                    let s = &survivors[parent_index];
                    let mut mutated = s.get_mutated(temperature, w, h);
                    mutated.score = Some(diff(&target_image, &mutated.draw(&canvas)));
                    mutated
                })
                .collect();

            heap = BottomNHeap::new();
            for s in survivors {
                heap.insert(s);
            }
            for m in mutated_offspring {
                heap.insert(m);
            }

            let current_iteration_best_candidate_score = heap.iter().min_by_key(|t| t.score.unwrap_or(u64::MAX)).map(|t| t.score.unwrap_or(u64::MAX)).unwrap_or(u64::MAX);

            if current_iteration_best_candidate_score < best_candidate_score_ever_this_triangle_evolution {
                best_candidate_score_ever_this_triangle_evolution = current_iteration_best_candidate_score;
            }

            println!(
                "  Iteration {:03} — Best: {}, Best Overall: {}, Temp: {:.5}",
                iter + 1,
                current_iteration_best_candidate_score,
                best_candidate_score_ever_this_triangle_evolution,
                temperature,
            );

            // Condition to stop:
            // 1. We have completed the minimum required iterations (`num_iterations`)
            // AND the best candidate found so far for this triangle *would improve* the overall image.
            if iter >= num_iterations - 1 && best_candidate_score_ever_this_triangle_evolution < score_before_this_triangle {
                println!("  Stopping: Achieved overall image improvement within or after initial iterations.");
                break;
            }

            // 2. We have completed the minimum required iterations
            // AND we have also completed the emergency iterations
            // AND we have *not* achieved overall image improvement.
            if iter >= num_iterations - 1 + emergency_iterations {
                println!("  Stopping: Reached emergency iteration limit without improving overall image score.");
                break;
            }

            iter += 1;
        }

        if let Some(best_triangle_for_this_gen) = heap.iter().min_by_key(|t| t.score.unwrap_or(u64::MAX)) {
            canvas = best_triangle_for_this_gen.draw(&canvas);

            if triangle_index % save_every_n == 0 {
                let filename = format!("images/step_{:04}.png", triangle_index);
                println!("  Saving current image to {}", filename);
                canvas.save(&filename)?;
            }
        }
    }

    canvas.save("images/final_result.png")?;
    println!("\nGenetic algorithm finished! Final image saved to images/final_result.png");

    Ok(())
}
