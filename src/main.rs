use image::{open, Rgba, RgbaImage};
use imageproc::{drawing::draw_polygon_mut, point::Point, drawing::Blend};
use rand::Rng;

struct Triangle{
    points: [Point<i32>; 3],
    color: Rgba<u8>,
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
            ]) }
    }

    fn draw(self, image: &RgbaImage) -> RgbaImage{
        let mut blended_canvas = Blend(image.clone());
        draw_polygon_mut(&mut blended_canvas, &self.points, self.color);
        blended_canvas.0
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let target_image = open("colors.png")?.to_rgba8();
    let (w, h) = (target_image.width(), target_image.height());
    
    let t = Triangle::get_random(w, h);
    let mut cool_image = t.draw(&target_image);
    for _ in 1..100000{
        let t = Triangle::get_random(w, h);
        cool_image = t.draw(&cool_image);
    }
    cool_image.save("output.png")?;
    Ok(())
}
