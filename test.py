import os
import random
from typing import Optional, List, Iterator
from PIL import Image, ImageDraw
import numpy as np
from concurrent.futures import ThreadPoolExecutor


class BottomNHeap:
    def __init__(self):
        self.heap: List[Triangle] = []

    def insert(self, elem):
        self.heap.append(elem)
        self.heap.sort()

    def iter(self) -> Iterator:
        return iter(self.heap)

    def __len__(self):
        return len(self.heap)

    def is_empty(self):
        return len(self.heap) == 0


class Triangle:
    def __init__(self, points: List[tuple[int, int]], color: tuple[int, int, int, int], score: Optional[int] = None):
        self.points = points
        self.color = color
        self.score = score

    def __lt__(self, other):
        return (self.score or float('inf')) < (other.score or float('inf'))

    def __eq__(self, other):
        return self.score == other.score

    @staticmethod
    def get_random(max_x, max_y):
        points = [(random.randint(0, max_x), random.randint(0, max_y)) for _ in range(3)]
        color = tuple(random.randint(0, 255) for _ in range(4))
        return Triangle(points, color)

    def get_mutated(self, temperature, max_x, max_y):
        dx = int(temperature * max_x)
        dy = int(temperature * max_y)
        dc = int(temperature * 255)

        points = [
            (
                max(0, min(max_x, p[0] + random.randint(-dx, dx))),
                max(0, min(max_y, p[1] + random.randint(-dy, dy))),
            ) for p in self.points
        ]

        color = tuple(
            max(0, min(255, c + random.randint(-dc, dc)))
            for c in self.color
        )

        return Triangle(points, color, self.score)

    def draw(self, image: Image.Image) -> Image.Image:
        if len(set(self.points)) < 3:
            return image.copy()

        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.polygon(self.points, fill=self.color)
        return Image.alpha_composite(image, overlay)


def diff(img1: Image, img2: Image) -> int:
    a = np.asarray(img1, dtype=np.int16)
    b = np.asarray(img2, dtype=np.int16)
    diff = (a - b).astype(np.int32)  # ensure enough range
    squared = diff ** 2
    return int(np.sum(squared, dtype=np.uint64))  # prevent overflow


def main():
    target_image = Image.open("mona.jpg").convert("RGBA")
    w, h = target_image.size
    os.makedirs("images", exist_ok=True)

    population_size = int(input("Population size: "))
    num_iterations = int(input("Number of iterations per triangle: "))
    num_triangles = int(input("Number of triangles: "))
    temperature = float(input("Initial temperature (e.g., 0.1): "))
    emergency_iterations = int(input("Emergency iterations (e.g., 100): "))
    save_every_n = int(input("Save every Nth triangle: "))

    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 255))

    def evaluate_triangle(_) -> Triangle:
        tri = Triangle.get_random(w, h)
        tri.score = diff(target_image, tri.draw(canvas))
        return tri

    with ThreadPoolExecutor() as pool:
        initial_population = list(pool.map(evaluate_triangle, range(population_size)))

    heap = BottomNHeap()
    for tri in initial_population:
        heap.insert(tri)

    for triangle_index in range(num_triangles):
        print(f"\n--- Evolving Triangle #{triangle_index + 1} ---")
        score_before = diff(target_image, canvas)

        iter_ = 0
        best_score = float('inf')

        while True:
            num_survivors = population_size // 2
            num_mutants = population_size - num_survivors

            sorted_heap = sorted(heap.iter(), key=lambda t: t.score or float('inf'))
            survivors = sorted_heap[:num_survivors]

            def mutate_and_score(_) -> Triangle:
                parent = random.choice(survivors)
                child = parent.get_mutated(temperature, w, h)
                child.score = diff(target_image, child.draw(canvas))
                return child

            with ThreadPoolExecutor() as pool:
                mutants = list(pool.map(mutate_and_score, range(num_mutants)))

            heap = BottomNHeap()
            for t in survivors + mutants:
                heap.insert(t)

            current_best = min(heap.iter(), key=lambda t: t.score or float('inf')).score or float('inf')
            best_score = min(best_score, current_best)

            print(f"  Iter {iter_:03} — Best: {current_best}, Overall best: {best_score}, Temp: {temperature:.5f}")

            if iter_ >= num_iterations - 1 and best_score < score_before:
                print("  Stopping: Improvement achieved.")
                break
            if iter_ >= num_iterations - 1 + emergency_iterations:
                print("  Stopping: Emergency limit reached.")
                break

            iter_ += 1

        best = min(heap.iter(), key=lambda t: t.score or float('inf'))
        canvas = best.draw(canvas)

        if triangle_index % save_every_n == 0:
            filename = f"images/step_{triangle_index:04}.png"
            print(f"  Saving to {filename}")
            canvas.save(filename)

    canvas.save("images/final_result.png")
    print("Final image saved to images/final_result.png")


if __name__ == "__main__":
    main()
