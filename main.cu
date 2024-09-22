#include <iostream>
#include <random>

#include "Vec3.h"
#include "Ray.h"
#include "Hitable.h"
#include "Sphere.h"
#include "HitableList.h"
#include "Image.h"
#include "Camera.h"

Vec3 color(const Ray& r, Hitable* world) {
  HitRecord rec;
  if (world->hit(r, 0.0, MAXFLOAT, rec)) {
    return 0.5 * Vec3(rec.normal.x() + 1, rec.normal.y() + 1, rec.normal.z() + 1);
  } else {
    Vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
  }
}

class RandomGenerator {
private:
  std::default_random_engine engine;
  std::uniform_real_distribution<double> distribution; // Range [0, 1)
public:
  // no seed
  RandomGenerator() : engine(std::random_device{}()), distribution(0.0, 1.0) {}
  // with seed
  RandomGenerator(unsigned int seed) : engine(seed), distribution(0.0, 1.0) {}

  double getDouble() {
    return distribution(engine);
  }
};

int main() {
  int nx = 200;
  int ny = 100;


  Hitable* list[2];
  list[0] = new Sphere(Vec3(0, 0, -1), 0.5);
  list[1] = new Sphere(Vec3(0, -100.5, -1), 100);
  Hitable* world = new HitableList(list, 2);
  Camera camera;

  Image image(nx, ny);

  // Generate the image
  for (int j = ny - 1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {
      float u = float(i) / float(nx);
      float v = float(j) / float(ny);


//      Ray camera_ray(origin, lower_left_corner + u * horizontal + v * vertical);
      Ray camera_ray = camera.getRay(u,v);
      Vec3 px_color = color(camera_ray, world);

      // Write the pixel color to the image
      image.write_pixel(i, ny - 1 - j, px_color);
    }
  }

  image.save("../output.png");

  return 0;
}
