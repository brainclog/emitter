#include <iostream>
#include <random>

#include "Vec3.h"
#include "Ray.h"
#include "Hitable.h"
#include "Sphere.h"
#include "HitableList.h"
#include "Image.h"
#include "Camera.h"
#include "Material.h"


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



Vec3 color(const Ray& r, Hitable* world, int depth) {
  HitRecord rec;
  if (world->hit(r, 0.001, MAXFLOAT, rec)) {
    Ray scattered;
    Vec3 attenuation;
    if(depth<50 && rec.mat_ptr->scatter(r, rec, attenuation, scattered)){
      return attenuation * color(scattered, world, depth+1);
    } else{
      return Vec3(0,0,0);
    }
  } else {
    Vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
  }
};


int main() {
  const int nx = 800;
  const int ny = 400;
  const int ns = 100;

  const int object_N = 5;
  Hitable* list[object_N];
  list[0] = new Sphere(Vec3(0, 0, -1), 0.5, std::make_shared<lambertian>(Vec3(0.8, 0.2, 0.3)));
  list[1] = new Sphere(Vec3(0, -100.5, -1), 100, std::make_shared<lambertian>(Vec3(0.8, 0.8, 0.1)));
  list[2] = new Sphere(Vec3(1, 0, -1), 0.5, std::make_shared<metal>(Vec3(0.8, 0.6, 0.2), 0.0));
  list[3] = new Sphere(Vec3(-1, 0, -1), 0.5, std::make_shared<dielectric>(1.5));
  list[4] = new Sphere(Vec3(-1, 0, -1), -0.45, std::make_shared<dielectric>(1.5));
  Hitable* world = new HitableList(list, object_N);
  Image image(nx, ny);
  auto rng = RandomGenerator(1);

  Vec3 lookfrom(3,3,2);
  Vec3 lookat(0,0,-1);
  float dist_to_focus = (lookfrom-lookat).length();
  float aperture = 2.0;
  Camera camera = Camera(lookfrom, lookat, Vec3(0,1,0) ,20, float(nx)/float(ny), aperture, dist_to_focus);


  // Generate the image
  for (int j = ny - 1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {
      Vec3 px_color = Vec3(0,0,0);

      for(int s = 0; s < ns; s++){
        float u = float(i + rng.getDouble()) / float(nx);
        float v = float(j + rng.getDouble()) / float(ny);
        //      Ray camera_ray(origin, lower_left_corner + u * horizontal + v * vertical);
        Ray camera_ray = camera.getRay(u,v);
        px_color += color(camera_ray, world, 0);
      }
      px_color /= float(ns);

      // gamma 1.8
      px_color = Vec3(std::pow(px_color.e[0], 1/1.8), std::pow(px_color.e[1], 1/1.8), std::pow(px_color.e[2], 1/1.8));

      // write pixel color
      image.write_pixel(i, ny - 1 - j, px_color);
    }
  }

  image.save("../output.png");

  return 0;
}
