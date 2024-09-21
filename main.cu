#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "Vec3.h"
#include "Ray.h"
#include "Hitable.h"
#include "Sphere.h"
#include "HitableList.h"

Vec3 color(const Ray& r, Hitable *world){
  HitRecord rec;
  if(world->hit(r, 0.0, MAXFLOAT, rec)){
    return 0.5*Vec3(rec.normal.x()+1, rec.normal.y()+1, rec.normal.z()+1);
  }
  else {
    Vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0-t) * Vec3(1.0,1.0,1.0) + t * Vec3(0.5,0.7,1.0);
  }

}

int main() {
  int nx = 200;
  int ny = 100;

  Vec3 lower_left_corner(-2.0, -1.0, -1.0);
  Vec3 horizontal (4.0, 0.0, 0.0);
  Vec3 vertical (0.0, 2.0, 0.0);
  Vec3 origin(0.0, 0.0, 0.0);

  Hitable *list[2];
  list[0] = new Sphere(Vec3(0,0,-1), 0.5);
  list[1] = new Sphere(Vec3(0,-100.5,-1), 100);
  Hitable *world = new HitableList(list, 2);

    // Allocate memory for the pixel data
    unsigned char *image = new unsigned char[nx * ny * 3]; // 3 channels for RGB
    for (int j = ny-1; j >=0; j--) {
        for (int i = 0; i < nx; i++) {
            // these will interpolate with the loop, u interpolates from left to right, v interpolates bottom to up.
            float u = float(i) / float (nx);
            double v = float(j) / float(ny);

            // declare camera ray that points from origin towards point on the *screen* (background)
            // right now it sweeps across the plane formed by the horizontal-vertical plane
            Ray camera_ray(origin, lower_left_corner + u*horizontal + v*vertical);
            Vec3 p = camera_ray.point_at_parameter(2.0);
            Vec3 px_color = color(camera_ray, world);

            // Scale to 0-255 and store in the image array
            image[(j * nx + i) * 3 + 0] = static_cast<unsigned char>(255.99f * px_color.x()); // Red
            image[(j * nx + i) * 3 + 1] = static_cast<unsigned char>(255.99f * px_color.y()); // Green
            image[(j * nx + i) * 3 + 2] = static_cast<unsigned char>(255.99f * px_color.z()); // Blue
        }
    }

    // Write the image to a PNG file
    stbi_write_png("../output.png", nx, ny, 3, image, nx * 3); // Stride is nx * 3
    delete[] image; // Free the allocated memory

    return 0;
}

