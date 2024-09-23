#include "Ray.h"

class Camera {
public:
  Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, float vfov, float aspect) {
    Vec3 u,v,w;

    float theta = vfov*M_PI/180;
    float half_height = tan(theta/2);
    float half_width = aspect * half_height;
    origin = lookfrom;
    w = unit_vector(lookfrom - lookat);
    u = unit_vector(cross(vup, w));
    v = cross(w, u);


//    lower_left_corner = Vec3(-half_width, -half_height, -1.0);
    lower_left_corner = origin - half_width*u - half_height*v - w;
    horizontal = 2*half_width*u;
    vertical = 2*half_height*v;
  }

  Ray getRay(float s, float t) {return {origin, lower_left_corner + s * horizontal + t * vertical - origin};}

  Vec3 lower_left_corner;
  Vec3 horizontal;
  Vec3 vertical;
  Vec3 origin;

};