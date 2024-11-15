#include "Ray.h"

__device__ Vec3 random_in_unit_disk(curandState *local_rand_state){

  Vec3 p;
  do{
    p = 2.0*Vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - Vec3(1,1,0);
  } while (dot(p,p) >=1.0);
  return p;
}

class Camera {
public:
  __device__ Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, float vfov, float aspect, float aperture, float focus_dist, float t0, float t1) {
    lens_radius = aperture/2.0f;
    float theta = vfov*(float)M_PI/180.0f;
    float half_height = tan(theta/2.0f);
    float half_width = aspect * half_height;
    origin = lookfrom;
    w = unit_vector(lookfrom - lookat);
    u = unit_vector(cross(vup, w));
    v = cross(w, u);
    lower_left_corner = origin - half_width*focus_dist*u - half_height*focus_dist*v - focus_dist*w;
    horizontal = 2*half_width*focus_dist*u;
    vertical = 2*half_height*focus_dist*v;
  }

  __device__ Ray getRay(float s, float t, curandState *local_rand_state) {
    Vec3 rd = lens_radius*random_in_unit_disk(local_rand_state);
    Vec3 offset = u * rd.x() + v*rd.y();
    float time = time0 + curand_uniform(local_rand_state)*(time1-time0);
    return {origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset};
  }

  Vec3 lower_left_corner;
  Vec3 horizontal;
  Vec3 vertical;
  Vec3 origin;
  Vec3 u,v,w;
  float lens_radius;
  float time0, time1;

};