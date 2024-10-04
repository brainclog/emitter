#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Ray.h"

class AABB {
public:
  __device__ AABB() {}
  __device__ AABB(const Vec3& a, const Vec3& b) { _min = a; _max = b; }

  __device__ Vec3 min() const { return _min; }
  __device__ Vec3 max() const { return _max; }


  Vec3 _min;
  Vec3 _max;

  __device__ bool hit(const Ray &r, float tmin, float tmax) const;



};

__device__ AABB surrounding_box(const AABB& box0, const AABB& box1) {
  Vec3 small(fmin(box0.min().x(), box1.min().x()),
             fmin(box0.min().y(), box1.min().y()),
             fmin(box0.min().z(), box1.min().z()));
  Vec3 big(fmax(box0.max().x(), box1.max().x()),
           fmax(box0.max().y(), box1.max().y()),
           fmax(box0.max().z(), box1.max().z()));
  return {small, big};
}

__device__ inline bool AABB::hit(const Ray& r, float tmin, float tmax) const{
  for (int a = 0; a < 3; a++){
    float invD = 1.0f / r.direction()[a];
    float t0 = (min()[a] - r.origin()[a]) * invD;
    float t1 = (max()[a] - r.origin()[a]) * invD;
    if (invD < 0.0f){ // swap t0 and t1
      float temp = t0;
      t0 = t1;
      t1 = temp;
    }
    tmin = t0 > tmin ? t0 : tmin;
    tmax = t1 < tmax ? t1 : tmax;
    if (tmax <= tmin)
      return false;
  }
  return true;
}