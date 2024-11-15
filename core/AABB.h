#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Ray.h"
#include "util/Interval.h"

class AABB {
public:
  __device__ AABB() {}

  __device__ AABB(const Interval& x, const Interval& y, const Interval& z)
          : x(x), y(y), z(z) {
    _min = Vec3(x.min, y.min, z.min);
    _max = Vec3(x.max, y.max, z.max);
  }

  __device__ AABB(const Vec3& a, const Vec3& b) {
    _min = a; _max = b;

    x = (a[0] <= b[0]) ? Interval(a[0], b[0]) : Interval(b[0], a[0]);
    y = (a[1] <= b[1]) ? Interval(a[1], b[1]) : Interval(b[1], a[1]);
    z = (a[2] <= b[2]) ? Interval(a[2], b[2]) : Interval(b[2], a[2]);
  }

  __device__ AABB(const AABB& box0, const AABB& box1) { //union of two boxes
    x = Interval(box0.x, box1.x);
    y = Interval(box0.y, box1.y);
    z = Interval(box0.z, box1.z);
    _min = Vec3(x.min, y.min, z.min);
    _max = Vec3(x.max, y.max, z.max);
  }

  __device__ Vec3 min() const { return _min; }
  __device__ Vec3 max() const { return _max; }


  Vec3 _min;
  Vec3 _max;

  Interval x, y, z; 

  __device__ bool hit(const Ray &r, float tmin, float tmax) const;

  __device__ const Interval& axis_interval(int n) const {
    if (n == 1) return y;
    if (n == 2) return z;
    return x;
  }
  __device__ int longest_axis() const {

    if (x.size() > y.size())
      return x.size() > z.size() ? 0 : 2;
    else
      return y.size() > z.size() ? 1 : 2;
  }

  __device__ Vec3 centroid() {
    Vec3 center{(x.min + x.max) / 2, (y.min + y.max) / 2, (z.min + z.max) / 2};
    if(isfinite(center[0]) || isfinite(center[1]) || isfinite(center[2])){
      return {0,0,0};
    }
    return center;
  }

  __device__ void expand(float delta) {
    x.expand(delta);
    y.expand(delta);
    z.expand(delta);
    _min = Vec3(x.min, y.min, z.min);
    _max = Vec3(x.max, y.max, z.max);
  }

  __device__ static AABB empty() {
    return { Interval::empty(), Interval::empty(), Interval::empty()};
  }

  __device__ static AABB universe() {
    return { Interval::universe(), Interval::universe(), Interval::universe()};
  }
};




__device__ bool AABB::hit(const Ray& r, float tmin, float tmax) const{
  Interval ray_t(tmin, tmax);
  const Vec3& ray_orig = r.origin();
  const Vec3&   ray_dir  = r.direction();

  for (int axis = 0; axis < 3; axis++) {
    const Interval& ax = axis_interval(axis);
    const float adinv = 1.0f / ray_dir[axis];

    auto t0 = (ax.min - ray_orig[axis]) * adinv;
    auto t1 = (ax.max - ray_orig[axis]) * adinv;

    if (t0 < t1) {
      if (t0 > ray_t.min) ray_t.min = t0;
      if (t1 < ray_t.max) ray_t.max = t1;
    } else {
      if (t1 > ray_t.min) ray_t.min = t1;
      if (t0 < ray_t.max) ray_t.max = t0;
    }

    if (ray_t.max <= ray_t.min)
      return false;
  }
  return true;
}
