#pragma once

#include <memory>
#include "Ray.h"
#include "AABB.h"
//#include "Material.h"

class Material;

struct HitRecord{
  float t;
  Vec3 p;
  Vec3 normal;
  Material *mat_ptr;
  float u;
  float v;

};



class Hitable {
public:
  __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const = 0;
  __device__ virtual bool bounding_box(float t0, float t1, AABB& box) const = 0;
};



__device__ inline float ffmin (float a, float b) {return a < b ? a : b;}
__device__ inline float ffmax (float a, float b) {return a > b ? a : b;}



