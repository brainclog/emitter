#pragma once

#include <memory>
#include "Ray.h"
//#include "Material.h"

class Material;

struct HitRecord{
  float t;
  Vec3 p;
  Vec3 normal;
//  *Material mat_ptr;
};

class Hitable {
public:
  __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const = 0;
};