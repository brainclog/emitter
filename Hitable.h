#pragma once

#include <memory>
#include "Ray.h"
//#include "Material.h"

class Material;

struct HitRecord{
  float t;
  Vec3 p;
  Vec3 normal;
  std::shared_ptr<Material> mat_ptr;
};

class Hitable {
public:
  virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const = 0;
};