#pragma once
#include <memory>
#include <utility>
#include "Hitable.h"

class Material;

class Sphere: public Hitable {
public:
  __device__ Sphere() {}
  __device__ Sphere(Vec3 cen, float r, Material *mat)
          : center(cen), radius(r), mat_ptr(mat) {}
  __device__ virtual bool hit(const Ray&r, float tmin, float tmax, HitRecord& rec) const;

  Vec3 center;
  float radius;
  Material *mat_ptr;
};

__device__ bool Sphere::hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const {
  Vec3 oc = r.origin() - center;
  rec.mat_ptr = mat_ptr;
  float a = dot(r.direction(), r.direction());
  float b = dot(oc, r.direction());
  float c = dot(oc, oc) - radius * radius;
  float discriminant = b*b - a * c;
  if(discriminant > 0){
    float temp = (-b - sqrtf(discriminant))/a;
    if(temp < t_max && temp > t_min){
      rec.t = temp;
      rec.p = r.point_at_parameter(rec.t);
      rec.normal = (rec.p - center) / radius;
      return true;
    }
    temp = (-b + sqrtf(discriminant))/a;
    if(temp < t_max && temp > t_min){
      rec.t = temp;
      rec.p = r.point_at_parameter(rec.t);
      rec.normal = (rec.p - center) / radius;
      return true;
    }
  }
  return false;
}
