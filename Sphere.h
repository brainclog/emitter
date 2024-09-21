#pragma once

#include "Hitable.h"

class Sphere: public Hitable {
public:
  Sphere() {}
  Sphere(Vec3 cen, float r) : center(cen), radius(r) {};
  virtual bool hit(const Ray&r, float tmin, float tmax, HitRecord& rec) const;

  Vec3 center;
  float radius;
};

bool Sphere::hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const {
  Vec3 oc = r.origin() - center;
  double a = dot(r.direction(), r.direction());
  double b = dot(oc, r.direction());
  double c = dot(oc, oc) - radius * radius;
  double discriminant = b*b - a * c;
  if(discriminant > 0){
    float temp = (-b - std::sqrt(b*b - a*c))/a;
    if(temp < t_max && temp > t_min){
      rec.t = temp;
      rec.p = r.point_at_parameter(rec.t);
      rec.normal = (rec.p - center) / radius;
      return true;
    }
    temp = (-b + std::sqrt(b*b-a*c))/a;
    if(temp < t_max && temp > t_min){
      rec.t = temp;
      rec.p = r.point_at_parameter(rec.t);
      rec.normal = (rec.p - center) / radius;
      return true;
    }
  }
  return false;
}
