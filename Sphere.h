//#pragma once
//#include <memory>
//#include <utility>
//#include "Hitable.h"
//
//class Material;
//
//class Sphere: public Hitable {
//public:
//  Sphere() {}
//  Sphere(Vec3 cen, float r, std::shared_ptr<Material> mat)
//          : center(cen), radius(r), material(std::move(mat)) {}
//  virtual bool hit(const Ray&r, float tmin, float tmax, HitRecord& rec) const;
//
//  Vec3 center;
//  float radius;
//  std::shared_ptr<Material> material;
//};
//
//bool Sphere::hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const {
//  rec.mat_ptr = material;
//  Vec3 oc = r.origin() - center;
//  double a = dot(r.direction(), r.direction());
//  double b = dot(oc, r.direction());
//  double c = dot(oc, oc) - radius * radius;
//  double discriminant = b*b - a * c;
//  if(discriminant > 0){
//    float temp = (-b - std::sqrt(b*b - a*c))/a;
//    if(temp < t_max && temp > t_min){
//      rec.t = temp;
//      rec.p = r.point_at_parameter(rec.t);
//      rec.normal = (rec.p - center) / radius;
//      return true;
//    }
//    temp = (-b + std::sqrt(b*b-a*c))/a;
//    if(temp < t_max && temp > t_min){
//      rec.t = temp;
//      rec.p = r.point_at_parameter(rec.t);
//      rec.normal = (rec.p - center) / radius;
//      return true;
//    }
//  }
//  return false;
//}
