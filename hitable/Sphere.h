#pragma once
#include <memory>
#include <utility>
#include "Hitable.h"
#include "Texture.h"

class Material;

class Sphere: public Hitable {
public:
  __device__ Sphere() {}
  __device__ Sphere(Vec3 cen, float r, Material *mat)
          : center(cen), radius(r), mat_ptr(mat) {
    bbox = AABB(center - Vec3(radius, radius, radius), center + Vec3(radius, radius, radius));
  }
  __device__ bool hit(const Ray&r, float tmin, float tmax, HitRecord& rec) const override;
//  __device__ virtual bool bounding_box(float t0, float t1, AABB& box) const;
  __device__ AABB* get_bbox() override { return &bbox; }

  Vec3 center;
  float radius;
  Material *mat_ptr;

  AABB bbox;
};

//__device__ bool Sphere::bounding_box(float t0, float t1, AABB& box) const {
//  box = AABB(center - Vec3(radius, radius, radius), center + Vec3(radius, radius, radius));
//  return true;
//}

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
      get_sphere_uv(rec.normal, rec.u, rec.v);
      return true;
    }
    temp = (-b + sqrtf(discriminant))/a;
    if(temp < t_max && temp > t_min){
      rec.t = temp;
      rec.p = r.point_at_parameter(rec.t);
      rec.normal = (rec.p - center) / radius;
      get_sphere_uv(rec.normal, rec.u, rec.v);
      return true;
    }
  }
  return false;
}

//
//class MovingSphere: public Hitable {
//public:
//  __device__ MovingSphere() {}
//  __device__ MovingSphere(Vec3 cen0, Vec3 cen1, float t0, float t1, float r, Material *m)
//          : center0(cen0), center1(cen1), time0(t0), time1(t1), radius(r), mat_ptr(m) {};
//  __device__ virtual bool hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const;
//  __device__ Vec3 center(float time) const;
//  Vec3 center0, center1;
//  float time0, time1;
//  float radius;
//  Material *mat_ptr;
//
//};
//
//__device__ Vec3 MovingSphere::center(float time) const {
//  return center0 + ((time - time0) / (time1 - time0)) * (center1 - center0);
//}
//
//
//__device__ bool MovingSphere::hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const {
//  Vec3 oc = r.origin() - center(r.time());
//  rec.mat_ptr = mat_ptr;
//  float a = dot(r.direction(), r.direction());
//  float b = dot(oc, r.direction());
//  float c = dot(oc, oc) - radius * radius;
//  float discriminant = b*b - a * c;
//  if(discriminant > 0){
//    float temp = (-b - sqrtf(discriminant))/a;
//    if(temp < t_max && temp > t_min){
//      rec.t = temp;
//      rec.p = r.point_at_parameter(rec.t);
//      rec.normal = (rec.p - center(r.time())) / radius;
//      get_sphere_uv(rec.normal, rec.u, rec.v);
//
//      return true;
//    }
//    temp = (-b + sqrtf(discriminant))/a;
//    if(temp < t_max && temp > t_min){
//      rec.t = temp;
//      rec.p = r.point_at_parameter(rec.t);
//      rec.normal = (rec.p - center(r.time())) / radius;
//      get_sphere_uv(rec.normal, rec.u, rec.v);
//      return true;
//    }
//  }
//  return false;
//}