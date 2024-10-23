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
//  __device__ virtual bool bounding_box(float t0, float t1, AABB& box) const = 0;
  __device__ virtual AABB* get_bbox() = 0;
};

__device__ inline float ffmin (float a, float b) {return a < b ? a : b;}
__device__ inline float ffmax (float a, float b) {return a > b ? a : b;}


// hitable utility functions
class flip_normals : public Hitable {
public:
  __device__ flip_normals(Hitable *p) : ptr(p) {}
  __device__ virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const {
    if (ptr->hit(r, t_min, t_max, rec)) {
      rec.normal = -rec.normal;
      return true;
    }
    return false;
  }
  __device__ virtual AABB* get_bbox() { return ptr->get_bbox(); }

  Hitable *ptr;
};

class translate : public Hitable {
public:
  __device__ translate(Hitable *p, const Vec3 &displacement) : ptr(p), offset(displacement) {}
  __device__ virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const;
//  __device__ virtual bool bounding_box(float t0, float t1, AABB &box) const;
  __device__ virtual AABB* get_bbox() { return ptr->get_bbox(); }

  Hitable *ptr;
  Vec3 offset;
};

__device__ bool translate::hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const {
  Ray moved_r(r.origin() - offset, r.direction(), r.time());
  if (ptr->hit(moved_r, t_min, t_max, rec)) {
    rec.p += offset;
    return true;
  }
  return false;
}


class rotate_y : public Hitable {
public:
  __device__ rotate_y(Hitable *p, float angle);
  __device__ virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const;
//  __device__ virtual bool bounding_box(float t0, float t1, AABB &box) const {
//    box = bbox;
//    return hasbox;
//  }
  __device__ virtual AABB* get_bbox() { return ptr->get_bbox(); }

  Hitable *ptr;
  float sin_theta;
  float cos_theta;
//  bool hasbox;
  AABB bbox;
};

__device__ rotate_y::rotate_y(Hitable *p, float angle) : ptr(p) {
  float radians = (M_PI / 180.0f) * angle;
  sin_theta = sin(radians);
  cos_theta = cos(radians);
//  hasbox = ptr->bounding_box(0, 1, bbox);
  Vec3 min(FLT_MAX, FLT_MAX, FLT_MAX);
  Vec3 max(-FLT_MAX, -FLT_MAX, -FLT_MAX);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        float x = i*bbox.max().x() + (1-i)*bbox.min().x();
        float y = j*bbox.max().y() + (1-j)*bbox.min().y();
        float z = k*bbox.max().z() + (1-k)*bbox.min().z();
        float newx = cos_theta*x + sin_theta*z;
        float newz = -sin_theta*x + cos_theta*z;
        Vec3 tester(newx, y, newz);
        for (int c = 0; c < 3; c++) {
          if (tester[c] > max[c]) max[c] = tester[c];
          if (tester[c] < min[c]) min[c] = tester[c];
        }
      }
    }
  }
  bbox = AABB(min, max);
}

__device__ bool rotate_y::hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const {
  Vec3 origin = r.origin();
  Vec3 direction = r.direction();
  origin[0] = cos_theta*r.origin()[0] - sin_theta*r.origin()[2];
  origin[2] = sin_theta*r.origin()[0] + cos_theta*r.origin()[2];
  direction[0] = cos_theta*r.direction()[0] - sin_theta*r.direction()[2];
  direction[2] = sin_theta*r.direction()[0] + cos_theta*r.direction()[2];
  Ray rotated_r(origin, direction, r.time());
  if (ptr->hit(rotated_r, t_min, t_max, rec)) {
    Vec3 p = rec.p;
    Vec3 normal = rec.normal;
    p[0] = cos_theta*rec.p[0] + sin_theta*rec.p[2];
    p[2] = -sin_theta*rec.p[0] + cos_theta*rec.p[2];
    normal[0] = cos_theta*rec.normal[0] + sin_theta*rec.normal[2];
    normal[2] = -sin_theta*rec.normal[0] + cos_theta*rec.normal[2];
    rec.p = p;
    rec.normal = normal;
    return true;
  }
  else return false;
}