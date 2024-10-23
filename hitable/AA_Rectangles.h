#pragma once
#include "hitable/Hitable.h"

class XY_Rectangle : public Hitable {
public:
  __device__ XY_Rectangle() {}
  __device__ XY_Rectangle(float _x0, float _x1, float _y0, float _y1, float _k, Material *mat)
      : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mp(mat) {
    bbox = AABB(Vec3(x0, y0, k - 0.0001f), Vec3(x1, y1, k + 0.0001f));
  }
  __device__ virtual bool hit(const Ray &r, float t0, float t1, HitRecord &rec) const;
  __device__ AABB* get_bbox() override { return &bbox; }

//  __device__ virtual bool bounding_box(float t0, float t1, AABB &box) const {
//    box = AABB(Vec3(x0, y0, k - 0.0001f), Vec3(x1, y1, k + 0.0001f));
//    return true;
//  }

  Material *mp;
  float x0, x1, y0, y1, k;
  AABB bbox;
};

__device__ bool XY_Rectangle::hit(const Ray &r, float t0, float t1, HitRecord &rec) const {
  float t = (k - r.origin().z()) / r.direction().z();
  if (t < t0 || t > t1) {
    return false;
  }
  float x = r.origin().x() + t * r.direction().x();
  float y = r.origin().y() + t * r.direction().y();
  if (x < x0 || x > x1 || y < y0 || y > y1) {
    return false;
  }
  rec.u = (x - x0) / (x1 - x0);
  rec.v = (y - y0) / (y1 - y0);
  rec.t = t;
  rec.mat_ptr = mp;
  rec.p = r.point_at_parameter(t);
  rec.normal = Vec3(0, 0, 1);
  return true;
}

class XZ_Rectangle : public Hitable {
public:
  __device__ XZ_Rectangle() {}
  __device__ XZ_Rectangle(float _x0, float _x1, float _z0, float _z1, float _k, Material *mat)
      : x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mp(mat) {
    bbox = AABB(Vec3(x0, k - 0.0001f, z0), Vec3(x1, k + 0.0001f, z1));
  }
  __device__ virtual bool hit(const Ray &r, float t0, float t1, HitRecord &rec) const;
  __device__ AABB* get_bbox() override { return &bbox; }

//  __device__ virtual bool bounding_box(float t0, float t1, AABB &box) const {
//    box = AABB(Vec3(x0, k - 0.0001f, z0), Vec3(x1, k + 0.0001f, z1));
//    return true;
//  }

  Material *mp;
  float x0, x1, z0, z1, k;
  AABB bbox;
};

__device__ bool XZ_Rectangle::hit(const Ray &r, float t0, float t1, HitRecord &rec) const {
  float t = (k - r.origin().y()) / r.direction().y();
  if (t < t0 || t > t1) {
    return false;
  }
  float x = r.origin().x() + t * r.direction().x();
  float z = r.origin().z() + t * r.direction().z();
  if (x < x0 || x > x1 || z < z0 || z > z1) {
    return false;
  }
  rec.u = (x - x0) / (x1 - x0);
  rec.v = (z - z0) / (z1 - z0);
  rec.t = t;
  rec.mat_ptr = mp;
  rec.p = r.point_at_parameter(t);
  rec.normal = Vec3(0, 1, 0);
  return true;
}

class YZ_Rectangle : public Hitable {
public:
  __device__ YZ_Rectangle() {}
  __device__ YZ_Rectangle(float _y0, float _y1, float _z0, float _z1, float _k, Material *mat)
      : y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mp(mat) {
    bbox = AABB(Vec3(k - 0.0001f, y0, z0), Vec3(k + 0.0001f, y1, z1));
  }
  __device__ virtual bool hit(const Ray &r, float t0, float t1, HitRecord &rec) const;
  __device__ AABB* get_bbox() override { return &bbox; }

  Material *mp;
  float y0, y1, z0, z1, k;
  AABB bbox;
};

__device__ bool YZ_Rectangle::hit(const Ray &r, float t0, float t1, HitRecord &rec) const {
  float t = (k - r.origin().x()) / r.direction().x();
  if (t < t0 || t > t1) {
    return false;
  }
  float y = r.origin().y() + t * r.direction().y();
  float z = r.origin().z() + t * r.direction().z();
  if (y < y0 || y > y1 || z < z0 || z > z1) {
    return false;
  }
  rec.u = (y - y0) / (y1 - y0);
  rec.v = (z - z0) / (z1 - z0);
  rec.t = t;
  rec.mat_ptr = mp;
  rec.p = r.point_at_parameter(t);
  rec.normal = Vec3(1, 0, 0);
  return true;
}

