#pragma once

#include "Hitable.h"

// code for Triangle primitive with Möller–Trumbore algorithm for ray-triangle intersection

class Triangle : public Hitable{
public:
  __device__ Triangle() {}
  __device__ Triangle(Vec3 v0, Vec3 v1, Vec3 v2, Material *mat, bool culling = false)
                      : mat(mat), culling(culling) {
    vertices[0] = v0;
    vertices[1] = v1;
    vertices[2] = v2;
  }
  __device__ virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const;
  __device__ virtual bool bounding_box(float t0, float t1, AABB &box) const;

  const float EPSILON = 0.00001f;
  Vec3 vertices[3];
  bool culling;
  Material *mat;
};

__device__ bool Triangle::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {

  Vec3 e1 = vertices[1] - vertices[0];
  Vec3 e2 = vertices[2] - vertices[0];
  Vec3 p = cross(r.direction(), e2);
  float det = dot(e1, p);

  if (culling && det < EPSILON) return false;
  if (!culling && fabs(det) < EPSILON) return false;

  float inv_det = 1.0f / det;
  Vec3 t = r.origin() - vertices[0];
  float u = dot(t, p) * inv_det;
  if (u < 0.0f || u > 1.0f) return false;

  Vec3 q = cross(t, e1);
  float v = dot(r.direction(), q) * inv_det;
  if (v < 0.0f || u + v > 1.0f) return false;

  float t_hit = dot(e2, q) * inv_det;
  if (t_hit < t_min || t_hit > t_max) return false;

  rec.t = t_hit;
  rec.p = r.point_at_parameter(t_hit);
  rec.normal = unit_vector(cross(e1, e2));
  rec.mat_ptr = mat;
  return true;

}

__device__ bool Triangle::bounding_box(float t0, float t1, AABB& box) const {
  // do not build bounding box for triangle primitive, leaf node
  return true;
}