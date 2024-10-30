#pragma once

#include "hitable/Hitable.h"
#include "GlobalTriangle.h"

// code for Triangle primitive with Möller–Trumbore algorithm for ray-triangle intersection

class Triangle : public Hitable{
public:
  __device__ Triangle() {}
  __device__ Triangle(Vec3 v0, Vec3 v1, Vec3 v2, Material *mat, bool culling = false)
                      : mat(mat), culling(culling) {
    vertices[0] = v0;
    vertices[1] = v1;
    vertices[2] = v2;

    // init bbox;
    float minX = min(vertices[0][0], min(vertices[1][0], vertices[2][0]));
    float minY = min(vertices[0][1], min(vertices[1][1], vertices[2][1]));
    float minZ = min(vertices[0][2], min(vertices[1][2], vertices[2][2]));

    float maxX = max(vertices[0][0], max(vertices[1][0], vertices[2][0]));
    float maxY = max(vertices[0][1], max(vertices[1][1], vertices[2][1]));
    float maxZ = max(vertices[0][2], max(vertices[1][2], vertices[2][2]));

    bbox = AABB(Vec3(minX, minY, minZ), Vec3(maxX, maxY, maxZ));

//    printf("Newly created triangle has bbox with min: %f %f %f and max: %f %f %f\n", bbox._min.x(), bbox._min.y(), bbox._min.z(), bbox._max.x(), bbox._max.y(), bbox._max.z());

  }


  __device__ bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const override;
  __device__ AABB* get_bbox() override { return &bbox; }


  const float EPSILON = 0.00001f;
  Vec3 vertices[3];
  bool culling;
  Material *mat;

  AABB bbox;
  int triArrayIndex;
};

__device__ bool Triangle::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
  printf("Testing collision with triangle ArrayIndex %d\n", triArrayIndex);

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

  printf("Collision with triangle nodesArrayIndex %d\n", triArrayIndex);

  rec.t = t_hit;
  rec.p = r.point_at_parameter(t_hit);
  rec.normal = unit_vector(cross(e1, e2));
  rec.mat_ptr = mat;
  return true;

}

