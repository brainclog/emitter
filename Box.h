#pragma once
#include "hitable/Hitable.h"
#include "AA_Rectangles.h"

class Box : public Hitable {
public:
  __device__ Box() {}
  __device__ Box(const Vec3& p0, const Vec3& p1, Material* mat);
  __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const;
  __device__ virtual bool bounding_box(float t0, float t1, AABB& box) const {
    box = AABB(pmin, pmax);
    return true;
  }

  Vec3 pmin, pmax;
  Hitable *list_ptr;
};

__device__ Box::Box(const Vec3 &p0, const Vec3 &p1, Material *mat) {
  pmin = p0;
  pmax = p1;
  Hitable **list = new Hitable*[6];
  list[0] = new XY_Rectangle(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), mat);
  list[1] = new flip_normals(new XY_Rectangle(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), mat));
  list[2] = new XZ_Rectangle(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), mat);
  list[3] = new flip_normals(new XZ_Rectangle(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), mat));
  list[4] = new YZ_Rectangle(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), mat);
  list[5] = new flip_normals(new YZ_Rectangle(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), mat));
  list_ptr = new HitableList(list, 6);

}


__device__ bool Box::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
  return list_ptr->hit(r, t_min, t_max, rec);
}

