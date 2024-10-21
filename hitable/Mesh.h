//#pragma once
//
//#include "Hitable.h"
//#include "Triangle.h"
//#include "HitableList.h"
//#include "ObjFile.h"
//
//class Mesh : public Hitable{
//public:
//  __device__ Mesh(){}
//  __device__ Mesh(int *d_points, int nPoints, int *d_faces, int nFaces, Material *mat);
//
//  __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const;
//  __device__ virtual bool bounding_box(float t0, float t1, AABB& box) const;
//
//  HitableList *triangles;
//};
//
//__device__ bool Mesh::hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const {
//
//}
//
//
//
