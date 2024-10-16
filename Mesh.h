#pragma once

#include "Hitable.h"
#include "Triangle.h"
#include "HitableList.h"

class Mesh : public Hitable{
public:
  __device__ Mesh(){}
  __device__ Mesh(int *d_points, int nPoints, int *d_faces, int nFaces, Material *mat);


};

