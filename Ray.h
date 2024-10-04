#pragma once
#include "Vec3.h"

class Ray{

public:
  __device__ Ray() {}
  __device__ Ray(const Vec3& a, const Vec3& b, float ti = 0.0f) {A = a; B = b; _time = ti;}
  __device__ Vec3 origin() const       {return A;}
  __device__ Vec3 direction() const    {return B;}
  __device__ float time() const              {return _time;}
  __device__ Vec3 point_at_parameter(float t) const {return A + t*B;}

  Vec3 A;
  Vec3 B;
  float _time;



};
