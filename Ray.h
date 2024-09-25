#pragma once
#include "Vec3.h"

class Ray{

public:
  __device__ Ray() {}
  __device__ Ray(const Vec3& a, const Vec3& b) {A = a; B = b;}
  __device__ Vec3 origin() const       {return A;}
  __device__ Vec3 direction() const    {return B;}
  __device__ Vec3 point_at_parameter(float t) const {return A + t*B;}

  Vec3 A;
  Vec3 B;
//  Vec3 randomUnitVector(){
//    Vec3 p;
//    do {
//      p = 2.0 * Vec3(drand48(), drand48(), drand48()) - Vec3(1,1,1);
//    } while (p.length_squared() >= 1.0);
//    return p;
//  }

};
