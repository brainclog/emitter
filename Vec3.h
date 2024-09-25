#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>

//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include <cmath>
#include <cstdlib>
#include <iostream>

class Vec3  {


public:
  __host__ __device__ Vec3() {}
  __host__ __device__ Vec3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }
  __host__ __device__ inline float x() const { return e[0]; }
  __host__ __device__ inline float y() const { return e[1]; }
  __host__ __device__ inline float z() const { return e[2]; }
  __host__ __device__ inline float r() const { return e[0]; }
  __host__ __device__ inline float g() const { return e[1]; }
  __host__ __device__ inline float b() const { return e[2]; }

  __host__ __device__ inline const Vec3& operator+() const { return *this; }
  __host__ __device__ inline Vec3 operator-() const { return Vec3(-e[0], -e[1], -e[2]); }
  __host__ __device__ inline float operator[](int i) const { return e[i]; }
  __host__ __device__ inline float& operator[](int i) { return e[i]; };

  __host__ __device__ inline Vec3& operator+=(const Vec3 &v2);
  __host__ __device__ inline Vec3& operator-=(const Vec3 &v2);
  __host__ __device__ inline Vec3& operator*=(const Vec3 &v2);
  __host__ __device__ inline Vec3& operator/=(const Vec3 &v2);
  __host__ __device__ inline Vec3& operator*=(const float t);
  __host__ __device__ inline Vec3& operator/=(const float t);

  __host__ __device__ inline float length() const { return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]); }
  __host__ __device__ inline float length_squared() const { return e[0]*e[0] + e[1]*e[1] + e[2]*e[2]; }
  __host__ __device__ inline void make_unit_vector();
//  __host__ __device__ inline Vec3 unit_vector(Vec3 v);
;

  float e[3];
};



inline std::istream& operator>>(std::istream &is, Vec3 &t) {
  is >> t.e[0] >> t.e[1] >> t.e[2];
  return is;
}

inline std::ostream& operator<<(std::ostream &os, const Vec3 &t) {
  os << t.e[0] << " " << t.e[1] << " " << t.e[2];
  return os;
}

__host__ __device__ inline void Vec3::make_unit_vector() {
  float k = 1.0 / sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
  e[0] *= k; e[1] *= k; e[2] *= k;
}

__host__ __device__ inline Vec3 operator+(const Vec3 &v1, const Vec3 &v2) {
  return Vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline Vec3 operator-(const Vec3 &v1, const Vec3 &v2) {
  return Vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline Vec3 operator*(const Vec3 &v1, const Vec3 &v2) {
  return Vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline Vec3 operator/(const Vec3 &v1, const Vec3 &v2) {
  return Vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline Vec3 operator*(float t, const Vec3 &v) {
  return Vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline Vec3 operator/(Vec3 v, float t) {
  return Vec3(v.e[0]/t, v.e[1]/t, v.e[2]/t);
}

__host__ __device__ inline Vec3 operator*(const Vec3 &v, float t) {
  return Vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline float dot(const Vec3 &v1, const Vec3 &v2) {
  return v1.e[0] *v2.e[0] + v1.e[1] *v2.e[1]  + v1.e[2] *v2.e[2];
}

__host__ __device__ inline Vec3 cross(const Vec3 &v1, const Vec3 &v2) {
  return Vec3( (v1.e[1]*v2.e[2] - v1.e[2]*v2.e[1]),
               (-(v1.e[0]*v2.e[2] - v1.e[2]*v2.e[0])),
               (v1.e[0]*v2.e[1] - v1.e[1]*v2.e[0]));
}


__host__ __device__ inline Vec3& Vec3::operator+=(const Vec3 &v){
  e[0]  += v.e[0];
  e[1]  += v.e[1];
  e[2]  += v.e[2];
  return *this;
}

__host__ __device__ inline Vec3& Vec3::operator*=(const Vec3 &v){
  e[0]  *= v.e[0];
  e[1]  *= v.e[1];
  e[2]  *= v.e[2];
  return *this;
}

__host__ __device__ inline Vec3& Vec3::operator/=(const Vec3 &v){
  e[0]  /= v.e[0];
  e[1]  /= v.e[1];
  e[2]  /= v.e[2];
  return *this;
}

__host__ __device__ inline Vec3& Vec3::operator-=(const Vec3& v) {
  e[0]  -= v.e[0];
  e[1]  -= v.e[1];
  e[2]  -= v.e[2];
  return *this;
}

__host__ __device__ inline Vec3& Vec3::operator*=(const float t) {
  e[0]  *= t;
  e[1]  *= t;
  e[2]  *= t;
  return *this;
}

__host__ __device__ inline Vec3& Vec3::operator/=(const float t) {
  float k = 1.0/t;

  e[0]  *= k;
  e[1]  *= k;
  e[2]  *= k;
  return *this;
}

__host__ __device__ inline Vec3 unit_vector(Vec3 v) {
  return v / v.length();
}

__host__ __device__ inline Vec3 randomUnitVector(){
  Vec3 p;
  do {
    p = 2.0 * Vec3(drand48(), drand48(), drand48()) - Vec3(1,1,1);
  } while (p.length_squared() >= 1.0);
  return p;
}

inline Vec3 reflect(const Vec3& v, const Vec3& n) {
  return v - 2*dot(v,n)*n;
}
