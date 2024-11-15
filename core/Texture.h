#pragma once
#include "math_constants.h"
#include <cstdio>
class Texture {
public:
  __device__ virtual Vec3 value(float u, float v, const Vec3& p) const = 0;
};

class ConstantTexture : public Texture {
public:
  __device__ ConstantTexture() {}
  __device__ ConstantTexture(Vec3 c) : color(c) {}
  __device__ virtual Vec3 value(float u, float v, const Vec3& p) const {
    return color;
  }
  Vec3 color;

};

class CheckerTexture : public Texture {
public:
  __device__ CheckerTexture() {}
  __device__ CheckerTexture(Texture* t0, Texture* t1) : even(t0), odd(t1) {}
  __device__ virtual Vec3 value(float u, float v, const Vec3& p) const {
    float sines = sin(10*p.x())*sin(10*p.y())*sin(10*p.z());
    if (sines < 0) {
      return odd->value(u, v, p);
    } else {
      return even->value(u, v, p);
    }
  }

  Texture *odd;
  Texture *even;

};

__device__ void get_sphere_uv(const Vec3& p, float& u, float& v) {
  auto theta = acos(-p.y());
  auto phi = atan2(-p.z(), p.x()) + CUDART_PI_F;

  u = phi / (2*CUDART_PI_F);
  v = theta / CUDART_PI_F;
}

class ImageTexture : public Texture {
public:
  __device__ ImageTexture() {}
  __device__ ImageTexture(cudaTextureObject_t tex) : tex(tex) {}
  __device__ virtual Vec3 value(float u, float v, const Vec3& p) const;

  cudaTextureObject_t tex;
  unsigned char *data;
  int nx, ny;
};

__device__ Vec3 ImageTexture::value(float u, float v, const Vec3 &p) const {
  float4 texel = tex2D<float4>(tex, u, 1.0f - v);
  return Vec3(texel.x, texel.y, texel.z);
}








//class Perlin {
//public:
//
//  __device__ Perlin(curandState *local_rand_state) {
//    ranfloat = perlin_generate(local_rand_state);
//    perm_x = perlin_generate_perm(local_rand_state);
//    perm_y = perlin_generate_perm(local_rand_state);
//    perm_z = perlin_generate_perm(local_rand_state);
//  }
//  __device__ float noise(const Vec3& p) const {
//    float u = p.x() - floor(p.x());
//    float v = p.y() - floor(p.y());
//    float w = p.z() - floor(p.z());
//    int i = int(4*p.x()) & 255;
//    int j = int(4*p.y()) & 255;
//    int k = int(4*p.z()) & 255;
//    return ranfloat[perm_x[i] ^ perm_y[j] ^ perm_z[k]];
//  }
//  static float *ranfloat;
//  static int *perm_x;
//  static int *perm_y;
//  static int *perm_z;
//
//
//  __device__ static float* perlin_generate(curandState *local_rand_state) {
//    float *p = new float[256];
//    for (int i = 0; i < 256; ++i) {
//      p[i] = curand_uniform(local_rand_state);
//    }
//    return p;
//  }
//
//  __device__ static void permute(int *p, int n, curandState *local_rand_state) {
//    for (int i = n-1; i > 0; i--) {
//      int target = int(curand_uniform(local_rand_state)*(i+1));
//      int tmp = p[i];
//      p[i] = p[target];
//      p[target] = tmp;
//    }
//  }
//
//  __device__ static int* perlin_generate_perm(curandState *local_rand_state) {
//    int *p = new int[256];
//    for (int i = 0; i < 256; ++i) {
//      p[i] = i;
//    }
//    permute(p, 256, local_rand_state);
//    return p;
//  }
//
//
//};
//
//
//
//class NoiseTexture : public Texture {
//public:
//  __device__ NoiseTexture(curandState *local_rand_state) : noise(local_rand_state) {}
//  __device__ virtual Vec3 value(float u, float v, const Vec3& p) const {
//    return Vec3(1,1,1)*noise.noise(p);
//  }
//  Perlin noise;
//
//};
