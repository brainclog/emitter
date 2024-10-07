#include "Hitable.h"
#include "Texture.h"

__device__ Vec3 randomUnitVector(curandState *local_rand_state){
  Vec3 p;
  do {
    p = 2.0f * Vec3(curand_uniform(local_rand_state),
                    curand_uniform(local_rand_state),
                    curand_uniform(local_rand_state))
        - Vec3(1.0,1.0,1.0);
  } while (p.length_squared() >= 1.0);
  return p;
}

__device__ Vec3 reflect(const Vec3& v, const Vec3& n) {
  return v - 2.0f*dot(v,n)*n;
}

class Material  {
public:
  __device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState *local_rand_state) const = 0;
};

class lambertian : public Material {
public:
  __device__ explicit lambertian(Texture *a) : albedo(a) {}
  __device__ bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState *local_rand_state)
  const override {
    Vec3 target = rec.p + rec.normal + randomUnitVector(local_rand_state);
    scattered = Ray(rec.p, target-rec.p, r_in.time());
    attenuation = albedo->value(0,0,rec.p);
    return true;
  }

private:
  Texture *albedo;
};


class metal : public Material {
public:
  __device__ explicit metal(Texture *a, double f) : albedo(a) {if (f < 1) fuzz = f; else fuzz = 1; }

  __device__ bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState *local_rand_state)
  const override {
    Vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    scattered = Ray(rec.p, reflected + fuzz*randomUnitVector(local_rand_state), r_in.time());
    attenuation = albedo->value(0,0,rec.p);
    return (dot(scattered.direction(), rec.normal) > 0.0f);
  }

private:
  Texture *albedo;
  float fuzz;
};


__device__ bool refract(const Vec3& v, const Vec3& n, float ni_over_nt, Vec3& refracted){
  Vec3 uv = unit_vector(v);
  float dt = dot(uv, n);
  float discriminant = 1.0f - ni_over_nt*ni_over_nt*(1-dt*dt);
  if (discriminant > 0.0f){
    refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
    return true;
  }
  else
    return false;
}

__device__ float schlick(float cosine, float ref_idx){
  float r0 = (1.0f-ref_idx) / (1.0f+ref_idx);
  r0 = r0*r0;
  return r0 + (1.0f-r0) * pow((1.0f - cosine), 5.0f);
}

class dielectric : public Material {
public:
  __device__ dielectric (float ri) : ref_idx(ri) {}
  __device__ bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState *local_rand_state)
  const override {
    Vec3 outward_normal;
    Vec3 reflected = reflect(r_in.direction(), rec.normal);
    float ni_over_nt;
    attenuation = Vec3(1.0f,1.0f,1.0f);
    Vec3 refracted;
    float reflect_prob;
    float cosine;
    if(dot(r_in.direction(), rec.normal) > 0) {
      outward_normal = -rec.normal;
      ni_over_nt = ref_idx;
      cosine = ref_idx * dot(r_in.direction(), rec.normal) / r_in.direction().length();
    }
    else {
      outward_normal = rec.normal;
      ni_over_nt = 1.0f / ref_idx;
      cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
    }
    if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted)){
      reflect_prob = schlick(cosine, ref_idx);
    }
    else {
      scattered = Ray(rec.p, reflected, r_in.time());
      reflect_prob = 1.0f;
    }
    if(curand_uniform(local_rand_state) < reflect_prob){
      scattered = Ray(rec.p, reflected, r_in.time());
    }
    else {
      scattered = Ray(rec.p, refracted, r_in.time());
    }
    return true;
  }

  float ref_idx;

};

