//#include "Hitable.h"
//class Material {
//public:
//  virtual ~Material() = default;
//
//  virtual bool scatter(
//          const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered
//  ) const {
//    return false;
//  }
//
//
//
//};
//class lambertian : public Material {
//public:
//  explicit lambertian(const Vec3& albedo) : albedo(albedo) {}
//  bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered)
//  const override {
//    Vec3 target = rec.p + rec.normal + randomUnitVector();
//    scattered = Ray(rec.p, target-rec.p);
//    attenuation = albedo;
//    return true;
//  }
//
//private:
//  Vec3 albedo;
//};
//
//
//class metal : public Material {
//public:
//  explicit metal(const Vec3& albedo, double f) : albedo(albedo) {if (f < 1) fuzz = f; else fuzz = 1; }
//
//  bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered)
//  const override {
//    Vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
//    scattered = Ray(rec.p, reflected + fuzz*randomUnitVector());
//    attenuation = albedo;
//    return (dot(scattered.direction(), rec.normal) > 0);
//  }
//
//private:
//  Vec3 albedo;
//  double fuzz;
////  double fuzz;
//};
//
////inline Vec3 refract(const Vec3& uv, const Vec3& n, double etai_over_etat) {
////  auto cos_theta = std::fmin(dot(-uv, n), 1.0);
////  Vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
////  Vec3 r_out_parallel = -std::sqrt(std::fabs(1.0 - r_out_perp.length_squared())) * n;
////  return r_out_perp + r_out_parallel;
////}
//
//bool refract(const Vec3& v, const Vec3& n, float ni_over_nt, Vec3& refracted){
//  Vec3 uv = unit_vector(v);
//  float dt = dot(uv, n);
//  float discriminant = 1.0 - ni_over_nt*ni_over_nt*(1-dt*dt);
//  if (discriminant > 0){
//    refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
//    return true;
//  }
//  else
//    return false;
//}
//
//float schlick(float cosine, float ref_idx){
//  float r0 = (1-ref_idx) / (1+ref_idx);
//  r0 = r0*r0;
//  return r0 + (1-r0) * pow((1 - cosine), 5);
//}
//
//class dielectric : public Material {
//public:
//  dielectric (float ri) : ref_idx(ri) {}
//  bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered)
//  const override {
//    Vec3 outward_normal;
//    Vec3 reflected = reflect(r_in.direction(), rec.normal);
//    float ni_over_nt;
//    attenuation = Vec3(1.0,1.0,1.0);
//    Vec3 refracted;
//    float reflect_prob;
//    float cosine;
//    if(dot(r_in.direction(), rec.normal) > 0) {
//      outward_normal = -rec.normal;
//      ni_over_nt = ref_idx;
//      cosine = ref_idx * dot(r_in.direction(), rec.normal) / r_in.direction().length();
//    }
//    else {
//      outward_normal = rec.normal;
//      ni_over_nt = 1.0 / ref_idx;
//      cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
//    }
//    if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted)){
//      reflect_prob = schlick(cosine, ref_idx);
//    }
//    else {
//      scattered = Ray(rec.p, reflected);
//      reflect_prob = 1.0;
//    }
//    if(drand48() < reflect_prob){
//      scattered = Ray(rec.p, reflected);
//    }
//    else {
//      scattered = Ray(rec.p, refracted);
//    }
//    return true;
//  }
//
//  float ref_idx;
//
//};
//
