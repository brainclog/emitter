#include "Hitable.h"
class Material {
public:
  virtual ~Material() = default;

  virtual bool scatter(
          const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered
  ) const {
    return false;
  }



};
class lambertian : public Material {
public:
  explicit lambertian(const Vec3& albedo) : albedo(albedo) {}
  bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered)
  const override {
    Vec3 target = rec.p + rec.normal + randomUnitVector();
    scattered = Ray(rec.p, target-rec.p);
    attenuation = albedo;
    return true;
  }

private:
  Vec3 albedo;
};


class metal : public Material {
public:
  explicit metal(const Vec3& albedo, double f) : albedo(albedo) {if (f < 1) fuzz = f; else fuzz = 1; }

  bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered)
  const override {
    Vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    scattered = Ray(rec.p, reflected + fuzz*randomUnitVector());
    attenuation = albedo;
    return (dot(scattered.direction(), rec.normal) > 0);
  }

private:
  Vec3 albedo;
  double fuzz;
//  double fuzz;
};
