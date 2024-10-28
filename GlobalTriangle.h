#pragma once

// hacking together BAD solution, this is BAD!! NOT GOOD

class GlobalAABB { // helper for global triangle
public:
  GlobalAABB() {}
  GlobalAABB(const Vec3& a, const Vec3& b) { _min = a; _max = b; }

  Vec3 min() const { return _min; }
  Vec3 max() const { return _max; }

  Vec3 _min;
  Vec3 _max;

  Vec3 centroid() const {
    return (_min + _max) / 2;
  }

};

class GlobalTriangle { // helper for loading the mesh and constructing BVH
public:
  __host__ __device__ GlobalTriangle() {}

  __host__ __device__ GlobalTriangle(Vec3 v0, Vec3 v1, Vec3 v2){
    vertices[0] = v0;
    vertices[1] = v1;
    vertices[2] = v2;
    make_bounding_box();
  }

  __host__ __device__ void make_bounding_box();


  Vec3 vertices[3];
  GlobalAABB bbox;


};

__host__ __device__ void GlobalTriangle::make_bounding_box() {

  float minX = fmin(vertices[0][0], fmin(vertices[1][0], vertices[2][0]));
  float minY = fmin(vertices[0][1], fmin(vertices[1][1], vertices[2][1]));
  float minZ = fmin(vertices[0][2], fmin(vertices[1][2], vertices[2][2]));

  float maxX = fmax(vertices[0][0], fmax(vertices[1][0], vertices[2][0]));
  float maxY = fmax(vertices[0][1], fmax(vertices[1][1], vertices[2][1]));
  float maxZ = fmax(vertices[0][2], fmax(vertices[1][2], vertices[2][2]));

  bbox = {Vec3(minX, minY, minZ), Vec3(maxX, maxY, maxZ)};
}