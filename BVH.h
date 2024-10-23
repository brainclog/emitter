#pragma once
#include <thrust/sort.h>

// Linear BVH constrcution . . . . . i think

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
unsigned int expandBits(unsigned int v)
{
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
unsigned int morton3D(float x, float y, float z)
{
  x = fmin(fmax(x * 1024.0f, 0.0f), 1023.0f);
  y = fmin(fmax(y * 1024.0f, 0.0f), 1023.0f);
  z = fmin(fmax(z * 1024.0f, 0.0f), 1023.0f);
  unsigned int xx = expandBits((unsigned int)x);
  unsigned int yy = expandBits((unsigned int)y);
  unsigned int zz = expandBits((unsigned int)z);
  return xx * 4 + yy * 2 + zz;
}

class BVH_Node : public Hitable {
public:
  __device__ BVH_Node() {}
  __device__ BVH_Node(Hitable **objects, int n, float time0, float time1, curandState *local_rand_state);

  __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const;
  __device__ virtual bool bounding_box(float t0, float t1, AABB& box) const;
  Hitable *left;
  Hitable *right;
  AABB box;

  bool is_leaf;
  int start, end;
  Hitable **list;
  int n;


};


__device__ bool BVH_Node::bounding_box(float t0, float t1, AABB &b) const {
  b = box;
  return true;
}

////
////struct BoxCmp {
////  __device__ BoxCmp(int axis) : axis(axis) {}
////
////  __device__ bool operator()(Hitable* a, Hitable* b) {
////    AABB box_left, box_right;
////
////    if (!a->bounding_box(0, 0, box_left) || !b->bounding_box(0, 0, box_right)) {
////      printf("No bounding box in bvh_node constructor.\n");
////      return false;
////    }
////    float left_min, right_min;
////    if (axis == 1) {
////      left_min = box_left.min().x();
////      right_min = box_right.min().x();
////    } else if (axis == 2) {
////      left_min = box_left.min().y();
////      right_min = box_right.min().y();
////    } else { // mens axis == 3;
////      left_min = box_left.min().z();
////      right_min = box_right.min().z();
////    }
////
////    return left_min < right_min;
////  }
////
////  // Axis: 1 = x, 2 = y, 3 = z
////  int axis;
////};
//
//__device__ BVH_Node::BVH_Node(Hitable **objects, int n, float time0, float time1, curandState *local_rand_state) {
//    int axis = int(3 * curand_uniform(local_rand_state));
//  if (axis == 0) {
//  thrust::sort(objects, objects + n, BoxCmp(1));
//  }
//  else if (axis == 1) {
//  thrust::sort(objects, objects + n, BoxCmp(2));
//  } else {
//  thrust::sort(objects, objects + n, BoxCmp(3));
//  }
//
//  if (n == 1) {
//  left = right = objects[0];
//  } else if (n == 2) {
//  left = objects[0];
//  right = objects[1];
//  } else {
//  left = new BVH_Node(objects, n / 2, time0, time1, local_rand_state);
//  right = new BVH_Node(objects + n / 2, n - n / 2, time0, time1, local_rand_state);
//  }
//
//  AABB box_left, box_right;
//
//  if (!left->bounding_box(time0, time1, box_left) || !right->bounding_box(time0, time1, box_right)) {
//  return;
//  }
//
//  box = surrounding_box(box_left, box_right);
//}
//
//
//__device__ bool BVH_Node::hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const {
//  if(box.hit(r, t_min, t_max)) {
//    HitRecord left_rec, right_rec;
//
//    //test left and right hitable objs
//    bool hit_left = left->hit(r, t_min, t_max, left_rec);
//    bool hit_right = right->hit(r, t_min, t_max, right_rec);
//
//    // if hit both, return the closest one
//    if(hit_left && hit_right) {
//      if(left_rec.t < right_rec.t) {
//        rec = left_rec;
//      } else {
//        rec = right_rec;
//      }
//      return true;
//    }
//    else if(hit_left) {
//      rec = left_rec;
//      return true;
//    } else if(hit_right) {
//      rec = right_rec;
//      return true;
//    } else {
//      return false;
//    }
//  }
//  else return false;
//}
