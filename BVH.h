#pragma once
#include <thrust/sort.h>

// Linear BVH constrcution . . . . . i think

// taken from NVIDIA article
// https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/

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

__device__ int findSplit( unsigned int* sortedMortonCodes,
               int           first,
               int           last)
{
  // Identical Morton codes => split the range in the middle.

  unsigned int firstCode = sortedMortonCodes[first];
  unsigned int lastCode = sortedMortonCodes[last];

  if (firstCode == lastCode)
    return (first + last) >> 1;

  // Calculate the number of highest bits that are the same
  // for all objects, using the count-leading-zeros intrinsic.

  int commonPrefix = __clz(firstCode ^ lastCode);

  // Use binary search to find where the next bit differs.
  // Specifically, we are looking for the highest object that
  // shares more than commonPrefix bits with the first one.

  int split = first; // initial guess
  int step = last - first;

  do
  {
    step = (step + 1) >> 1; // exponential decrease
    int newSplit = split + step; // proposed new position

    if (newSplit < last)
    {
      unsigned int splitCode = sortedMortonCodes[newSplit];
      int splitPrefix = __clz(firstCode ^ splitCode);
      if (splitPrefix > commonPrefix)
        split = newSplit; // accept proposal
    }
  }
  while (step > 1);

  return split;
}

__device__ int2 determineRange(unsigned int* sortedMortonCodes, int numObjects, int idx) {
  // Determine the direction of traversal
  int direction = (idx == 0) ? 1 : (__clz(sortedMortonCodes[idx] ^ sortedMortonCodes[idx - 1]) <
                                    __clz(sortedMortonCodes[idx] ^ sortedMortonCodes[idx + 1]) ? 1 : -1);

  // Find the beginning of the range
  int first = idx;
  int step = 1;
  while (first - step >= 0 && first - step < numObjects) {
    int newFirst = first - step * direction;
    if (newFirst < 0 || newFirst >= numObjects)
      break;

    unsigned int firstCode = sortedMortonCodes[idx];
    unsigned int newCode = sortedMortonCodes[newFirst];
    int prefix = __clz(firstCode ^ newCode);

    // Check if prefix differs from the common prefix
    if (prefix < __clz(firstCode ^ sortedMortonCodes[idx]))
      break;

    first = newFirst;
    step *= 2;
  }

  // Find the end of the range
  int last = idx;
  step = 1;
  while (last + step >= 0 && last + step < numObjects) {
    int newLast = last + step * direction;
    if (newLast < 0 || newLast >= numObjects)
      break;

    unsigned int lastCode = sortedMortonCodes[idx];
    unsigned int newCode = sortedMortonCodes[newLast];
    int prefix = __clz(lastCode ^ newCode);

    // Check if prefix differs from the common prefix
    if (prefix < __clz(lastCode ^ sortedMortonCodes[idx]))
      break;

    last = newLast;
    step *= 2;
  }

  // Ensure first is always less than or equal to last
  if (first > last) {
    int temp = first;
    first = last;
    last = temp;
  }

  return make_int2(first, last);
}

class BVH_Node : public Hitable {
public:
  __device__ BVH_Node(){ childA=nullptr; childB=nullptr; obj = nullptr ;parent = nullptr; processed = 0;}
//  __device__ BVH_Node(Hitable **objects, int n, float time0, float time1, curandState *local_rand_state);

  __device__ bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;
  __device__ AABB* get_bbox() override { return &bbox; }
  BVH_Node* childA;
  BVH_Node* childB;


  Hitable* obj;
  int processed;
  //Pointer to the Parent node [useful in bounding box or colision operations
  //to walk the tree from bottom to top]
  BVH_Node* parent;

  AABB bbox;

};

class BVH {
public:
  BVH_Node* root;
  BVH_Node* leafNodes;
  int numObjects;

};


__device__ BVH generateHierarchy( unsigned int* sortedMortonCodes,
//                        int*          sortedObjectIDs,
                          Hitable*      triArray,
                          int           numObjects)
{
  BVH bvh;
  BVH_Node* leafNodes = new BVH_Node[numObjects];
  BVH_Node* internalNodes = new BVH_Node[numObjects - 1];

  // Construct leaf nodes.
  // Note: This step can be avoided by storing
  // the tree in a slightly different way.

  for (int idx = 0; idx < numObjects; idx++) // in parallel
    leafNodes[idx].obj = &(triArray[idx]);
//  leafNodes[idx].objectID = sortedObjectIDs[idx];


  // Construct internal nodes.

  for (int idx = 0; idx < numObjects - 1; idx++) // in parallel
  {
    // Find out which range of objects the node corresponds to.
    // (This is where the magic happens!)

    int2 range = determineRange(sortedMortonCodes, numObjects, idx);
    int first = range.x;
    int last = range.y;

    // Determine where to split the range.

    int split = findSplit(sortedMortonCodes, first, last);

    // Select childA.

    BVH_Node* childA;
    if (split == first)
      childA = &leafNodes[split];
    else
      childA = &internalNodes[split];

    // Select childB.

    BVH_Node* childB;
    if (split + 1 == last)
      childB = &leafNodes[split + 1];
    else
      childB = &internalNodes[split + 1];

    // Record parent-child relationships.

    internalNodes[idx].childA = childA;
    internalNodes[idx].childB = childB;
    childA->parent = &internalNodes[idx];
    childB->parent = &internalNodes[idx];
  }

  // Node 0 is the root.

  bvh.root = &internalNodes[0];
  bvh.leafNodes = leafNodes;
  bvh.numObjects = numObjects;
  return bvh;
}

// build aabbs __global__ function

// now that we have a hierarchy of nodes in place, the only thing left to do is to assign a conservative bounding box for
// each of them. The approach I adopt in my paper is to do a parallel bottom-up reduction, where each thread starts from
// a single leaf node and walks toward the root. To find the bounding box of a given node, the thread simply looks up the
// bounding boxes of its children and calculates their union. To avoid duplicate work, the idea is to use an atomic flag
// per node to terminate the first thread that enters it, while letting the second one through. This ensures that every
// node gets processed only once, and not before both of its children are processed. do not use stack, we will use cuda parallelism
__global__ void bbox_init_kernel(BVH &bvh) {
  BVH_Node *leafNodes = bvh.leafNodes;
  int numObjects = bvh.numObjects;

  // Calculate global thread index with stride
  for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < numObjects;
       tid += blockDim.x * gridDim.x) {

    // Start from leaf node
    BVH_Node* currentNode = &leafNodes[tid];

    // Initialize leaf node's bbox (do not need, as triangles have init in constructor
//    computeNodeBBox(currentNode);

    // Walk up the tree
    while (currentNode->parent != nullptr) {
      BVH_Node* parent = currentNode->parent;

      // Atomically increment the processed count
      int wasProcessed = atomicAdd(&parent->processed, 1);

      if (wasProcessed == 0) {
        // First thread to reach this node - terminate
        return;
      }

      // Second thread to reach this node - compute bbox and continue up
      currentNode->parent->bbox = AABB(currentNode->parent->childA->bbox, currentNode->parent->childB->bbox);
      currentNode = parent;
    }
  }
}



//
//class BVH_Node : public Hitable {
//public:
//  __device__ BVH_Node() {}
//  __device__ BVH_Node(Hitable **objects, int n, float time0, float time1, curandState *local_rand_state);
//
//  __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const;
//  __device__ AABB* get_bbox() override { return &bbox; }
//
//  Hitable *left;
//  Hitable *right;
//  AABB box;
//
//  bool is_leaf;
//  int start, end;
//  Hitable **list;
//  int n;
//
//  AABB bbox;
//
//
//};


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
