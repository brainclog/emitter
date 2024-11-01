#pragma once
#include <thrust/sort.h>

// Linear BVH constrcution . . . . . i think

// taken from NVIDIA article
// https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.


// debug function to print binary of a number
__device__ void printBinary(unsigned int n) {
  for(int i = 31; i >= 0; i--) {
    printf("%d", (n >> i) & 1);
  }
}

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

//__device__ int commonPrefixLength(unsigned int a, unsigned int b) {
//  return __clz(a ^ b);
//}
__device__ int prefixLength(int idx1, int idx2, unsigned int* sortedMortonCodes, int numObjects) {
  if (idx2 < 0 || idx2 >= numObjects) return -1;
  unsigned int code1 = sortedMortonCodes[idx1];
  unsigned int code2 = sortedMortonCodes[idx2];
  return __clz(code1 ^ code2); // use the CUDA intrinsic __clz
}

__device__ int2 determineRange(unsigned int* sortedMortonCodes, int numObjects, int idx) {
  if (idx == 0) {
    return make_int2(0, numObjects - 1);
  }

  // Determine direction of the range (+1 or -1)
  int direction = (prefixLength(idx, idx + 1, sortedMortonCodes, numObjects) >
                   prefixLength(idx, idx - 1, sortedMortonCodes, numObjects)) ? 1 : -1;

  // Compute the upper bound for the length of the range
  int deltaMin = prefixLength(idx, idx - direction, sortedMortonCodes, numObjects);
  int lMax = 2;
  while (prefixLength(idx, idx + lMax * direction, sortedMortonCodes, numObjects) > deltaMin) {
    lMax *= 2;
  }

  // Find the other end using binary search
  int l = 0;
  int t = lMax / 2;
  while (t >= 1) {
    if (prefixLength(idx, idx + (l + t) * direction, sortedMortonCodes, numObjects) > deltaMin) {
      l += t;
    }
    t /= 2;
  }
  int j = idx + l * direction;

  return (direction < 0) ? make_int2(j, idx) : make_int2(idx, j);
}
//
//__device__ int2 determineRange(unsigned int* sortedMortonCodes,
//                               int numObjects,
//                               int idx)
//{
//  int2 range;
//  printf(" --- \nStarting determineRange with idx: %d, ", idx);
//
//  // Handle edge cases
//  if (numObjects <= 1) {
//    range.x = 0;
//    range.y = 0;
//    return range;
//  }
//
//  unsigned int targetCode = sortedMortonCodes[idx];
//
//  // print binary of targetCode
//  printf("Target morton code:");
//  printBinary(targetCode);
//  printf("\n");
//
//
//
//  // Binary search for lower bound
//  int left = 0;
//  int right = idx;
//  int commonPrefix = __clz(targetCode ^ sortedMortonCodes[left]);
//
//  printf("Determining lower bound:\n");
//
//  printf(" Starting first iteration, left: %d, right: %d, commonPrefix: %d\n", left, right, commonPrefix);
//
//  while (left < right) {
//    printf(" In while loop, left: %d, right: %d\n", left, right);
//
//    int mid = (left + right) >> 1; // find middle of left and right, left is 0 on first iteration
//    int midPrefix = __clz(targetCode ^ sortedMortonCodes[mid]);
//
//    printf("(targetCode, midCode): ");
//    printBinary(targetCode);
//    printf(", ");
//    printBinary(sortedMortonCodes[mid]);
//    printf("\n");
//
//    printf(" Calculated: mid: %d, midPrefix: %d\n", mid, midPrefix);
//
//    if (midPrefix == commonPrefix) {
//      printf(" Mid Prefix == common prefix, setting right to mid\n");
//      // If we find same prefix, look in lower half
//      right = mid;
//    } else {
//      printf(" Mid Prefix != common prefix, setting left to mid + 1\n");
//      // If prefix is different, look in upper half
//      left = mid + 1;
//    }
//  }
//  printf(" Exited while loop, lower bound is: %d\n\n", left);
//  range.x = left;  // first nodesArrayIndex
//
//  printf("Determining upper bound:\n");
//
//
//  // Binary search for upper bound
//  left = idx;
//  right = numObjects - 1;
//
//  printf(" Starting first iteration, left: %d, right: %d, commonPrefix: %d\n", left, right, commonPrefix);
//
//  while (left < right) {
//    printf(" In while loop, left: %d, right: %d\n", left, right);
//
//    // Round up for the mid point to ensure we make progress
//    // when left and right differ by 1
//
//    int mid = (left + right + 1) >> 1;
//    int midPrefix = __clz(targetCode ^ sortedMortonCodes[mid]);
//
//    printf("(targetCode, midCode): ");
//    printBinary(targetCode);
//    printf(", ");
//    printBinary(sortedMortonCodes[mid]);
//    printf("\n");
//
//    printf(" Calculated: mid: %d, midPrefix: %d\n", mid, midPrefix);
//
//
//
//    if (midPrefix == commonPrefix) {
//      printf(" Mid Prefix == common prefix, setting right to mid\n");
//
//      // If we find same prefix, look in upper half
//      left = mid;
//    } else {
//      printf(" Mid Prefix != common prefix, setting left to mid + 1\n");
//
//      // If prefix is different, look in lower half
//      right = mid - 1;
//    }
//  }
//  printf(" Exited while loop, upper bound is: %d\n\n", left);
//  range.y = left;  // last nodesArrayIndex
//
//  printf("Returning range: %d, %d\n --- \n", range.x, range.y);
//  return range;
//}

class BVH_Node : public Hitable {
public:
  __device__ BVH_Node(){ childA=nullptr; childB=nullptr; obj = nullptr ;parent = nullptr; processed = 0; bbox = AABB::empty();}

  __device__ bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;
  __device__ AABB* get_bbox() override { return &bbox; }

  BVH_Node* childA;
  BVH_Node* childB;

  Triangle* obj; // internal node is nullptr
  int processed; // atomic int for building aabb

  BVH_Node* parent;

  AABB bbox;
//  int nodesArrayIndex;

};

class BVH {
public:
  BVH_Node* root;
  BVH_Node* leafNodes;
  int numObjects;

};

__device__ void generateHierarchy( unsigned int* sortedMortonCodes,
//                        int*          sortedObjectIDs,
                          Triangle*      triArray,
                          int           numObjects,
                          BVH* bvh)
{
  // use placement new to create the

  BVH_Node* leafNodes = new BVH_Node[numObjects];
  BVH_Node* internalNodes = new BVH_Node[numObjects - 1];

  // Construct leaf nodes.
  // Note: This step can be avoided by storing
  // the tree in a slightly different way.

  for (int idx = 0; idx < numObjects; idx++) {
    leafNodes[idx].obj = &(triArray[idx]);
    leafNodes[idx].bbox = triArray[idx].bbox;
//    triArray[idx].nodesArrayIndex = idx;
//    leafNodes[idx].nodesArrayIndex = idx;
//
//    printf("Triangle %d, obj: %p\n", idx, leafNodes[idx].obj);
//    // display vertices of the triangle
//    printf(" with vertices: \n");
//    for (int i = 0; i < 3; i++) {
//      printf(" Vertex %d: (%f, %f, %f)\n", i, triArray[idx].vertices[i].x(), triArray[idx].vertices[i].y(), triArray[idx].vertices[i].z());
//    }
//    printf("Triangle at nodesArrayIndex %d has vertices: (%f, %f, %f), (%f, %f, %f), (%f, %f, %f) , bbox with min: (%f, %f, %f) and max: (%f, %f, %f)\n ",idx,
//           triArray[idx].vertices[0].x(), triArray[idx].vertices[0].y(), triArray[idx].vertices[0].z(),
//           triArray[idx].vertices[1].x(), triArray[idx].vertices[1].y(), triArray[idx].vertices[1].z(),
//           triArray[idx].vertices[2].x(), triArray[idx].vertices[2].y(), triArray[idx].vertices[2].z(),
//           triArray[idx].bbox.min().x(), triArray[idx].bbox.min().y(), triArray[idx].bbox.min().z(),
//           triArray[idx].bbox.max().x(), triArray[idx].bbox.max().y(), triArray[idx].bbox.max().z());
  }

//  leafNodes[idx].objectID = sortedObjectIDs[idx];


  // Construct internal nodes.

  for (int idx = 0; idx < numObjects - 1; idx++) // in parallel
  {
    // Find out which range of objects the node corresponds to.
    // (This is where the magic happens!)

    int2 range = determineRange(sortedMortonCodes, numObjects, idx);
    // print range  and nodesArrayIndex
//     printf("idx: %d, range: %d, %d\n", idx, range.x, range.y);
    int first = range.x;
    int last = range.y;

    // Determine where to split the range.

    int split = findSplit(sortedMortonCodes, first, last);

//    assert(first >= 0 && first < numObjects);
//    assert(last > first && last <= numObjects);
//    assert(split >= first && split < last);

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

    // print idx
    // printf("idx: %d\n", idx);
    childA->parent = &internalNodes[idx];
    childB->parent = &internalNodes[idx];
  }

  // Node 0 is the root.

//  for (int i = 0; i < numObjects - 1; i++) {
//    internalNodes[i].nodesArrayIndex = i;
//  }

  bvh->root = &internalNodes[0];
  bvh->leafNodes = leafNodes;
  bvh->numObjects = numObjects;
}

// build aabbs __global__ function

// now that we have a hierarchy of nodes in place, the only thing left to do is to assign a conservative bounding box for
// each of them. The approach I adopt in my paper is to do a parallel bottom-up reduction, where each thread starts from
// a single leaf node and walks toward the root. To find the bounding box of a given node, the thread simply looks up the
// bounding boxes of its children and calculates their union. To avoid duplicate work, the idea is to use an atomic flag
// per node to terminate the first thread that enters it, while letting the second one through. This ensures that every
// node gets processed only once, and not before both of its children are processed. do not use stack, we will use cuda parallelism
__global__ void bbox_init_kernel(BVH *bvh,Triangle *d_triangles ) {
  BVH_Node *leafNodes = bvh->leafNodes;
  int numObjects = bvh->numObjects;

//  for (int i = 0; i < numObjects; i++) {
//    // print out bbox  vertices for each triangle
//    printf("bbox intervals are: x(%f, %f), y(%f, %f), z(%f, %f)\n", d_triangles[i].bbox.x.min, d_triangles[i].bbox.x.max,
//           leafNodes[i].bbox.y.min, leafNodes[i].bbox.y.max, d_triangles[i].bbox.z.min, d_triangles[i].bbox.z.max);
//  }


  // Calculate global thread nodesArrayIndex with stride
  for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < numObjects;
       tid += blockDim.x * gridDim.x) {
    //print tid
//    printf("tid: %d\n", tid);

    // Start from leaf node
    BVH_Node* currentNode = &leafNodes[tid];
    // triangle at nodesArrayIndex tid has bbox with this max and min:
//    printf("Triangle at nodesArrayIndex %d has vertices: (%f, %f, %f), (%f, %f, %f), (%f, %f, %f) , bbox with min: (%f, %f, %f) and max: (%f, %f, %f)\n ",tid,
//           currentNode->obj->vertices[0].x(), currentNode->obj->vertices[0].y(), currentNode->obj->vertices[0].z(),
//           currentNode->obj->vertices[1].x(), currentNode->obj->vertices[1].y(), currentNode->obj->vertices[1].z(),
//           currentNode->obj->vertices[2].x(), currentNode->obj->vertices[2].y(), currentNode->obj->vertices[2].z(),
//           currentNode->bbox.min().x(), currentNode->bbox.min().y(), currentNode->bbox.min().z(),
//           currentNode->bbox.max().x(), currentNode->bbox.max().y(), currentNode->bbox.max().z());

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



__device__ bool BVH_Node::hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const {
  // traverse the tree, stack based; its a binary tree is max depth is log2(nFaces), so lets just set max depth to 32
//  printf("BVH Node Hit");
  BVH_Node* stack[16];
  int stackPtr = 0;
  stack[stackPtr++] = (BVH_Node*)this;
  while(stackPtr > 0) {
    BVH_Node* node = stack[--stackPtr];


//    printf("---\nChecking BVH node with nodesArrayIndex %d, is leaf: %d\n", node->nodesArrayIndex, node->obj != nullptr);
//    int triIndex = -1;
//    if (node->obj != nullptr) triIndex = node->obj->nodesArrayIndex;



//    if(true) {
    if(node->bbox.hit(r, t_min, t_max)) {

      //    if(true) {
//      printf(" bbox hit\n");
      // leaf node
      if(node-> obj != nullptr) {
//        printf("Leaf node, testing collision with triangle, triangle has nodesArrayIndex %d\n", node->obj->triArrayIndex);


//        return node->obj->hit(r, t_min, t_max, rec);
        bool hit = node->obj->hit(r, t_min, t_max, rec);
        if (hit) {
//          printf("Hit inside BVH node ");
          return true;
        }

      } else { // internal node, add children to stack
//        printf("Adding children to stack: nodes with indices %d, %d\n", node->childA->nodesArrayIndex, node->childB->nodesArrayIndex);
        stack[stackPtr++] = node->childA;
        stack[stackPtr++] = node->childB;
      }
    }

  }
  // no hit
  return false;
}
