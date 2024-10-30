#include <iostream>
#include <random>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <curand_kernel.h>
#include "cuda_texture_types.h"
#include "cuda_runtime.h"
#include <bitset>

#include "Vec3.h"
#include "Ray.h"
#include "hitable/Hitable.h"
#include "hitable/Sphere.h"
#include "hitable/HitableList.h"
#include "util/Image.h"
#include "Camera.h"
#include "Material.h"
#include "Texture.h"
#include "ConfigParser.h"
#include "hitable/AA_Rectangles.h"
#include "Box.h"
#include "hitable/Triangle.h"
#include "hitable/Mesh.h"
#include "util/ObjFile.h"
#include "BVH.h"
#include <filesystem>
#include <fstream>
#include "util/Interval.h"



// final color function
__device__ Vec3 color(const Ray& r, Hitable **world, curandState *local_rand_state) {
  Ray cur_ray = r;
  Vec3 cur_attenuation = {1.0,1.0,1.0};
  Vec3 result = {0,0,0};

  for(int i = 0; i < 50; i++) {
    HitRecord rec;

    // does the ray hit anything?
    if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
      Vec3 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);

      Ray scattered;
      Vec3 attenuation;

      if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
        result += cur_attenuation * emitted;
        cur_attenuation *= attenuation;
        cur_ray = scattered;
      }
      else {
        result += cur_attenuation * emitted;

        break;
      }
    }
    else {
      // the ray did not his anything, return background color times current attenuation..!
      Vec3 unit_direction = unit_vector(cur_ray.direction());
      float t = 0.5f*(unit_direction.y() + 1.0f);
      Vec3 background = (1.0f-t)*Vec3(0.5, 0.5, 0.75) + t*Vec3(0.3, 0.5, 0.7);

//      Vec3 background = Vec3(0.0, 0.0, 0.0);
      result += cur_attenuation * background; // maybe * instead? or just =
      break;
    }
  }
  return result; // exceeded recursion
}

__global__ void render_init(int nx, int ny, curandState *rand_state, unsigned long long SEED) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if((i >= nx) || (j >= ny)) return;
  int pixel_index = j*nx + i;
  curand_init(SEED, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(Vec3 *fb, int max_x, int max_y, int ns,
                       Camera ** cam, Hitable **world, curandState *rand_state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if((i >= max_x) || (j >= max_y)) return;
  int pixel_index = j*max_x + i;
  curandState local_rand_state = rand_state[pixel_index];
  Vec3 px_color(0,0,0);
  for(int s=0; s < ns; s++) {
    float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
    float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
    Ray r = (*cam)->getRay(u, v, &local_rand_state);
    px_color += color(r, world, &local_rand_state);
  }
  rand_state[pixel_index] = local_rand_state;
  px_color /= float(ns);
  px_color[0] = sqrt(px_color[0]);
  px_color[1] = sqrt(px_color[1]);
  px_color[2] = sqrt(px_color[2]);
  fb[pixel_index] = px_color;
}

//__global__ void make_camera(const int nx, const int ny, Camera **d_camera,
//                            const Vec3& lookfrom, const Vec3& lookat, const Vec3& vup,
//                            const float vfov, const float aperture, const float focus_dist){
////  Vec3 lookfrom(3,3,2);
////  Vec3 lookat(0,0,-1);
////  float focus_dist = (lookfrom-lookat).length();
////  float aperture = 0.3;
//  *d_camera   = new Camera(lookfrom,
//                           lookat,
//                           Vec3(0,1,0),
//                           vfov,
//                           float(nx)/float(ny),
//                           aperture,
//                           focus_dist, 0.0f, 1.0f);
//}

__global__ void create_spheres_scene(Hitable **d_list, Hitable **d_world, Camera **d_camera, int nx, int ny, int object_N, cudaTextureObject_t textureObject) {

  if (threadIdx.x == 0 && blockIdx.x == 0) {
//  create ImageTexture from cuda texture object
    ImageTexture* earth_img = new ImageTexture(textureObject);

    Texture *bigSphereChecker = new CheckerTexture(new ConstantTexture(Vec3(0.2, 0.3, 0.1)), new ConstantTexture(Vec3(0.9, 0.9, 0.9)));

    *(d_list) = new Sphere(Vec3(0, 0, -1), 0.5, new lambertian(earth_img));
//    *(d_list) = new Sphere(Vec3(0, 0, -1), 0.5, new lambertian(new ConstantTexture(Vec3(0.8, 0.2, 0.3))));
    *(d_list+1) = new Sphere(Vec3(0, -100.5, -1), 100, new lambertian(bigSphereChecker));
    *(d_list+2) = new Sphere(Vec3(1, 0, -1), 0.5, new DiffuseLight(new ConstantTexture(Vec3(1,0.9,(float)135/255))));
//    *(d_list+2) = new Sphere(Vec3(1, 0, -1), 0.5, new metal(new ConstantTexture(Vec3(0.8, 0.6, 0.2)), 0.0f));
    *(d_list+3) = new Sphere(Vec3(-1, 0, -1), 0.5, new dielectric(1.5));
    *(d_list+4) = new Sphere(Vec3(-1, 0, -1), -0.45, new dielectric(1.5));
    //triangle test
    *(d_list+5) = new Triangle(Vec3(0, 2, -2), Vec3(2, -1.2, -2), Vec3(-2, -1.2, -2), new lambertian(new ConstantTexture(Vec3(0.8, 0.2, 0.3))));
    *d_world    = new HitableList(d_list, object_N);



    Vec3 lookfrom(3,3,2);
    Vec3 lookat(0,0,-1);
    float dist_to_focus = (lookfrom-lookat).length();
    float aperture = 0.3;
    *d_camera   = new Camera(lookfrom,
                             lookat,
                             Vec3(0,1,0),
                             40.0,
                             float(nx)/float(ny),
                             aperture,
                             dist_to_focus, 0.0f, 1.0f);
  }
}

__global__ void free_spheres_scene(Hitable **d_list, Hitable **d_world, Camera **d_camera) {
  for(int i=0; i < 6; i++) {
//    delete ((Sphere*)d_list[i])->mat_ptr;
    delete d_list[i];
  }

  delete *d_world;
  delete *d_camera;
}

__global__ void create_cornell_box_scene(Hitable **d_list, Hitable **d_world, Camera **d_camera, int nx, int ny, int object_N) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    Material *red = new lambertian(new ConstantTexture(Vec3(0.65, 0.05, 0.05)));
    Material *white = new lambertian(new ConstantTexture(Vec3(0.73, 0.73, 0.73)));
    Material *green = new lambertian(new ConstantTexture(Vec3(0.12, 0.45, 0.15)));
    Material *light = new DiffuseLight(new ConstantTexture(Vec3(15, 15, 15)));
    Material *shiny = new metal(new ConstantTexture(Vec3(1.0, 1.0, 1.0)), 0.03f);

    *(d_list) = new flip_normals(new YZ_Rectangle(0, 555, 0, 555, 555, green));
    *(d_list+1) = new YZ_Rectangle(0, 555, 0, 555, 0, red);
    *(d_list+2) = new XZ_Rectangle(213, 343, 227, 332, 554, light);
    *(d_list+3) = new flip_normals( new XZ_Rectangle(0, 555, 0, 555, 555, white));
    *(d_list+4) = new XZ_Rectangle(0, 555, 0, 555, 0, white);
    *(d_list+5) = new flip_normals( new XY_Rectangle(0, 555, 0, 555, 555, white));



    *(d_list+6) = new translate( new rotate_y( new Box(Vec3(0, 0, 0), Vec3(165, 165, 165), new dielectric(1.5)), -18), Vec3(130, 0, 65));
    *(d_list+7) = new translate( new rotate_y( new Box(Vec3(0, 0, 0), Vec3(165, 330, 165), shiny), 15), Vec3(265, 0, 295));



    *d_world = new HitableList(d_list, object_N);
    Vec3 lookfrom(278, 278, -800);
    Vec3 lookat(278, 278, 0);
    float dist_to_focus = 10.0;
    float aperture = 0.0;
    *d_camera = new Camera(lookfrom,
                           lookat,
                           Vec3(0, 1, 0), 40.0,
                           float(nx) / float(ny),
                           aperture,
                           dist_to_focus, 0.0f, 1.0f);
  }
}

__global__ void free_cornell_box_scene(Hitable **d_list, Hitable **d_world, Camera **d_camera) {

  //memory leak here for now but using virtual destructors causes crash for some reason..
  for(int i=0; i < 8; i++) {
    delete d_list[i];
  }

  delete *d_world;
  delete *d_camera;
}

__global__ void meshFromTriangleArray(Hitable **d_mesh, Hitable **d_MeshTriangles, Vec3 *d_points, size_t nPoints, Vec3 *d_faces, size_t nFaces) {
  int l = 0;
  Material *mat = new lambertian(new ConstantTexture(Vec3(0.8, 0.2, 0.3)));
  for (int i = 0; i < nFaces; i++) {
    Vec3 face = d_faces[i];
    Vec3 p0 = d_points[(int)face[0]]; // get the points from the array
    Vec3 p1 = d_points[(int)face[1]];
    Vec3 p2 = d_points[(int)face[2]];
    *(d_MeshTriangles + l) = new Triangle(p0, p1, p2, mat); // fill up the array with Triangle hitables
    l++;
  }

  *d_mesh = new HitableList(d_MeshTriangles, l); // now create the HitableList with triangle array
}


__host__ Hitable** loadMeshFromOBJFile(const std::string &filename, float scale = 1.0f) {

  ObjFile teapot_obj(filename);

  size_t nPoints = teapot_obj.points.size();
  size_t pointsArraySize = nPoints * sizeof(Vec3);
  size_t nFaces = teapot_obj.faces.size();
  size_t facesArraySize = nFaces * sizeof(Vec3);

  // scale the mesh
  for (int i = 0; i < nPoints; i++) teapot_obj.points[i] *= scale;

  //  allocate memory for vertices on device
  Vec3 *d_points;
  checkCudaErrors(cudaMalloc((void **)&d_points, pointsArraySize));
  checkCudaErrors(cudaMemcpy(d_points, teapot_obj.points.data(), pointsArraySize, cudaMemcpyHostToDevice));

  //  allocate memory for faces on device
  Vec3 *d_faces;
  checkCudaErrors(cudaMalloc((void **)&d_faces, facesArraySize));
  checkCudaErrors(cudaMemcpy(d_faces, teapot_obj.faces.data(), facesArraySize, cudaMemcpyHostToDevice));

  // allocate memory for Hitable list for the triangles of the loaded mesh
  Hitable **d_MeshTriangles;
  checkCudaErrors(cudaMalloc((void **)&d_MeshTriangles, nFaces * sizeof(Hitable *)));

  Hitable **d_mesh;
  checkCudaErrors(cudaMalloc((void **)&d_mesh, sizeof(Hitable *)));
  // all the prep for the mehs is done, now create the mesh on device from the loaded data
  meshFromTriangleArray<<<1, 1>>>(d_mesh, d_MeshTriangles, d_points, nPoints, d_faces, nFaces);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // free faces and points on device
//  checkCudaErrors(cudaFree(d_faces));
//  checkCudaErrors(cudaFree(d_points));

  return d_mesh;

}


__host__ void sortByMortonCode(std::vector<GlobalTriangle>& triangles, std::vector<unsigned int>& mortonCodes) {
  // Combine into pairs
  std::vector<std::pair<unsigned int, GlobalTriangle>> pairs(triangles.size());
  for(size_t i = 0; i < triangles.size(); i++) {
    pairs[i] = {mortonCodes[i], triangles[i]};
  }

// Sort pairs based on morton codes
  std::sort(pairs.begin(), pairs.end(),
            [](const auto& a, const auto& b) {
              return a.first < b.first;
            });

// Split back into separate arrays
  for(size_t i = 0; i < pairs.size(); i++) {
    mortonCodes[i] = pairs[i].first;
    triangles[i] = pairs[i].second;
  }

  // print out pairs of the triangle and morton code (in binary)
//  for (size_t i = 0; i < mortonCodes.size(); i++) {
//    std::cout << "Triangle " << i << " has morton code: " << mortonCodes[i] << " which is: " << std::bitset<32>(mortonCodes[i]) << std::endl;
//  }
//  exit(0);

}


__global__ void convertToDeviceTriangleArray(Triangle *d_triangles, GlobalTriangle *d_global_triangles, size_t nFaces) {
  Material *mat = new lambertian(new ConstantTexture(Vec3(0.8, 0.2, 0.3)));
  for (int i = 0; i < nFaces; i++) { //TODO : parallelize loop with block thread nodesArrayIndex and stride
    Vec3 p0 = d_global_triangles[i].vertices[0]; // get the points from the array
    Vec3 p1 = d_global_triangles[i].vertices[1];
    Vec3 p2 = d_global_triangles[i].vertices[2];

    new (&d_triangles[i]) Triangle(p0, p1, p2, mat); // use placement new
  }

}

// three components: morton codes array, triangles array, # triangles

__global__ void buildBVH(Triangle *d_triangles, unsigned int* d_mortonCodes, size_t nFaces, BVH *d_bvh) {
  //  __device__ Node* generateHierarchy( unsigned int* sortedMortonCodes,
//  __device__ Node* generateHierarchy( unsigned int* sortedMortonCodes,
//                                      int*          sortedObjectIDs,
//                                      int           numObjects)
//  {

  // pointer to d_bvh is already allocated, use placement new to put it there


  generateHierarchy(d_mortonCodes, d_triangles, (int)nFaces, d_bvh);

  //loop through triangles and print all bbox min ma for each triangle
//
//  for(int i = 0; i < nFaces; i++) {
//    printf("Triangle AFTER GENERATE HIERARCHY %d: min: %f %f %f, max: %f %f %f\n", i, d_triangles[i].bbox.min().x(), d_triangles[i].bbox.min().y(), d_triangles[i].bbox.min().z(), d_triangles[i].bbox.max().x(), d_triangles[i].bbox.max().y(), d_triangles[i].bbox.max().z());
//  }





}

//goal
__host__ void load_mesh_BVH(const std::string &filename, BVH* d_bvh, float scale = 1.0f) {

  ObjFile objFile(filename);

  size_t nPoints = objFile.points.size();
  size_t pointsArraySize = nPoints * sizeof(Vec3);
  size_t nFaces = objFile.faces.size();
  size_t facesArraySize = nFaces * sizeof(Vec3);
  for (int i = 0; i < nPoints; i++) objFile.points[i] *= scale;

  // make triangles
  std::vector<GlobalTriangle> triangles;
  for (size_t i = 0; i < nFaces; i++) {
    Vec3 v0 = objFile.points[static_cast<int>(objFile.faces[i][0])];
    Vec3 v1 = objFile.points[static_cast<int>(objFile.faces[i][1])];
    Vec3 v2 = objFile.points[static_cast<int>(objFile.faces[i][2])];
    GlobalTriangle triangle(v0, v1, v2);
    triangles.push_back(triangle);
  } // bounding boxes already made in the constructor

  // loop over mesh to find out min and max values, store them for normalization later
  Vec3 mesh_min = triangles[0].bbox.centroid();
  Vec3 mesh_max = triangles[0].bbox.centroid();

  for (size_t i = 1; i < nFaces; i++) {
    Vec3 point = triangles[i].bbox.centroid();
    mesh_min = Vec3(
            fmin(mesh_min.x(), point.x()),
            fmin(mesh_min.y(), point.y()),
            fmin(mesh_min.z(), point.z())
    );
    mesh_max = Vec3(
            fmax(mesh_max.x(), point.x()),
            fmax(mesh_max.y(), point.y()),
            fmax(mesh_max.z(), point.z())
    );
  }

  std::vector<unsigned int> mortonCodes(nFaces);


  // get a normalized point and then get its morton code
  for (size_t i = 0; i < nFaces; i++) {
    Vec3 point = triangles[i].bbox.centroid();
    float x = (point.x() - mesh_min.x()) / (mesh_max.x() - mesh_min.x());
    float y = (point.y() - mesh_min.y()) / (mesh_max.y() - mesh_min.y());
    float z = (point.z() - mesh_min.z()) / (mesh_max.z() - mesh_min.z());
    unsigned int mortonCode = morton3D(x, y, z);
    mortonCodes[i] = mortonCode;
  }

  // sort triangles by morton code
  sortByMortonCode(triangles, mortonCodes);

  // memcpy morton codes to device
  unsigned int *d_morton_codes;
  checkCudaErrors(cudaMalloc((void **)&d_morton_codes, nFaces * sizeof(unsigned int)));
  checkCudaErrors(cudaMemcpy(d_morton_codes, mortonCodes.data(), nFaces * sizeof(unsigned int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaGetLastError());

  // the triangles are sorted by morton code, upload GlobalTriangle array to gpu

  GlobalTriangle *d_global_triangles;
  checkCudaErrors(cudaMalloc((void **)&d_global_triangles, nFaces * sizeof(GlobalTriangle)));
  checkCudaErrors(cudaMemcpy(d_global_triangles, triangles.data(), nFaces * sizeof(GlobalTriangle), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaGetLastError());
//  checkCudaErrors(cudaDeviceSynchronize());
  //malloc contiguous memory for Triangle array, not array of pointers

  Triangle *d_triangles;
  checkCudaErrors(cudaMalloc((void **)&d_triangles, nFaces * sizeof(Triangle)));

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  // convert GlobalTriangle array to Triangle array
  convertToDeviceTriangleArray<<<1, 1>>>(d_triangles, d_global_triangles, nFaces);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());



  buildBVH<<<1,1>>>(d_triangles, d_morton_codes, nFaces, d_bvh);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

//  exit(10);

  bbox_init_kernel<<<32,32>>>(d_bvh, d_triangles);

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

//  exit(3);






  //








//
//  //  allocate memory for vertices on device
//  Vec3 *d_points;
//  checkCudaErrors(cudaMalloc((void **)&d_points, pointsArraySize));
//  checkCudaErrors(cudaMemcpy(d_points, objFile.points.data(), pointsArraySize, cudaMemcpyHostToDevice));
//
//  //  allocate memory for faces on device
//  Vec3 *d_faces;
//  checkCudaErrors(cudaMalloc((void **)&d_faces, facesArraySize));
//  checkCudaErrors(cudaMemcpy(d_faces, objFile.faces.data(), facesArraySize, cudaMemcpyHostToDevice));
//
//
//  // allocate memory for Hitable list for the triangles of the loaded mesh
//  Hitable **d_MeshTriangles;
//  checkCudaErrors(cudaMalloc((void **)&d_MeshTriangles, nFaces * sizeof(Hitable *)));
//
//  Hitable **d_mesh;
//  checkCudaErrors(cudaMalloc((void **)&d_mesh, sizeof(Hitable *)));
//  // all the prep for the mehs is done, now create the mesh on device from the loaded data
//  meshFromTriangleArray<<<1, 1>>>(d_mesh, d_MeshTriangles, d_points, nPoints, d_faces, nFaces);
//  checkCudaErrors(cudaGetLastError());
//  checkCudaErrors(cudaDeviceSynchronize());

  // free faces and points on device
//  checkCudaErrors(cudaFree(d_faces));
//  checkCudaErrors(cudaFree(d_points));

//  return d_mesh;

}



__global__ void create_mesh_scene(Hitable **mesh, Hitable **d_world, Camera **d_camera, int nx, int ny, int object_N) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *d_world = *mesh;
    Vec3 lookfrom(10,10,10);
    Vec3 lookat(0,0,0);
    float dist_to_focus = (lookfrom-lookat).length();
    float aperture = 0.0;
    *d_camera   = new Camera(lookfrom,
                             lookat,
                             Vec3(0,1,0),
                             40.0,
                             float(nx)/float(ny),
                             aperture,
                             dist_to_focus, 0.0f, 1.0f);
  }

}

__global__ void free_mesh_scene(Hitable **mesh, Hitable **d_world, Camera **d_camera) {
  delete *mesh;
  delete *d_world;
  delete *d_camera;
}

__global__ void create_BVH_scene(BVH* d_bvh, Hitable **d_world, Camera **d_camera, int nx, int ny, int object_N) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *d_world = d_bvh->root;
    Vec3 lookfrom(10,10,10);
    Vec3 lookat(0,0,0);
    float dist_to_focus = (lookfrom-lookat).length();
    float aperture = 0.0;
    *d_camera   = new Camera(lookfrom,
                             lookat,
                             Vec3(0,1,0),
                             40.0,
                             float(nx)/float(ny),
                             aperture,
                             dist_to_focus, 0.0f, 1.0f);
  }

}

__global__ void free_BVH_scene(Hitable **mesh, Hitable **d_world, Camera **d_camera) {
  delete *mesh;
  delete *d_world;
  delete *d_camera;
}

__global__ void create_mesh_and_cornell_box_scene(Hitable **d_list, Hitable **d_world, Camera **d_camera, int nx, int ny, int object_N, Hitable** d_mesh) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    Material *red = new lambertian(new ConstantTexture(Vec3(0.65, 0.05, 0.05)));
    Material *white = new lambertian(new ConstantTexture(Vec3(0.73, 0.73, 0.73)));
    Material *green = new lambertian(new ConstantTexture(Vec3(0.12, 0.45, 0.15)));
    Material *light = new DiffuseLight(new ConstantTexture(Vec3(15, 15, 15)));
    Material *shiny = new metal(new ConstantTexture(Vec3(1.0, 1.0, 1.0)), 0.03f);

    *(d_list) = new flip_normals(new YZ_Rectangle(0, 555, 0, 555, 555, green));
    *(d_list+1) = new YZ_Rectangle(0, 555, 0, 555, 0, red);
    *(d_list+2) = new XZ_Rectangle(213, 343, 227, 332, 554, light);
    *(d_list+3) = new flip_normals( new XZ_Rectangle(0, 555, 0, 555, 555, white));
    *(d_list+4) = new XZ_Rectangle(0, 555, 0, 555, 0, white);
    *(d_list+5) = new flip_normals( new XY_Rectangle(0, 555, 0, 555, 555, white));

    *(d_list+6) = *d_mesh;




//    *(d_list+6) = new translate( new rotate_y( new Box(Vec3(0, 0, 0), Vec3(165, 165, 165), new dielectric(1.5)), -18), Vec3(130, 0, 65));
//    *(d_list+7) = new translate( new rotate_y( new Box(Vec3(0, 0, 0), Vec3(165, 330, 165), shiny), 15), Vec3(265, 0, 295));



    *d_world = new HitableList(d_list, object_N);
    Vec3 lookfrom(278, 278, -800);
    Vec3 lookat(278, 278, 0);
    float dist_to_focus = 10.0;
    float aperture = 0.0;
    *d_camera = new Camera(lookfrom,
                           lookat,
                           Vec3(0, 1, 0), 40.0,
                           float(nx) / float(ny),
                           aperture,
                           dist_to_focus, 0.0f, 1.0f);
  }
}

__global__ void free_mesh_cornell_scene(Hitable **d_list, Hitable **d_world, Camera **d_camera) {

  //memory leak here for now but using virtual destructors causes crash for some reason..
  for(int i=0; i < 7; i++) {
    delete d_list[i];
  }

  delete *d_world;
  delete *d_camera;
}

cudaTextureObject_t createImageTexture(const char *const filename){
  int width, height, channels;
  unsigned char* img = stbi_load(filename, &width, &height, &channels, 0);
  if (!img) {
    // Handle error
    fprintf(stderr, "Failed to load image texture with filename: %s\n", filename);
    exit(2);
  }
//  printf("width: %d, height: %d, channels: %d\n", width, height, channels);

  // allocate texture array on device and copy image data to it

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
  cudaArray* cuArray;
  cudaMallocArray(&cuArray, &channelDesc, width, height);

  cudaMemcpy2DToArray(cuArray, 0, 0, img, width * channels, width * channels, height, cudaMemcpyHostToDevice);

  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;

  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeWrap;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeNormalizedFloat;
  texDesc.normalizedCoords = 1;

  cudaTextureObject_t texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

  stbi_image_free(img);

  return texObj;
}


int main() {

  // read config file

  ConfigParser config("../config.txt");

  const int nx = config.getInt("nx");
  const int ny = config.getInt("ny");
  const int ns = config.getInt("ns");
  const int tx = config.getInt("tx");
  const int ty = config.getInt("ty");
  const int rSEED = config.getInt("rSEED");
  const int object_N = config.getInt("object_N");

  int num_pixels = nx*ny;

  size_t newStackSize = 1024 + 512;
  cudaError_t err = cudaDeviceSetLimit(cudaLimitStackSize, newStackSize);
  if (err != cudaSuccess) {
    printf("Error setting stack size: %s\n", cudaGetErrorString(err));
  }


//  size_t stack_size;
//  cudaDeviceGetLimit(&stack_size, cudaLimitStackSize);
//  std::cout << "Stack size: " << stack_size << std::endl;
//  exit(123)


  std::cout << "Starting ray tracer: "<< "width: " << nx << ", height: " << ny << ", samples: " << ns << std::endl;

//  Hitable **d_mesh = loadMeshFromOBJFile("../models/plane.obj", 3.f);
  BVH *d_bvh;
  checkCudaErrors(cudaMalloc((void **)&d_bvh, sizeof(BVH)));
  load_mesh_BVH("../models/cube.obj", d_bvh, 3.f);

  // load texture into gpu memory
  cudaTextureObject_t texObj = createImageTexture("../textures/earthmap1k.png");

  Hitable **d_list;
  checkCudaErrors(   cudaMalloc(  (void **)&d_list, object_N*sizeof(Hitable *)));
  Hitable **d_world;
  checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(Hitable *)));
  Camera **d_camera;
  checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(Camera *)));


//  create_mesh_and_cornell_box_scene<<<1, 1>>>(d_list, d_world, d_camera, nx, ny, object_N, d_mesh);
//  create_cornell_box_scene<<<1,1>>>(d_list,d_world, d_camera, nx, ny, object_N);
//  create_spheres_scene<<<1,1>>>(d_list,d_world, d_camera, nx, ny, object_N, texObj);
//  create_mesh_scene<<<1, 1>>>(d_mesh, d_world, d_camera, nx, ny, object_N);
  create_BVH_scene<<<1, 1>>>(d_bvh, d_world, d_camera, nx, ny, object_N);

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // allocate a cuRAND d_rand_state object for every pixel
  curandState *d_rand_state;
  checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));


  dim3 blocks((nx + tx - 1) / tx, (ny + ty - 1) / ty);
  dim3 threads(tx,ty);

  //initialize RNG
  render_init<<<blocks, threads>>>(nx, ny, d_rand_state, rSEED);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // make image to write to
  Image image(nx, ny);

  auto start = std::chrono::high_resolution_clock::now();
  // main render function
  render<<<blocks, threads>>>(image.fb, nx, ny, ns, d_camera, d_world, d_rand_state);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "Time taken: " << duration.count() << " microseconds also known as " << duration.count()/1000000.0 << " seconds" << std::endl;


  image.save("../output.png");
  image.free();

  checkCudaErrors(cudaDeviceSynchronize());

//  free_cornell_box_scene<<<1, 1>>>(d_list, d_world, d_camera);
//  free_spheres_scene<<<1, 1>>>(d_list, d_world, d_camera);

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_camera));
  checkCudaErrors(cudaFree(d_world));
  checkCudaErrors(cudaFree(d_list));
  checkCudaErrors(cudaFree(d_rand_state));


  cudaDeviceReset();

  return 0;
}