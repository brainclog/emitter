#include <iostream>
#include <random>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <curand_kernel.h>

#include "Vec3.h"
#include "Ray.h"
#include "Hitable.h"
#include "Sphere.h"
#include "HitableList.h"
#include "Image.h"
#include "Camera.h"
#include "Material.h"



// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
  if (result) {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
              file << ":" << line << " '" << func << "' \n";
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(99);
  }
}


// final color function
__device__ Vec3 color(const Ray& r, Hitable **world, curandState *local_rand_state) {
  Ray cur_ray = r;
  Vec3 cur_attenuation = Vec3(1.0,1.0,1.0);
  for(int i = 0; i < 50; i++) {
    HitRecord rec;
    if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
      Ray scattered;
      Vec3 attenuation;
      if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
        cur_attenuation *= attenuation;
        cur_ray = scattered;
      }
      else {
        return {0.0,0.0,0.0};
      }
    }
    else {
      Vec3 unit_direction = unit_vector(cur_ray.direction());
      float t = 0.5f*(unit_direction.y() + 1.0f);
      Vec3 c = (1.0f-t)*Vec3(1.0, 1.0, 1.0) + t*Vec3(0.5, 0.7, 1.0);
      return cur_attenuation * c;
    }
  }
  return {0.0,0.0,0.0}; // exceeded recursion
}

__global__ void render_init(int nx, int ny, curandState *rand_state, unsigned long long SEED) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if((i >= nx) || (j >= ny)) return;
  int pixel_index = j*nx + i;
  curand_init(SEED, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(Vec3 *fb, int max_x, int max_y, int ns,
                       Camera ** cam, Hitable **world, curandState *rand_state){
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

__global__ void create_world(Hitable **d_list, Hitable **d_world, Camera **d_camera, int nx, int ny, int object_N) {

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *(d_list) = new Sphere(Vec3(0, 0, -1), 0.5, new lambertian(Vec3(0.8, 0.2, 0.3)));
    *(d_list+1) = new Sphere(Vec3(0, -100.5, -1), 100, new lambertian(Vec3(0.8, 0.8, 0.1)));
    *(d_list+2) = new Sphere(Vec3(1, 0, -1), 0.5, new metal(Vec3(0.8, 0.6, 0.2), 0.0));
    *(d_list+3) = new Sphere(Vec3(-1, 0, -1), 0.5, new dielectric(1.5));
    *(d_list+4) = new Sphere(Vec3(-1, 0, -1), -0.45, new dielectric(1.5));
    *d_world    = new HitableList(d_list, object_N);
    Vec3 lookfrom(3,3,2);
    Vec3 lookat(0,0,-1);
    float dist_to_focus = (lookfrom-lookat).length();
    float aperture = 2.0;
    *d_camera   = new Camera(lookfrom,
                             lookat,
                             Vec3(0,1,0),
                             20.0,
                             float(nx)/float(ny),
                             aperture,
                             dist_to_focus);
  }
}

__global__ void free_world(Hitable **d_list, Hitable **d_world, Camera **d_camera) {
  for(int i=0; i < 5; i++) {
    delete ((Sphere*)d_list[i])->mat_ptr; // this line makes the code crash ! ! ! ! ! ! ! !
    delete d_list[i];
  }

  delete *d_world;
  delete *d_camera;

}


int main() {
  const int nx = 800;
  const int ny = 400;
  const int ns = 100;
  int tx = 8;
  int ty = 8;
  const int rSEED = 1;

  int num_pixels = nx*ny;
  size_t fb_size = 3 * num_pixels * sizeof(float);

  const int object_N = 5;

  Hitable **d_list;
  checkCudaErrors(   cudaMalloc(  (void **)&d_list  , object_N*sizeof(Hitable *)));
  Hitable **d_world;
  checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(Hitable *)));
  Camera **d_camera;
  checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(Camera *)));

  create_world<<<1,1>>>(d_list,d_world, d_camera, nx, ny, object_N);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

//  free_world<<<1,1>>>(d_list,d_world, d_camera);
//  checkCudaErrors(cudaGetLastError());
//  checkCudaErrors(cudaDeviceSynchronize());

  // allocate FB
  Vec3 *fb;
  checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size)   );

  // allocate a cuRAND d_rand_state object for every pixel
  curandState *d_rand_state;
  checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));



  dim3 blocks((nx + tx - 1) / tx, (ny + ty - 1) / ty);
  dim3 threads(tx,ty);

  //initialize RNG
  render_init<<<blocks, threads>>>(nx, ny, d_rand_state, rSEED);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  auto start = std::chrono::high_resolution_clock::now();
  // main render function
  render<<<blocks, threads>>>(fb, nx, ny, ns, d_camera, d_world, d_rand_state);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "Time taken: " << duration.count() << " microseconds" << std::endl;

  // make image to write to
  Image image(nx, ny);
  // Generate the image
  for (int j = ny - 1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {
      image.write_pixel(i, ny - 1 - j, fb[j*nx+i]);
    }
  }

  image.save("../output.png");

  checkCudaErrors(cudaDeviceSynchronize());
  free_world<<<1,1>>>(d_list,d_world, d_camera);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_camera));
  checkCudaErrors(cudaFree(d_world));
  checkCudaErrors(cudaFree(d_list));
  checkCudaErrors(cudaFree(d_rand_state));
  checkCudaErrors(cudaFree(fb));

  cudaDeviceReset();

  return 0;
}