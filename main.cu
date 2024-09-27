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


class RandomGenerator {
private:
  std::default_random_engine engine;
  std::uniform_real_distribution<double> distribution; // Range [0, 1)
public:
  // no seed
  RandomGenerator() : engine(std::random_device{}()), distribution(0.0, 1.0) {}
  // with seed
  RandomGenerator(unsigned int seed) : engine(seed), distribution(0.0, 1.0) {}

  double getDouble() {
    return distribution(engine);
  }
};




// final color function from cpu side



__device__ Vec3 color(const Ray& r, Hitable **world) {
  HitRecord rec;
  if ((*world)->hit(r, 0.001, FLT_MAX, rec)) {
    return 0.5f*Vec3(rec.normal.x()+1.0f, rec.normal.y()+1.0f, rec.normal.z()+1.0f);
  }
  else {
    Vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f-t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
  }
}

__global__ void render_init(int nx, int ny, curandState *rand_state, unsigned long long SEED) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if((i >= nx) || (j >= ny)) return;
  int pixel_index = j*nx + i;
  //Each thread gets same seed, a different sequence number, no offset
  curand_init(SEED, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(Vec3 *fb, int max_x, int max_y,
                       Vec3 lower_left_corner,
                       Vec3 horizontal,
                       Vec3 vertical,
                       Vec3 origin, Hitable **world, curandState *rand_state){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if((i >= max_x) || (j >= max_y)) return;
  int pixel_index = j*max_x + i;
  curandState local_rand_state = rand_state[pixel_index];
  Vec3 col(0,0,0);
  float u = float(i) / float(max_x);
  float v = float(j) / float(max_y);
  Ray r(origin, lower_left_corner + u*horizontal + v*vertical);
  fb[pixel_index] = color(r, world);
}

__global__ void create_world(Hitable **d_list, Hitable **d_world) {

  //  const int object_N = 5;
//  Hitable* list[object_N];
//  list[0] = new Sphere(Vec3(0, 0, -1), 0.5, std::make_shared<lambertian>(Vec3(0.8, 0.2, 0.3)));
//  list[1] = new Sphere(Vec3(0, -100.5, -1), 100, std::make_shared<lambertian>(Vec3(0.8, 0.8, 0.1)));
//  list[2] = new Sphere(Vec3(1, 0, -1), 0.5, std::make_shared<metal>(Vec3(0.8, 0.6, 0.2), 0.0));
//  list[3] = new Sphere(Vec3(-1, 0, -1), 0.5, std::make_shared<dielectric>(1.5));
//  list[4] = new Sphere(Vec3(-1, 0, -1), -0.45, std::make_shared<dielectric>(1.5));
//  Hitable* world = new HitableList(list, object_N);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *(d_list)   = new Sphere(Vec3(0,0,-1), 0.5);
    *(d_list+1) = new Sphere(Vec3(0,-100.5,-1), 100);
    *d_world    = new HitableList(d_list,2);
  }
}

__global__ void free_world(Hitable **d_list, Hitable **d_world) {
  delete *(d_list);
  delete *(d_list+1);
  delete *d_world;
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

  Hitable **d_list;
  checkCudaErrors(   cudaMalloc(  (void **)&d_list  , 2*sizeof(Hitable *)));
  Hitable **d_world;
  checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(Hitable *)));
  Camera **d_camera;
  checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(Camera *)));

  create_world<<<1,1>>>(d_list,d_world, d_camera, nx, ny);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());


  // allocate FB
  Vec3 *fb;
  checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size)   );

  // allocate a cuRAND d_rand_state object for every pixel
  curandState *d_rand_state;
  checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));

  auto start = std::chrono::high_resolution_clock::now();


  dim3 blocks((nx + tx - 1) / tx, (ny + ty - 1) / ty);
  dim3 threads(tx,ty);

  //initialize RNG
  render_init<<<blocks, threads>>>(nx, ny, d_rand_state, rSEED);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  //  Vec3 lookfrom(3,3,2);
  //  Vec3 lookat(0,0,-1);
  //  float dist_to_focus = (lookfrom-lookat).length();
  //  float aperture = 2.0;
  //  Camera camera = Camera(lookfrom, lookat, Vec3(0,1,0) ,20, float(nx)/float(ny), aperture, dist_to_focus);



  // main render function
  render<<<blocks, threads>>>(fb, nx, ny,
                              Vec3(-2.0, -1.0, -1.0),
                              Vec3(4.0, 0.0, 0.0),
                              Vec3(0.0, 2.0, 0.0),
                              Vec3(0.0, 0.0, 0.0), d_world);
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
//      Vec3 px_color = fb[j*nx + i];
      image.write_pixel(i, ny - 1 - j, fb[j*nx+i]);
//
//      for(int s = 0; s < ns; s++){
//        float u = float(i + rng.getDouble()) / float(nx);
//        float v = float(j + rng.getDouble()) / float(ny);
//        //      Ray camera_ray(origin, lower_left_corner + u * horizontal + v * vertical);
//        Ray camera_ray = camera.getRay(u,v);
//        px_color += color(camera_ray, world, 0);
//      }
//      px_color /= float(ns);
//
//      // gamma 1.8
//      px_color = Vec3(std::pow(px_color.e[0], 1/1.8), std::pow(px_color.e[1], 1/1.8), std::pow(px_color.e[2], 1/1.8));
//
//      // write pixel color

    }
  }

  image.save("../output.png");

  checkCudaErrors(cudaDeviceSynchronize());
  free_world<<<1,1>>>(d_list,d_world);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_list));
  checkCudaErrors(cudaFree(d_world));
  checkCudaErrors(cudaFree(fb));

  return 0;
}
