#include <iostream>
#include <random>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <curand_kernel.h>
#include "cuda_texture_types.h"
#include "cuda_runtime.h"

#include "Vec3.h"
#include "Ray.h"
#include "Hitable.h"
#include "Sphere.h"
#include "HitableList.h"
#include "Image.h"
#include "Camera.h"
#include "Material.h"
#include "Texture.h"
#include "ConfigParser.h"
#include "AA_Rectangles.h"
#include "Box.h"

#include <filesystem>
#include <fstream>


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
//      Vec3 unit_direction = unit_vector(cur_ray.direction());
//      float t = 0.5f*(unit_direction.y() + 1.0f);
//      Vec3 background = (1.0f-t)*Vec3(0.6, 0.6, 0.8) + t*Vec3(0.3, 0.5, 0.7);

      // c is background color
      Vec3 background = Vec3(0.0, 0.0, 0.0);
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
  for(int i=0; i < 5; i++) {
    delete ((Sphere*)d_list[i])->mat_ptr;
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
    Material *shiny = new metal(new ConstantTexture(Vec3(1.0, 1.0, 1.0)), 0.0f);

    *(d_list) = new flip_normals(new YZ_Rectangle(0, 555, 0, 555, 555, green));
    *(d_list+1) = new YZ_Rectangle(0, 555, 0, 555, 0, red);
    *(d_list+2) = new XZ_Rectangle(213, 343, 227, 332, 554, light);
    *(d_list+3) = new flip_normals( new XZ_Rectangle(0, 555, 0, 555, 555, white));
    *(d_list+4) = new XZ_Rectangle(0, 555, 0, 555, 0, white);
    *(d_list+5) = new flip_normals( new XY_Rectangle(0, 555, 0, 555, 555, white));



    *(d_list+6) = new translate( new rotate_y( new Box(Vec3(0, 0, 0), Vec3(165, 165, 165), shiny), -18), Vec3(130, 0, 65));
    *(d_list+7) = new translate( new rotate_y( new Box(Vec3(0, 0, 0), Vec3(165, 330, 165), new dielectric(1.5)), 15), Vec3(265, 0, 295));



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

__global__ void debug_texture_kernel(cudaTextureObject_t tex, float* output, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    float u = (float)x / (float)width;
    float v = (float)y / (float)height;
    float4 texel = tex2D<float4>(tex, u, 1.0f - v);
    int idx = (y * width + x) * 3;
    output[idx] = texel.x;
    output[idx+1] = texel.y;
    output[idx+2] = texel.z;
  }
}


cudaTextureObject_t createImageTexture(const char *const filename){
  int width, height, channels;
  unsigned char* img = stbi_load(filename, &width, &height, &channels, 0);
  if (!img) {
    // Handle error
    fprintf(stderr, "Failed to load image\n");
    exit(2);
  }
  printf("width: %d, height: %d, channels: %d\n", width, height, channels);

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
  size_t fb_size = 3 * num_pixels * sizeof(float);

  std::cout << "Starting ray tracer: "<< "width: " << nx << ", height: " << ny << ", samples: " << ns << std::endl;

  // load texture into gpu memory
  cudaTextureObject_t texObj = createImageTexture("../earthmap1kpng.png");

  Hitable **d_list;
  checkCudaErrors(   cudaMalloc(  (void **)&d_list, object_N*sizeof(Hitable *)));
  Hitable **d_world;
  checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(Hitable *)));
  Camera **d_camera;
  checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(Camera *)));

  create_cornell_box_scene<<<1,1>>>(d_list,d_world, d_camera, nx, ny, object_N);
//  create_spheres_scene<<<1,1>>>(d_list,d_world, d_camera, nx, ny, object_N, texObj);

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

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


//  std::ofstream outfile("../img_output.txt");
//  if (!outfile.is_open()) {exit(2);}

  // make image to write to
  Image image(nx, ny);
  // Generate the image
  for (int j = ny - 1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {
      // clamp the colors to 0-1 in case there is clipping
      fb[j*nx+i].e[0] = fb[j*nx+i].e[0] > 1.0f ? 1.0f : fb[j*nx+i].e[0];
      fb[j*nx+i].e[1] = fb[j*nx+i].e[1] > 1.0f ? 1.0f : fb[j*nx+i].e[1];
      fb[j*nx+i].e[2] = fb[j*nx+i].e[2] > 1.0f ? 1.0f : fb[j*nx+i].e[2];
//      outfile << fb[j*nx+i] << std::endl;
      image.write_pixel(i, ny - 1 - j, fb[j*nx+i]);
    }
  }
//  outfile.close();
  image.save("../output.png");

  checkCudaErrors(cudaDeviceSynchronize());

  free_cornell_box_scene<<<1, 1>>>(d_list, d_world, d_camera);
//  free_spheres_scene<<<1, 1>>>(d_list, d_world, d_camera);

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_camera));
  checkCudaErrors(cudaFree(d_world));
  checkCudaErrors(cudaFree(d_list));
  checkCudaErrors(cudaFree(d_rand_state));
  checkCudaErrors(cudaFree(fb));

  cudaDeviceReset();

  return 0;
}