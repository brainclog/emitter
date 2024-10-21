#pragma once
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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

// Image writing helper fun
class Image {
public:
  __host__ Image(int width, int height) : nx(width), ny(height) {
    image_data = new unsigned char[nx * ny * 3]; // 3 channels for RGB

    int num_pixels = nx*ny;
    size_t fb_size = 3 * num_pixels * sizeof(float);
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

  }



  ~Image() {
    delete[] image_data;
  }

  void free() {
    checkCudaErrors(cudaFree(fb));
  }

  void write_pixel(int i, int j, const Vec3& color) {

    int index = (j * nx + i) * 3;
    image_data[index + 0] = static_cast<unsigned char>(255.99f * color.x()); // R
    image_data[index + 1] = static_cast<unsigned char>(255.99f * color.y()); // G
    image_data[index + 2] = static_cast<unsigned char>(255.99f * color.z()); // B
  }

  void save(const std::string& filename) {

    for (int j = ny - 1; j >= 0; j--) {
      for (int i = 0; i < nx; i++) {
        // clamp the colors to 0-1 in case there is clipping
        fb[j*nx+i].e[0] = fb[j*nx+i].e[0] > 1.0f ? 1.0f : fb[j*nx+i].e[0];
        fb[j*nx+i].e[1] = fb[j*nx+i].e[1] > 1.0f ? 1.0f : fb[j*nx+i].e[1];
        fb[j*nx+i].e[2] = fb[j*nx+i].e[2] > 1.0f ? 1.0f : fb[j*nx+i].e[2];
        write_pixel(i, ny - 1 - j, fb[j*nx+i]);
      }
    }


    stbi_write_png(filename.c_str(), nx, ny, 3, image_data, nx * 3);
  }

public:
  // allocate FB
  Vec3 *fb;
  int nx, ny;
  unsigned char* image_data; // Pointer to image buffer
};

