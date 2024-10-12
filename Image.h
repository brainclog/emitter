#pragma once
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Image writing helper fun
class Image {
public:
  // Constructor initializes image size and allocates memory
  Image(int width, int height) : nx(width), ny(height) {
    image_data = new unsigned char[nx * ny * 3]; // 3 channels for RGB
  }

  // Destructor frees allocated memory
  ~Image() {
    delete[] image_data;
  }

  // Function to write pixel color at specified (i, j) position
  void write_pixel(int i, int j, const Vec3& color) {

    int index = (j * nx + i) * 3;
    image_data[index + 0] = static_cast<unsigned char>(255.99f * color.x()); // R
    image_data[index + 1] = static_cast<unsigned char>(255.99f * color.y()); // G
    image_data[index + 2] = static_cast<unsigned char>(255.99f * color.z()); // B
  }

  // Function to save the image to a file
  void save(const std::string& filename) {
    stbi_write_png(filename.c_str(), nx, ny, 3, image_data, nx * 3);
  }

private:
  int nx, ny;
  unsigned char* image_data; // Pointer to image buffer
};

