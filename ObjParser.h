#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include "Vec3.h"


void parseObjFile(const std::string &filename, Vec3 *points, Vec3 *faces, int &numPoints, int &numFaces) {
  std::ifstream file(filename);

  if (!file.is_open()) { // check if the file is open
    std::cerr << "Could not open the file: " << filename << "in OBJ parser" << std::endl;
    exit(3);
  }

  int nv = 0, nf = 0;

  std::string line;
  std::vector<Vec3> tempPoints;
  std::vector<Vec3> tempVertices;

  while (std::getline(file, line)) {
    std::stringstream ss(line); // turn line into string stream

    // read the first word of the line
    std::string prefix;
    ss >> prefix;

    if (prefix == "v") {
      // Vertex position line: v x y z
      // read next three floats and create Vec3
      Vec3 point;
      ss >> point[0] >> point[1] >> point[2];
      points[nv++] = point;
    } else if (prefix == "f") {
      // Face line: f v1 v2 v3 (assuming triangle mesh)
      // read next three floats and create Vec3
      Vec3 face;
      ss >> face[0] >> face[1] >> face[2];
      faces[nf++] = face;


    }
  }

  numPoints = nv;
  numFaces = nf;

  file.close();
}
