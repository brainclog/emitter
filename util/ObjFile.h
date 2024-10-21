#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include "Vec3.h"

class ObjFile {
public:
  ObjFile(const std::string &filename) {
    parseObjFile(filename);
  }

  __host__ void parseObjFile(const std::string &filename) {
    std::ifstream file(filename);

    if (!file.is_open()) { // check if the file is open
      std::cerr << "Could not open the file: " << filename << "in OBJ parser" << std::endl;
      exit(3);
    }

    std::cout << "Parsing OBJ file: " << filename << std::endl;


    std::string line;

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
        points.push_back(point);

      } else if (prefix == "f") {

        // Face line: f v1 v2 v3 (assuming triangle mesh)
        // read next three floats and create Vec3

        Vec3 face;
        ss >> face[0] >> face[1] >> face[2];

        face = face - Vec3(1,1,1); // specification is NOT zero indexed, subtract 1 from each index
        // print out face
//         std::cout << "Reading face index " << faces.size() << " : " << face[0] << " " << face[1] << " " << face[2] << std::endl;
        faces.push_back(face);

      }
    }


    std::cout << "OBJ file parsing successful." << filename << std::endl;
    std::cout << "Number of vertices: " << points.size() << std::endl;
    std::cout << "Number of faces: " << faces.size() << std::endl;


    file.close();
  }


  std::vector<Vec3> points;
  std::vector<Vec3> faces;




};

