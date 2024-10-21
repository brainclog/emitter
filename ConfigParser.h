#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

class ConfigParser {
public:

  ConfigParser(const std::string& filename) {
    parseFile(filename);
  }

  // retrieve int from config
  int getInt(const std::string& key) const {
    auto it = config.find(key);
    if (it != config.end()) {
      return std::stoi(it->second);
    }
    throw std::runtime_error("Key '" + key + "' not found in configuration.");
  }

private:
  std::unordered_map<std::string, std::string> config;

  void parseFile(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
      throw std::runtime_error("Unable to open config file: " + filename + "\n");
    }

    // read lines
    while (std::getline(file, line)) {
      std::istringstream iss(line);
      std::string key, value;

      if (std::getline(iss, key, '=') && std::getline(iss, value)) {
        //remove whitespace
        key = trim(key);
        value = trim(value);
        config[key] = value;
      }
    }
    file.close();
  }

  // remove whitespace
  std::string trim(const std::string& str) {
    const char* whitespace = " \t\n\r";
    size_t start = str.find_first_not_of(whitespace);
    size_t end = str.find_last_not_of(whitespace);

    if (start == std::string::npos) {
      return "";
    } else {
      return str.substr(start, end - start + 1);
    }
  }
};

