#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <string>
#include <vector>
#include <fstream>

class MNISTLoader {
public:
    bool verifyFiles(const std::string& imagesPath, const std::string& labelsPath);
    std::vector<std::vector<double>> loadImages(const std::string& path);
    std::vector<uint8_t> loadLabels(const std::string& path);

private:
    int readInt(std::ifstream& file);
};

#endif