#include "MNISTLoader.h"
#include <iostream>

int MNISTLoader::readInt(std::ifstream& file) {
    unsigned char bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    return ((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3]);
}

bool MNISTLoader::verifyFiles(const std::string& imagesPath, const std::string& labelsPath) {
    std::ifstream imageFile(imagesPath, std::ios::binary);
    std::ifstream labelFile(labelsPath, std::ios::binary);
    
    if (!imageFile || !labelFile) {
        std::cerr << "Could not open files" << std::endl;
        return false;
    }
    
    int imageMagic = readInt(imageFile);
    int numImages = readInt(imageFile);
    int numRows = readInt(imageFile);
    int numCols = readInt(imageFile);
    
    int labelMagic = readInt(labelFile);
    int numLabels = readInt(labelFile);
    
    return (imageMagic == 0x803 && labelMagic == 0x801 && numImages == numLabels);
}

std::vector<std::vector<double>> MNISTLoader::loadImages(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open image file: " + path);
    }

    // Read header
    int magic = readInt(file);
    int numImages = readInt(file);
    int numRows = readInt(file);
    int numCols = readInt(file);

    if (magic != 0x803) {
        throw std::runtime_error("Invalid magic number in image file");
    }

    // Read image data
    std::vector<std::vector<double>> images;
    images.reserve(numImages);

    for (int i = 0; i < numImages; ++i) {
        std::vector<double> image;
        image.reserve(numRows * numCols);

        for (int j = 0; j < numRows * numCols; ++j) {
            unsigned char pixel;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            image.push_back(static_cast<double>(pixel));
        }
        images.push_back(image);
    }

    return images;
}

std::vector<uint8_t> MNISTLoader::loadLabels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open label file: " + path);
    }

    // Read header
    int magic = readInt(file);
    int numLabels = readInt(file);

    if (magic != 0x801) {
        throw std::runtime_error("Invalid magic number in label file");
    }

    // Read labels
    std::vector<uint8_t> labels;
    labels.reserve(numLabels);

    for (int i = 0; i < numLabels; ++i) {
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), 1);
        labels.push_back(label);
    }

    return labels;
}