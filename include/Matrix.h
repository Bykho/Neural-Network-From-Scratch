#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <random>
#include <stdexcept>

template<typename T>
class Matrix {
private:
    std::vector<std::vector<T>> data;
    size_t rows;
    size_t cols;

public:
    // Zero matrix
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, T value);
    Matrix(const Matrix<T>& other);
    
    Matrix<T>& operator=(const Matrix<T>& other);

    // Core operations between matrices. The backbone of any neural network implementation
    Matrix<T> operator+(const Matrix<T>& other) const;
    Matrix<T> operator-(const Matrix<T>& other) const;
    Matrix<T> operator*(const Matrix<T>& other) const;
    Matrix<T> operator*(const T scalar) const;
    
    // The hadamard product computes element-wise multiplication between matrices.
    // This is crucial for backpropagation when computing gradients
    Matrix<T> hadamard(const Matrix<T>& other) const;
    Matrix<T> apply(T (*func)(T)) const;
    
    // Neural net initialization using Xavier/Glorot. Random weights are drawn from 
    // a normal distribution with specified mean and standard deviation to help with
    // gradient flow during training
    void randomize(T mean = 0.0, T stddev = 1.0);
    Matrix<T> transpose() const;
    T sum() const;
    T mean() const;
    
    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }
    T& at(size_t i, size_t j);
    const T& at(size_t i, size_t j) const;
    void print() const;
    
    // Shape validation
    bool hasSameShape(const Matrix<T>& other) const;
    bool canMultiply(const Matrix<T>& other) const;

private:
    void validateIndices(size_t i, size_t j) const;
    void validateDimensions(const Matrix<T>& other, const std::string& operation) const;
};

template<typename T>
Matrix<T> operator*(const T scalar, const Matrix<T>& matrix);

#endif