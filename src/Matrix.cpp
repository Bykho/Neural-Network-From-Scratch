#include "Matrix.h"
#include <iostream>
#include <iomanip>

template<typename T>
Matrix<T>::Matrix(size_t rows, size_t cols) {
    this->rows = rows;
    this->cols = cols;
    data.resize(rows, std::vector<T>(cols, T()));
}

// This one fills with a value instead of default constructing
template<typename T>
Matrix<T>::Matrix(size_t rows, size_t cols, T value) {
    this->rows = rows;
    this->cols = cols;
    data.resize(rows, std::vector<T>(cols, value));
}

template<typename T>
Matrix<T>::Matrix(const Matrix<T>& other) {
    rows = other.rows;
    cols = other.cols;
    
    // Gotta do deep copy or we'll have pointer issues
    data.resize(rows);
    for (size_t i = 0; i < rows; i++) {
        data[i].resize(cols);
        for (size_t j = 0; j < cols; j++) {
            data[i][j] = other.data[i][j];
        }
    }
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& other) {
    if (this != &other) {  // self-assignment check
        data.clear();
        rows = other.rows;
        cols = other.cols;
        
        data.resize(rows);
        for (size_t i = 0; i < rows; i++) {
            data[i].resize(cols);
            for (size_t j = 0; j < cols; j++) {
                data[i][j] = other.data[i][j];
            }
        }
    }
    return *this;
}

template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const {
    if (this == &other) {
        return *this;
    }
    if (other.rows != this->rows || other.cols != this->cols) {
        throw std::invalid_argument("Matrices must have the same dimensions for addition.");
    }
    Matrix<T> NewMatrix(other.rows, other.cols);
    // Simple element-wise addition loop
    for (size_t r = 0; r < other.rows; r++) {
        for (size_t c = 0; c < other.cols; c++) {
            NewMatrix.data[r][c] = other.data[r][c] + this->data[r][c]; 
        }
    }
    return NewMatrix;
}

template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& other) const {
    if (this == &other) {
        return *this;
    }
    if (other.rows != this->rows || other.cols != this->cols) {
        throw std::invalid_argument("Matrices must have the same dimensions for subtraction.");
    }
    Matrix<T> NewMatrix(other.rows, other.cols);
    for (size_t r = 0; r < other.rows; r++) {
        for (size_t c = 0; c < other.cols; c++) {
            NewMatrix.data[r][c] = this->data[r][c] - other.data[r][c];
        }
    }
    return NewMatrix;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const {
    // This is the tricky one - matrices can only multiply if dimensions match correctly
    if (this->cols != other.rows) {
        throw std::invalid_argument(
            "Matrix multiplication dimension mismatch: (" + 
            std::to_string(this->rows) + "x" + std::to_string(this->cols) + ") * (" +
            std::to_string(other.rows) + "x" + std::to_string(other.cols) + ")"
        );
    }

    Matrix<T> result(this->rows, other.cols, T());
    
    // Triple nested loop. Not very efficient but gets the job done
    for (size_t i = 0; i < this->rows; i++) {
        for (size_t j = 0; j < other.cols; j++) {
            for (size_t k = 0; k < this->cols; k++) {
                result.data[i][j] += this->data[i][k] * other.data[k][j];
            }
        }
    }
    
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const T scalar) const {
    Matrix<T> NewMatrix(this->rows, this->cols);
    for (size_t r = 0; r < this->rows; r++) {
        for (size_t c = 0; c < this->cols; c++) {
            NewMatrix.data[r][c] = this->data[r][c] * scalar; 
        }
    }
    return NewMatrix;
}

template<typename T>
Matrix<T> Matrix<T>::hadamard(const Matrix<T>& other) const {
    // Note to self: this is element-wise multiplication, different from matrix mult
    if (this == &other) {
        return *this;
    }
    if (other.rows != this->rows || other.cols != this->cols) {
        throw std::invalid_argument("Matrices must have the same dimensions for the Hadamard product.");
    }
    Matrix<T> NewMatrix(other.rows, other.cols);
    for (size_t r = 0; r < other.rows; r++) {
        for (size_t c = 0; c < other.cols; c++) {
            NewMatrix.data[r][c] = other.data[r][c] * this->data[r][c]; 
        }
    }
    return NewMatrix;
}

template<typename T>
Matrix<T> Matrix<T>::apply(T (*func)(T)) const {
    if (this->rows == 0 || this->cols == 0) {
        throw std::invalid_argument("Matrices cannot have 0 volume for the apply function.");
    }
    Matrix<T> newMatrix(this->rows, this->cols);
    // Run the function on each element - useful for activation functions later
    for (size_t r = 0; r < this->rows; r++) {
        for (size_t c = 0; c < this->cols; c++) {
            newMatrix.data[r][c] = func(this->data[r][c]);
        }
    }
    return newMatrix;
}

template<typename T>
void Matrix<T>::randomize(T mean, T stddev) {
    // Using normal distribution for weight initialization
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::normal_distribution<T> distribution(mean, stddev);

    for (size_t r = 0; r < this->rows; r++) {
        for (size_t c = 0; c < this->cols; c++) {
            this->data[r][c] = distribution(generator);
        }
    }
}

template<typename T>
Matrix<T> Matrix<T>::transpose() const {
    if (this->rows == 0 || this->cols == 0) {
        throw std::invalid_argument("Matrices cannot have 0 volume for the transpose method.");
    }

    // Flip rows and cols - needed this for backprop
    Matrix<T> newMatrix(this->cols, this->rows);
    for (size_t r = 0; r < this->rows; r++) {
        for (size_t c = 0; c < this->cols; c++) {
            newMatrix.data[c][r] = this->data[r][c];
        }
    }
    return newMatrix;
}

template<typename T>
T Matrix<T>::sum() const {
    if (this->rows == 0 || this->cols == 0) {
        throw std::invalid_argument("Matrices cannot have 0 volume for the sum method.");
    }
    T result = T();
    for (size_t r = 0; r < this->rows; r++) {
        for (size_t c = 0; c < this->cols; c++) {
            result += this->data[r][c];
        }
    }
    return result;
}

template<typename T>
T Matrix<T>::mean() const {
    if (this->rows == 0 || this->cols == 0) {
        throw std::invalid_argument("Matrices cannot have 0 volume for the mean method.");
    }
    T result = T();
    for (size_t r = 0; r < this->rows; r++) {
        for (size_t c = 0; c < this->cols; c++) {
            result += this->data[r][c];
        }
    }
    return result / static_cast<T>(this->rows * this->cols);
}

template<typename T>
T& Matrix<T>::at(size_t i, size_t j) {
    validateIndices(i, j);
    return data[i][j];
}

template<typename T>
const T& Matrix<T>::at(size_t i, size_t j) const {
    validateIndices(i, j);
    return data[i][j];
}

template<typename T>
void Matrix<T>::print() const {
    // Pretty print the matrix with fixed width columns
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            std::cout << std::setw(10) << std::setprecision(4) << data[r][c] << " ";
        }
        std::cout << std::endl;
    }
}

template<typename T>
bool Matrix<T>::hasSameShape(const Matrix<T>& other) const {
    return (this->rows == other.rows && this->cols == other.cols);
}

template<typename T>
bool Matrix<T>::canMultiply(const Matrix<T>& other) const {
    return (this->cols == other.rows);
}

template<typename T>
void Matrix<T>::validateIndices(size_t i, size_t j) const {
    if (i >= rows || j >= cols) {
        throw std::out_of_range("Matrix indices out of bounds");
    }
}

template<typename T>
void Matrix<T>::validateDimensions(const Matrix<T>& other, const std::string& operation) const {
    // Checks if matrices can be used together based on operation type
    if (operation == "addition" || operation == "subtraction" || operation == "hadamard") {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions must match for " + operation);
        }
    } else if (operation == "multiplication") {
        if (cols != other.rows) {
            throw std::invalid_argument("First matrix columns must match second matrix rows for multiplication");
        }
    }
}

template<typename T>
Matrix<T> operator*(const T scalar, const Matrix<T>& matrix) {
    return matrix * scalar;
}

// Only allowing float and double matrices for now
template class Matrix<float>;
template class Matrix<double>;