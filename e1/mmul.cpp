
#include <cmath>
#include <random>
#include <chrono>
#include <iostream>
#include <exception>
#include <functional>

#define MAX_2EXP 10
#define BATCH_SIZE 1

//-----------------------------------------------------------------------

class Mat {
public:
    const int _n;
    const int _size;
    double *_v;

    explicit Mat(int n) : _n(n), _size(n * n) {
        _v = new double[_size];
    }

    [[nodiscard]] double &at(int row, int col) const {
        return _v[row * _n + col];
    }

    void randomize() const {
        std::random_device rd;
        std::default_random_engine rng(rd());
        std::uniform_real_distribution<double> dist(1, 100);
        for (int i = 0; i < _size; ++i) {
            _v[i] = (int) dist(rng);
        }
    }

    virtual ~Mat() {
        delete[] _v;
    }
};

//-----------------------------------------------------------------------

void mat_mul_naive(const Mat &a, const Mat &b, Mat &c) {
    int n = a._n; // assume n x n
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            // dot-product
            for (int p = 0; p < n; p++) {
                c._v[row * n + col] += a._v[row * n + p] * b._v[p * n + col];
            }
        }
    }
}

void mat_mul_transpose(const Mat &a, const Mat &b, Mat &c) {
    int n = a._n; // assume n x n
    Mat tmp(n);
    for (int row = 0; row < n; ++row)
        for (int col = 0; col < n; ++col)
            tmp.at(row, col) = b.at(col, row);

    for (int row = 0; row < n; ++row)
        for (int col = 0; col < n; ++col)
            for (int k = 0; k < n; ++k)
                c.at(row, col) += a.at(row, k) * tmp.at(col, k);
}

// define the size of tile based on the size of a cache line
#ifndef CLS_KB
#define CLS_KB (64 * 1000)
#endif
#define FAC_KB (int)(CLS_KB / sizeof(double))

void mat_mul_tile(const Mat &a, const Mat &b, Mat &c) {
    if ((a._n & (a._n - 1)) != 0) {
        throw std::logic_error("size must be power of two: " + std::to_string(a._n));
    }
    int ts = FAC_KB;
    int n = a._n; // assume n x n
    for (int row = 0; row < n; row += ts)
        for (int col = 0; col < n; col += ts)
            for (int t = 0; t < n; t += ts)
                // tile multiplication
                for (int i = row; i < std::min(i + ts, n); ++i)
                    for (int j = col; j < std::min(j + ts, n); ++j)
                        for (int k = t; k < std::min(t + ts, n); ++k)
                            c.at(i, j) += a.at(i, k) * b.at(i, k);
}

//-----------------------------------------------------------------------

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::seconds;
using std::chrono::nanoseconds;

typedef duration<int64_t, std::milli> duration_t;

template<typename MulOp>
duration_t mmul_flps(int n, MulOp mulOp) {
    Mat a(n);
    Mat b(n);
    Mat c(n);

    auto elapsed = duration_t(0);
    for (int i = 0; i < BATCH_SIZE; ++i) {
        a.randomize();
        b.randomize();

        auto t1 = high_resolution_clock::now();
        mulOp(a, b, c);
        auto t2 = high_resolution_clock::now();
        elapsed += duration_cast<std::chrono::milliseconds>(t2 - t1);
    }
    return elapsed / BATCH_SIZE;
}

template<typename MulOp>
void mmul_exec(const std::string &name, MulOp mulOp) {
    std::cout << "benchmarking " << name << std::endl;

    for (int n = 6; n < MAX_2EXP; ++n) {
        int mat_size = (int) pow(2, n);
        duration_t elapsed = mmul_flps(mat_size, mulOp);

        duration<double, std::ratio<1>> selapsed = elapsed;
        double sec = selapsed.count();

        if (sec == 0) {
            std::cout << mat_size << ": nan" << std::endl;
            continue;
        }

        // std::cout << mat_size << ": avg s/run: " << sec << std::endl;
        double flops = (2 * pow(mat_size, 3)) / sec;
        // std::cout << mat_size << ": " << flops << " flops" << std::endl;
        // std::cout << mat_size << ": " << flops * pow(10, -6) << " mflops" << std::endl;
        std::cout << mat_size << ": " << flops / pow(10, 9) << " gflops" << std::endl;
    }

}

int main() {
    mmul_exec("naive", mat_mul_naive);
    mmul_exec("transpose", mat_mul_transpose);
    mmul_exec("tile", mat_mul_tile);
    return 0;
}