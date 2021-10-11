
#include <cmath>
#include <random>
#include <chrono>
#include <iostream>
#include <exception>

#define MAX_2EXP 12
#define BATCH_SIZE 5

#ifndef DUT_T
#define DUR_T duration<long long int, std::nano>
#endif

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

#define FAC_KB 64

void mat_mul_tile(const Mat &a, const Mat &b, Mat &c) {
    if ((a._n & (a._n - 1)) != 0) {
        throw std::logic_error("size must be power of two: " + std::to_string(a._n));
    }
    int ts = (int) sqrt(FAC_KB);
    int n = a._n; // assume n x n
    for (int row = 0; row < n; row += ts) {
        for (int col = 0; col < n; col += ts) {
            for (int t = 0; t < n; t += ts) {

                for (int i = row; i < std::min(row + ts, row); ++i) {
                    for (int j = col; j < std::min(col + ts, col); ++j) {
                        int sum = 0;
                        for (int k = t; k < std::min(t + ts, t); ++k) {
                            sum += a.at(i, k) * b.at(i, k);
                        }
                        c.at(i, j) += sum;
                    }
                }
            }
        }
    }
}

//-----------------------------------------------------------------------

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::seconds;
using std::chrono::nanoseconds;

DUR_T mmul_flps(int n) {
    Mat a(n);
    Mat b(n);
    Mat c(n);

    DUR_T total = high_resolution_clock::duration::zero();
    for (int i = 0; i < BATCH_SIZE; ++i) {
        a.randomize();
        b.randomize();

        auto t1 = high_resolution_clock::now();

        mat_mul_naive(a, b, c);
        //mat_mul_tile(a, b, c);

        auto t2 = high_resolution_clock::now();
        total += t2 - t1;
    }

    return total / BATCH_SIZE;
}

int main() {
    for (int n = 2; n < MAX_2EXP; ++n) {
        int mat_size = (int) pow(2, n);
        DUR_T d = mmul_flps(mat_size);

        duration<double, std::ratio<1>> sec_d = d;
        double sec = sec_d.count();

        std::cout << mat_size << ": avg s/run: " << sec << std::endl;
        if (sec == 0) {
            std::cout << mat_size << ": nan, " << sec << " nanos" << std::endl;
            continue;
        }
        double flops = 2.0 * pow(mat_size, 3) / sec;
        double mflops = flops * pow(10, -6);
        std::cout << mat_size << ": " << mflops << " mflops" << std::endl;
    }
    return 0;
}