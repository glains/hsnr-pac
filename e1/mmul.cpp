
#include <cmath>
#include <random>
#include <chrono>
#include <iostream>

// input size to iterate through
#define N_FR 100
#define N_TO 1000
#define N_STEP 100

#define BATCH_SIZE 1000

#ifndef DUT_T
#define DUR_T duration<int, std::nano>
#endif

//-----------------------------------------------------------------------

class Mat {
public:
    const int _n;
    const int _size;
    int *_v;

    explicit Mat(int n) : _n(n), _size(n * n) {
        _v = new int[_size];
    }

    void randomize() const {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> dist(1, 1000);
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

//-----------------------------------------------------------------------

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::nanoseconds;

DUR_T mmul_flps(int n) {
    Mat a(n);
    Mat b(n);
    Mat c(n);

    DUR_T total{};
    for (int i = 0; i < BATCH_SIZE; ++i) {
        a.randomize();
        b.randomize();

        auto t1 = high_resolution_clock::now();

        mat_mul_naive(a, b, c);

        auto t2 = high_resolution_clock::now();

        //auto ms_int = duration_cast<nanoseconds>(t2 - t1);
        total += t2 - t1;
    }

    return total / BATCH_SIZE;
}

int main() {
    for (int n = N_FR; n < N_TO; n += N_STEP) {
        DUR_T d = mmul_flps(n);

        int c = d.count();
        double seconds = c / pow(10, 9);
        std::cout << n << ": average " << seconds << " s/run" << std::endl;
        if (seconds == 0) {
            std::cout << n << ": nan, " << c << " nanos" << std::endl;
            continue;
        }
        double flops = (2 * n / seconds) * pow(10, -6);
        std::cout << n << ": " << (int) flops << " mflops" << std::endl;
    }
    return 0;
}