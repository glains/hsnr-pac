
#include <cmath>
#include <random>
#include <chrono>
#include <iostream>

#define MAX_2EXP 12
#define BATCH_SIZE 5

#ifndef DUT_T
#define DUR_T duration<long long int, std::nano>
#endif

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::nanoseconds;

class VTriad {
private:
    const int TRIAD = 3;

    const int _n;
    const int _size;
    int *_v;

public:
    explicit VTriad(int n) : _n(n), _size(n * TRIAD) {
        _v = new int[_size];
    }

    void fillRnd() {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> dist(1, 1000);
        for (int i = 0; i < _size; ++i) {
            _v[i] = (int) dist(rng);
        }
    }

    void triad(int *arr) {
        for (int i = 0; i < _n; ++i) {
            arr[i] = _v[i] + _v[i * 2] * _v[i * 3];
        }
    }

    virtual ~VTriad() {
        delete[] _v;
    }
};

DUR_T vtriad_flps(int n) {
    int a[n];

    DUR_T total = high_resolution_clock::duration::zero();
    for (int i = 0; i < BATCH_SIZE; ++i) {
        VTriad t(n);
        t.fillRnd();

        auto t1 = high_resolution_clock::now();

        t.triad(a);

        auto t2 = high_resolution_clock::now();
        total += t2 - t1;
    }

    return total / BATCH_SIZE;
}

int main() {
    for (int n = 2; n < MAX_2EXP; ++n) {
        int mat_size = (int) pow(2, n);
        DUR_T d = vtriad_flps(n);

        duration<double, std::ratio<1>> sec_d = d;
        double sec = sec_d.count();

        std::cout << mat_size << ": avg s/run: " << sec << std::endl;
        if (sec == 0) {
            std::cout << mat_size << ": nan, " << sec << " nanos" << std::endl;
            continue;
        }
        int flops = (int) (2.0 * mat_size / sec);
        double mflops = flops * pow(10, -6);
        std::cout << mat_size << ": " << flops << " flops" << std::endl;
        std::cout << mat_size << ": " << mflops << " mflops" << std::endl;
    }
    return 0;
}