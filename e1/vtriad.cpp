
#include <cmath>
#include <random>
#include <chrono>
#include <iostream>

// should be at least 1000
#define N_FR 10000
#define N_TO 200000
#define N_STEP 10000

#define BATCH_SIZE 1000

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

int main_2() {
    for (int n = N_FR; n < N_TO; n += N_STEP) {
        DUR_T d = vtriad_flps(n);

        int c = d.count();
        double seconds = c / pow(10, 9);
        if (seconds == 0) {
            std::cout << n << ": nan, " << c << " nanos" << std::endl;
            continue;
        }
        double flops = (2 * n / seconds) * pow(10, -6);
        std::cout << n << ": " << (int) flops << " mflops" << std::endl;
    }
    return 0;
}