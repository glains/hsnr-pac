
#include <omp.h>
#include <random>
#include <iostream>
#include <cstdlib>
#include <chrono>

using namespace std;


double pi_omp_reduce(int samples,
                     [[maybe_unused]] int threads) {
    int hits;

#pragma omp parallel shared(samples) \
num_threads(threads) reduction(+:hits)
    {
        int local_samples = samples / omp_get_num_threads();

        std::random_device rng;
        std::mt19937 gen(rng());
        std::uniform_real_distribution<> dis(-1, 1);

        for (int i = 0; i < local_samples; ++i) {
            double rand_x = dis(gen);
            double rand_y = dis(gen);
            if (rand_x * rand_x + rand_y * rand_y < 1.0) hits++;
        }
    }

    return ((double) hits * 4) / samples;
}

double pi_omp_critical(int samples,
                       [[maybe_unused]] int threads) {
    int hits;

#pragma omp parallel shared(samples, hits) \
num_threads(threads)
    {
        int local_samples = samples / omp_get_num_threads();

        std::random_device rng;
        std::mt19937 gen(rng());
        std::uniform_real_distribution<> dis(-1, 1);

        for (int i = 0; i < local_samples; ++i) {
            double rand_x = dis(gen);
            double rand_y = dis(gen);
            if (rand_x * rand_x + rand_y * rand_y < 1.0) {
#pragma omp critical(hits)
                { hits++; }
            }
        }
    }

    return ((double) hits * 4) / samples;
}

double pi_omp_lock(int samples,
                   [[maybe_unused]] int threads) {
    int hits;

    omp_lock_t lock;
    omp_init_lock(&lock);

#pragma omp parallel shared(samples) \
num_threads(threads)
    {
        int local_samples = samples / omp_get_num_threads();

        std::random_device rng;
        std::mt19937 gen(rng());
        std::uniform_real_distribution<> dis(-1, 1);

        for (int i = 0; i < local_samples; ++i) {
            double rand_x = dis(gen);
            double rand_y = dis(gen);
            if (rand_x * rand_x + rand_y * rand_y < 1.0) {
                omp_set_lock(&lock);
                hits++;
                omp_unset_lock(&lock);
            }
        }
    }
    omp_destroy_lock(&lock);

    return ((double) hits * 4) / samples;
}

double pi_omp_atomic(int samples,
                     [[maybe_unused]] int threads) {
    int hits;

#pragma omp parallel shared(samples) \
num_threads(threads)
    {
        int local_samples = samples / omp_get_num_threads();

        std::random_device rng;
        std::mt19937 gen(rng());
        std::uniform_real_distribution<> dis(-1, 1);

        for (int i = 0; i < local_samples; ++i) {
            double rand_x = dis(gen);
            double rand_y = dis(gen);
            if (rand_x * rand_x + rand_y * rand_y < 1.0) {
#pragma omp atomic
                hits++;
            }
        }
    }

    return ((double) hits * 4) / samples;
}

//-----------------------------------------------------------------------

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::seconds;
using std::chrono::milliseconds;

typedef std::chrono::duration<int64_t, std::milli> duration_t;

template<typename PiRng>
double benchmark(PiRng tmpl, int samples, int threads) {
    auto elapsed = duration_t(0);
    auto t1 = high_resolution_clock::now();
    double pi = tmpl(samples, threads);
    auto t2 = high_resolution_clock::now();
    elapsed += duration_cast<milliseconds>(t2 - t1);
    cout << "elapsed : " << elapsed.count() << endl;
    return pi;
}

void benchmark(int n, int t) {
    cout << "dac     : " << benchmark(pi_omp_reduce, n, t) << endl;
    cout << "critical: " << benchmark(pi_omp_critical, n, t) << endl;
    cout << "lock    : " << benchmark(pi_omp_lock, n, t) << endl;
    cout << "atomic  : " << benchmark(pi_omp_atomic, n, t) << endl;
}

int main(int argc, char *argv[]) {
    int samples;
    if (argc > 1) {
        char *end;
        samples = strtol(argv[1], &end, 10);
    } else {
        samples = 1000;
    }
    cout << "using " << samples << " samples" << endl;

    benchmark(samples, 1); // baseline
    for (int threads = 4; threads <= 32; threads *= 2) {
        cout << "threads : " << threads << endl;
        benchmark(samples, threads);
    }
}
