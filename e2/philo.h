#ifndef HSNR_PAC_E2_PHILO_H
#define HSNR_PAC_E2_PHILO_H

#include <cmath>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <random>
#include <iostream>
#include <thread>
#include <sstream>

using namespace std;

class Table;

class Philo {
public:
    int _idx;

    explicit Philo(int idx) : _idx(idx) {

    }

    [[noreturn]] void actOn(Table &t);

    void eat() const {
        std::stringstream ss;
        ss << _idx << " is eating" << endl;
        cout << ss.str();
        auto const duration = std::chrono::seconds(rndIntMs());
        std::this_thread::sleep_for(duration);
    }

    void think() const {
        std::stringstream ss;
        ss << _idx << " is thinking" << endl;
        cout << ss.str();
        auto const duration = std::chrono::seconds(rndIntMs());
        std::this_thread::sleep_for(duration);
    }

    static int rndIntMs() {
        std::random_device rdev;
        std::mt19937 rgen(rdev());
        std::uniform_int_distribution<int> idist(1, 1);
        return idist(rgen);
    }

    std::chrono::system_clock::time_point patientUntil();
};

class Table {
    unsigned int _n;
    mutex *_mutex;
    condition_variable *_conds;
    bool *_forks;

public:
    explicit Table(unsigned int n) : _n(n) {
        _mutex = new mutex[_n];
        _conds = new condition_variable[_n];
        _forks = new bool[_n];
        std::fill(_forks, _forks + _n, true);
    }

public:
    void tryEat(Philo &p);
};

#endif //HSNR_PAC_E2_PHILO_H
