#include "philo.h"
#include <cmath>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <random>
#include <thread>

using namespace std;

[[noreturn]] void Philo::actOn(Table &t) {
    while (true) {
        think();
        t.tryEat(*this);
    }
}

std::chrono::system_clock::time_point Philo::patientUntil() {
    auto now = std::chrono::system_clock::now();
    return now + std::chrono::milliseconds();
}

void Table::tryEat(Philo &p) {
    if (_n == 1) {
        throw std::logic_error("not enough forks");
    }
    auto l = p._idx;
    unique_lock<std::mutex> llck(_mutex[l]);
    _conds[l].wait(llck, [this, l] { return _forks[l]; });
    _forks[l] = false;

    auto r = (p._idx + 1) % _n;
    unique_lock<std::mutex> rlck(_mutex[r]);
    if (_conds[r].wait_until(rlck, p.patientUntil(), [this, r] { return _forks[r]; })) {
        _forks[r] = false;
        p.eat(); // success
        _forks[r] = true;
        rlck.unlock();
        _conds[r].notify_one(); // only one neighbour
    }

    _forks[l] = true;
    llck.unlock();
    _conds[l].notify_one();
}

int main() {
    const int size = 10;
    vector<Philo> philos;
    for (int i = 0; i < size; ++i) {
        philos.emplace_back(Philo(i));
    }

    Table table(size);

    vector<thread> threads;
    for (auto &p: philos) {
        auto t = thread([p, table]() mutable { p.actOn(table); });
        threads.push_back(std::move(t));
    }

    for (auto &t: threads) {
        if (t.joinable()) t.join();
    }

    return 0;
}
