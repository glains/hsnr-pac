#include "mmul.h"

#include <random>
#include <stdexcept>
#include "omp.h"
#include <mpi.h>

using namespace std;

//-----------------------------------------------------------------------
// Vec

inline double &Vec::at(int idx) const {
    return _v[idx];
}

void Vec::randomize() {
    std::random_device rd;
    std::default_random_engine rng(rd());
    std::uniform_real_distribution<double> dist(1, 100);
    for (int i = 0; i < _n; ++i) {
        _v[i] = dist(rng);
    }
}

//-----------------------------------------------------------------------
// Mat

inline double &Mat::at(int row, int col) const {
    return _v[row * _rows + col];
}

Vec Mat::mul(const Vec &vec, Strategy s) const {
    if (vec.size() != _cols) {
        throw std::invalid_argument("result is undefined");
    }
    switch (s) {
        case ROW: {
            return mulByRow(vec);
        }
        case COL: {

        }
        case BLK: {

        }
    }
}

Vec Mat::mulByRow(const Vec &vec) const {
#pragma omp parallel num_threads(4)
    {
        int flag;

        flag = MPI::Init();

        MPI::Finalize();
    }
}

Vec Mat::mulByCol(const Vec &vec) const {
    return Vec(0);
}

Vec Mat::mulByBlk(const Vec &vec) const {
    return Vec(0);
}

void Mat::randomize() {
    std::random_device rd;
    std::default_random_engine rng(rd());
    std::uniform_real_distribution<double> dist(1, 100);
    for (int i = 0; i < _size; ++i) {
        _v[i] = dist(rng);
    }
}
