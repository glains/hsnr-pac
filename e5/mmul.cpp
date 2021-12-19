#include "mmul.h"

#include <random>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <mpi.h>

using namespace std;

//-----------------------------------------------------------------------
// MPI

inline int MPI_rank() {
    return MPI::COMM_WORLD.Get_rank();
}

inline int MPI_rank_zero() {
    return MPI::COMM_WORLD.Get_rank() == 0;
}

inline int MPI_size() {
    return MPI::COMM_WORLD.Get_size();
}

inline void checkSuccess(const int &flag, const char *msg) {
    if (flag != MPI_SUCCESS) {
        stringstream ss;
        ss << msg << "; rank: " << MPI_rank() << "; flag: " << flag << endl;
        throw std::domain_error(ss.str());
    }
}

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

void Vec::sendSync() {
    cout << "send vector size: " << _n << endl;
}

void Vec::recvSync() {
    cout << "recv vector size: " << _n << endl;
}

double *Vec::begin() const {
    return _v;
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
    Vec res(_cols);
    switch (s) {
        case ROW: {
            res = mulByRow(vec);
            break;
        }
        case COL: {
            res = mulByCol(vec);
            break;
        }
        case BLK: {
            res = mulByBlk(vec);
            break;
        }
    }
    cout << "rank " << MPI_rank() << ": mul done" << endl;
    return res;
}

Vec Mat::mulByRow(const Vec &vec) const {
    cout << "rank " << MPI_rank() << ": strategy row" << endl;
    int flag;
    int rank = MPI_rank();
    int size = MPI_size();

    Vec res(_cols);

    int blks = _rows / size;
    int from = rank * blks;
    int to = from + blks;
    for (int row = from; row < to; ++row) {
        for (int col = 0; col < _cols; ++col) {
            res.at(col) += at(row, col) * vec.at(col);
        }
    }

    double *loc_off = res.begin() + from;
    flag = MPI_Bcast(loc_off, _cols, MPI_DOUBLE, rank, MPI_COMM_WORLD);
    checkSuccess(flag, "mpi: bcast result");

    // before current
    for (int row = 0; row < from; ++row) {
        double *rem_off = res.begin() + (blks * row);
        flag = MPI_Bcast(rem_off, _cols, MPI_DOUBLE, rank, MPI_COMM_WORLD);
        checkSuccess(flag, "mpi: bcast fetch before");
    }
    // after current
    for (int row = to + 1; row < _rows; ++row) {
        double *rem_off = res.begin() + (blks * row);
        flag = MPI_Bcast(rem_off, _cols, MPI_DOUBLE, rank, MPI_COMM_WORLD);
        checkSuccess(flag, "mpi: bcast fetch after");
    }

    return res;
}

Vec Mat::mulByCol(const Vec &vec) const {
    cout << "rank " << MPI_rank() << ": strategy col" << endl;

    MPI_Datatype col_type;
    MPI_Type_vector(3, 1, 3, MPI_INT, &col_type);
    MPI_Type_commit(&col_type);

    return Vec(0);
}

Vec Mat::mulByBlk(const Vec &vec) const {
    cout << "rank " << MPI_rank() << ": strategy blk" << endl;

    MPI_Datatype row_type;
    MPI_Type_vector(1, _cols, _cols, MPI_DOUBLE, &row_type);
    MPI_Type_commit(&row_type);

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

void Mat::sendSync() {
    cout << "rank " << MPI_rank() << ": send mat size: " << _rows << "x" << _cols << endl;
    int flag = MPI_Bcast(_v, _size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    checkSuccess(flag, "mpi: send values");
}

void Mat::recvSync() {
    cout << "rank " << MPI_rank() << ": recv mat size: " << _rows << "x" << _cols << endl;
    int flag = MPI_Bcast(_v, _size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    checkSuccess(flag, "mpi: recv values");
}
