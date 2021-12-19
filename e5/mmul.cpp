#include "mmul.h"

#include <random>
#include <stdexcept>
#include "omp.h"
#include <iostream>
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

inline void checkSuccess(const int &flag, const char *msg) {
    if (flag != MPI_SUCCESS) {
        throw std::domain_error(msg);
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
            return mulByCol(vec);
        }
        case BLK: {
            return mulByBlk(vec);
        }
    }
    cout << "rank " << MPI_rank() << ": mul done" << endl;
}

Vec Mat::mulByRow(const Vec &vec) const {
    cout << "rank " << MPI_rank() << ": strategy row" << endl;
    int flag;

    if (MPI_rank_zero()) {
        MPI_Datatype row_type;
        MPI_Type_vector(3, 1, 3, MPI_DOUBLE, &row_type);
        MPI_Type_commit(&row_type);

        MPI_Request request;
        MPI_Send(&_v, 1, row_type, RECEIVER, 0, MPI_COMM_WORLD);
    } else {
        double vec[_cols];
        double row[_cols];
        flag = MPI_Recv(&row, 3, MPI_DOUBLE, SENDER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (flag != MPI_SUCCESS) {
            throw std::invalid_argument("mpi: receive values");
        }

    }

    // send computed results 
}

Vec Mat::mulByCol(const Vec &vec) const {
    cout << "rank " << MPI_rank() << ": strategy col" << endl;

    MPI_Datatype col_type;
    MPI_Type_vector(3, 1, 3, MPI_INT, &col_type);
    MPI_Type_commit(&col_type);

}

Vec Mat::mulByBlk(const Vec &vec) const {
    cout << "rank " << MPI_rank() << ": strategy blk" << endl;

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
    int flag = 0;

    flag = MPI_Send(&_v, _size, MPI_DOUBLE, RECEIVER, 0, MPI_COMM_WORLD);
    checkSuccess(flag, "mpi: send values");
}

void Mat::recvSync() {
    cout << "rank " << MPI_rank() << ": recv mat size: " << _rows << "x" << _cols << endl;
    int flag = 0;

    flag = MPI_Recv(&_v, _size, MPI_DOUBLE, SENDER, 0, MPI_COMM_WORLD);
    checkSuccess(flag, "mpi: recv values");
}
