#include "mmul.h"

#include <random>
#include <stdexcept>
#include <iostream>
#include <mpi.h>

using namespace std;


bool rankZero() {
    return MPI::COMM_WORLD.Get_rank() == 0;
}

void mmul(int m, int n) {
    if (rankZero()) {
        Mat mat(m, n);
        mat.randomize();
        mat.sendSync();

        Vec v(n);
        v.randomize();
        v.sendSync();

        Vec r = mat.mul(v, ROW);
    } else {
        Mat mat(m, n);
        mat.recvSync();

        Vec v(n);
        v.recvSync();

        Vec r = mat.mul(v, ROW);
    }
}

const int PO2_FR = 8;
const int PO2_TO = 12;

int main(int argc, char **argv) {
    int flag;

    flag = MPI::Init();
    if (flag != MPI::MPI_SUCCESS) {
        throw std::invalid_argument("mpi: cannot init");
    }
    // TODO: validate if all participants are present

    for (int i = PO2_FR; i < PO2_TO; ++i) {
        int s = (int) pow(2, i);    // matrix size nxn
        int m = s, n = s;           // matrix size mxn

        mmul(m, n);
    }

    MPI::Finalize();
}