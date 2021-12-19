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
    Vec v(n);
    Mat mat(m, n);

    if (rankZero()) {
        v.randomize();
        v.sendSync();
        mat.randomize();
        mat.sendSync();
    } else {
        v.recvSync();
        mat.recvSync();
    }

    MPI::COMM_WORLD.Barrier();
    Vec r = mat.mul(v, ROW);
}

const int PO2_FR = 8;
const int PO2_TO = 12;

int main(int argc, char **argv) {
    int flag;

    MPI::Init();
    // TODO: validate if all participants are present

    for (int i = PO2_FR; i < PO2_TO; ++i) {
        int s = (int) pow(2, i);    // matrix size nxn
        int m = s, n = s;           // matrix size mxn

        mmul(m, n);
    }

    MPI::Finalize();
}