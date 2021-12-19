#include "mmul.h"

#include <random>
#include <mpi.h>

using namespace std;

const int PO2_FR = 4;
const int PO2_TO = 5;

#define ROOT 0

//-----------------------------------------------------------------------

bool isRoot() {
    return MPI::COMM_WORLD.Get_rank() == ROOT;
}

void mmul(int m, int n, Strategy s) {
    Vec v(n);
    Mat mat(m, n);

    if (isRoot()) {
        v.randomize();
        mat.randomize();
    }
    v.bcast(ROOT);
    mat.bcast(ROOT);

    MPI::COMM_WORLD.Barrier();
    Vec r = mat.mul(v, s);
}

void mmul(Strategy strat) {
    for (int i = PO2_FR; i < PO2_TO; ++i) {
        int s = (int) pow(2, i);    // matrix size nxn
        int m = s, n = s;           // matrix size mxn

        mmul(m, n, strat);
    }
}

//-----------------------------------------------------------------------

int main([[maybe_unused]] int argc,
         [[maybe_unused]] char **argv) {

    MPI::Init();
    // TODO: validate if all participants are present

    mmul(ROW);
    mmul(COL);
    mmul(BLK);

    MPI::Finalize();
}