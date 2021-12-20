#include "mmul.h"

#include <random>
#include <mpi.h>
#include <iomanip>

using namespace std;

const int PO2_FR = 10;
const int PO2_TO = 14;

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

    double t1 = MPI_Wtime();
    MPI::COMM_WORLD.Barrier();
    Vec r = mat.mul(v, s);
    MPI::COMM_WORLD.Barrier();
    double t2 = MPI_Wtime();

    double delta = t2 - t1;
    if (isRoot()) {
        cout << std::setprecision(0) << "rank 0: took " << delta << " sec." << endl << endl;
    }
}

void mmul(Strategy strat) {
    cout << "using strategy: " << strat << endl << endl;
    for (int i = PO2_FR; i < PO2_TO; ++i) {
        int s = (int) pow(2, i);
        int m = s, n = s; // mat size nxn

        mmul(m, n, strat);
    }
}

//-----------------------------------------------------------------------

int main([[maybe_unused]] int argc,
         [[maybe_unused]] char **argv) {

    MPI::Init();

    mmul(LOCAL);
    mmul(MPI_ROW);
    mmul(MPI_COL);
    mmul(MPI_BLK);

    MPI::Finalize();
}