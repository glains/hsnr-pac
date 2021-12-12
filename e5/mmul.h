#ifndef HSNR_PAC_E5_MMUL_H
#define HSNR_PAC_E5_MMUL_H

class Vec {
public:
    explicit Vec(int n) : _n(n) {
        _v = new double[_n];
    }

    [[nodiscard]]
    int size() const {
        return _n;
    }

    [[nodiscard]]
    double &at(int idx) const;

    void randomize();

    virtual ~Vec() {
        delete[] _v;
    }

private:
    const int _n;
    double *_v;
};

enum Strategy {
    ROW,
    COL,
    BLK
};

class Mat {
public:
    Mat(int rows, int cols) :
            _rows(rows), _cols(cols), _size(rows * cols) {
        _v = new double[_size];
    }

    [[nodiscard]]
    double &at(int row, int col) const;

    [[nodiscard]]
    Vec mul(const Vec &vec, Strategy s) const;

    void randomize();

    virtual ~Mat() {
        delete[] _v;
    }

private:
    const int _rows;
    const int _cols;
    const int _size;
    double *_v;

    [[nodiscard]]
    Vec mulByRow(const Vec &vec) const;

    [[nodiscard]]
    Vec mulByCol(const Vec &vec) const;

    [[nodiscard]]
    Vec mulByBlk(const Vec &vec) const;
};

#endif //HSNR_PAC_E5_MMUL_H
