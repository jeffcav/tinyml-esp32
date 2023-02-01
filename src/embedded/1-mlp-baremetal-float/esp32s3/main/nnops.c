#include <nnops.h>

float mac(const float *x, const float *y, int size) {
    float acc = 0.0;

    for (int i = 0; i < size; i++)
        acc += (x[i] * y[i]);

    return acc;
}

void mvm(const float *M, const float *v, float *out, int nrows, int ncols) {
    int row;

    for (row = 0; row < nrows; row++)
        out[row] = mac(&M[row*ncols], v, ncols);
}

void relu(const float *x, float *out, int size) {
    int i;
    for (i = 0; i < size; i++)
        out[i] = x[i]>0.0 ? x[i] : 0.0;
}

int argmax(const float *x, int size) {
    int i, i_max = 0;

    for (i = 1; i < size; i++) {
        if (x[i] > x[i_max]) {
            i_max = i;
        }
    }

    return i_max;
}
