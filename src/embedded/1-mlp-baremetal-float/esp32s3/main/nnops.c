#include <nnops.h>

// multiply-and-accumulate (dot product) between two vectors
void mac(const float *x, const float *y, float *out, int size) {
    int i;
    float acc = 0.0;

    for (i = 0; i < size; i++)
        acc += (x[i] * y[i]);

    *out = acc;
}

// matrix-vector multiplication
void mvm(const float *M, const float *v, float *out, int nrows, int ncols) {
    int row;

    for (row = 0; row < nrows; row++)
        mac(&M[row*ncols], v, &out[row], ncols);
}

// computes the relu activation function over a vector
void relu(const float *x, float *out, int size) {
    int i;
    for (i = 0; i < size; i++)
        out[i] = x[i]>0.0 ? x[i] : 0.0;
}

// returns index of max value
int argmax(const float *x, int size) {
    int i, i_max = 0;

    for (i = 1; i < size; i++) {
        if (x[i] > x[i_max]) {
            i_max = i;
        }
    }

    return i_max;
}
