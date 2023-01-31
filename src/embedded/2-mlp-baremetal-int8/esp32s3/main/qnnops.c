#include <math.h>
#include <qnnops.h>
#include <stdint.h>

// multiply-and-accumulate (dot product) between two vectors
void mac(const int8_t *x, const int8_t *y, int8_t x_zero, int8_t y_zero, int32_t *out, int size) {
    int32_t acc = 0;

    for (int i = 0; i < size; i++)
        acc += (int32_t)((x[i] - x_zero) * (y[i] - y_zero));

    *out = acc;
}

// matrix-vector multiplication
void mvm(const struct qlayer *qlayer, const int8_t *M, const int8_t *v, int32_t *out, int nrows, int ncols) {
    for (int row = 0; row < nrows; row++)
        mac(&M[row*ncols], v, qlayer->weights.zero, qlayer->input.zero, &out[row], ncols);
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

// quantizes vector from float to int8
void quantize(const float *x, struct qparams* qparams, int8_t *out, int size) {
    int i;

    for (i=0; i<size; i++) {
        out[i] = (int8_t)(round(x[i]/qparams->scale) + qparams->zero);
    }
}

void dequantize(const struct qlayer *qlayer, const int32_t *x, float *out, int size) {
    float mvm_scale;

    mvm_scale = (qlayer->input.scale * qlayer->weights.scale);
    for (int i = 0; i < size; i++) {
        float temp = x[i] * mvm_scale;
        out[i] = (temp - qlayer->output.zero) * qlayer->output.scale;
    }
}
