#include <math.h>
#include <qnnops.h>
#include <stdint.h>

int32_t mac(const int8_t *x, const int8_t *y, int size) {
    int32_t acc = 0;

    for (int i = 0; i < size; i++)
        acc += (int32_t)((int16_t)x[i] * (int16_t)y[i]);

    return acc;
}

void mvm(const int8_t *M, int8_t *v, int32_t *out, int8_t v_zero, int nrows, int ncols) {
    for (int i = 0; i < ncols; i++)
        v[i] = (v[i] - v_zero);

    for (int row = 0; row < nrows; row++)
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

void quantize(const float *x, float scale, int8_t zero, int8_t *out, int size) {
    for (int i = 0; i < size; i++)
        out[i] = (int8_t)(round(x[i] / scale) + zero);
}

void dequantize(const struct qlayer *qlayer, const int32_t *x, float *out, int size) {
    float mvm_scale;

    // TODO pre-calculate this value during compile-time
    mvm_scale = (qlayer->input.scale * qlayer->weights.scale);

    for (int i = 0; i < size; i++) {
        float temp = x[i] * mvm_scale;
        out[i] = (temp - qlayer->output.zero) * qlayer->output.scale;
    }
}
