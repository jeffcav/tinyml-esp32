#include <math.h>
#include <nnops.h>
#include <stdint.h>
#include <esp_dsp.h>

float mac(const float *x, const float *y, int size) {
    float acc = 0.0;

    for (int i = 0; i < size; i++)
        acc += (x[i] * y[i]);

    return acc;
}

int16_t mac8(const int8_t *x, const int8_t *y, int size) {
    int16_t acc = 0;

    for (int i = 0; i < size; i++)
        acc += (((int16_t)x[i] * (int16_t)y[i]));

    return acc;
}

void mvm(const float *M, const float *v, float *out, int nrows, int ncols) {
    int row;

    for (row = 0; row < nrows; row++)
        out[row] = mac(&M[row*ncols], v, ncols);
}

void mvmx(const float *M, const float *v, float *out, int nrows, int ncols) {
    int row;

    for (row = 0; row < nrows; row++)
        dsps_dotprod_f32_aes3(&M[row*ncols], v, &out[row], ncols);
}

void mvm8(const int8_t *M, int8_t *v, int16_t *out, int8_t v_zero, int nrows, int ncols) {
    for (int i = 0; i < ncols; i++)
        v[i] = (v[i] - v_zero);

    for (int row = 0; row < nrows; row++)
        out[row] = mac8(&M[row*ncols], v, ncols);
}

void mvm16(const int16_t *M, int16_t *v, int16_t *out, int16_t v_zero, int nrows, int ncols) {
    for (int i = 0; i < ncols; i++)
        v[i] = (v[i] - v_zero);

    for (int row = 0; row < nrows; row++)
        dsps_dotprod_s16_ae32(&M[row*ncols], v, &out[row], ncols, 15);
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

void quantize8(const float *x, float scale, int8_t zero, int8_t *out, int size) {
    for (int i = 0; i < size; i++)
        out[i] = (int8_t)(round(x[i] / scale) + zero);
}

void quantize16(const float *x, float scale, int16_t zero, int16_t *out, int size) {
    for (int i = 0; i < size; i++)
        out[i] = (int16_t)(round(x[i] / scale) + zero);
}

void dequantize16(const struct qlayer *qlayer, const int16_t *x, float *out, int size) {
    float mvm_scale;

    // TODO pre-calculate this value during compile-time
    mvm_scale = (qlayer->input.scale * qlayer->weights.scale);

    for (int i = 0; i < size; i++) {
        float temp = x[i] * mvm_scale;
        out[i] = (temp - qlayer->output.zero) * qlayer->output.scale;
    }
}
