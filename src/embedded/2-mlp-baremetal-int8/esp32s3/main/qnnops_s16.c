#include <math.h>
#include <qnnops.h>
#include <stdint.h>
#include <esp_dsp.h>

void mvm_s16(const int16_t *M, int16_t *v, int16_t *out, int16_t v_zero, int nrows, int ncols) {
    for (int i = 0; i < ncols; i++)
        v[i] = (v[i] - v_zero);

    for (int row = 0; row < nrows; row++)
        dsps_dotprod_s16_ae32(&M[row*ncols], v, &out[row], ncols, 15);
}

void quantize_s16(const float *x, float scale, int16_t zero, int16_t *out, int size) {
    for (int i = 0; i < size; i++)
        out[i] = (int16_t)(round(x[i] / scale) + zero);
}

void dequantize_s16(const struct qlayer *qlayer, const int16_t *x, float *out, int size) {
    float mvm_scale;

    // TODO pre-calculate this value during compile-time
    mvm_scale = (qlayer->input.scale * qlayer->weights.scale);

    for (int i = 0; i < size; i++) {
        float temp = x[i] * mvm_scale;
        out[i] = (temp - qlayer->output.zero) * qlayer->output.scale;
    }
}
