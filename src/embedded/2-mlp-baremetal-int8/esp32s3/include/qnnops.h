#ifndef __QNNOPS__
#define __QNNOPS__

#include <stdint.h>

struct qparams {
    int8_t zero;
    float scale;
};

struct qlayer {
    struct qparams input, weights, output;
};

int32_t mac(const int8_t *x, const int8_t *y, int size);
void mvm(const int8_t *M, int8_t *v, int32_t *out, int8_t v_zero, int nrows, int ncols);

void relu(const float *x, float *out, int size);
int argmax(const float *x, int size);

void quantize(const float *x, float scale, int8_t zero, int8_t *out, int size);
void dequantize(const struct qlayer *qlayer, const int32_t *x, float *out, int size);

#endif // end of NNOPS
