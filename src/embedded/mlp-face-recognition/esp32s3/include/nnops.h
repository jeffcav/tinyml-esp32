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

float mac(const float *x, const float *y, int size);
int16_t mac8(const int8_t *x, const int8_t *y, int size);

void mvm(const float *M, const float *v, float *out, int nrows, int ncols);
void mvmx(const float *M, const float *v, float *out, int nrows, int ncols);
void mvm8(const int8_t *M, int8_t *v, int16_t *out, int8_t v_zero, int nrows, int ncols);
void mvm16(const int16_t *M, int16_t *v, int16_t *out, int16_t v_zero, int nrows, int ncols);

void relu(const float *x, float *out, int size);
int argmax(const float *x, int size);

void quantize8(const float *x, float scale, int8_t zero, int8_t *out, int size);
void quantize16(const float *x, float scale, int16_t zero, int16_t *out, int size);

void dequantize16(const struct qlayer *qlayer, const int16_t *x, float *out, int size);

#endif // end of NNOPS
