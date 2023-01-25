#ifndef __NNOPS__
#define __NNOPS__

#include <stdint.h>

void add(const float *x, const float *y, float *out, int size);
void mac(const float *x, const float *y, float *out, int size);
void mvm(const float *M, const float *v, float *out, int nrows, int ncols);

void relu(float *x, float *out, int size);
int argmax(float *x, int size);

#endif // end of NNOPS
