#ifndef __NNOPS__
#define __NNOPS__

#include <stdint.h>

float mac(const float *x, const float *y, int size);
void mvm(const float *M, const float *v, float *out, int nrows, int ncols);

void relu(const float *x, float *out, int size);
int argmax(const float *x, int size);

#endif // end of NNOPS
