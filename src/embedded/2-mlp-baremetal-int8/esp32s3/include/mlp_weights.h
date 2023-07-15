#ifndef MLP_WEIGHTS
#define MLP_WEIGHTS

#include <stdint.h>

extern const float input_zero;
extern const float input_scale;

extern const int8_t layer_1_weights_zero;
extern const float layer_1_weights_scale;

extern const int8_t layer_1_zero;
extern const float layer_1_scale;

extern const int8_t layer_1_weights[12672];
extern const int16_t layer_1_weights_s16[12672];
extern const int8_t layer_3_weights_zero;
extern const float layer_3_weights_scale;

extern const int8_t layer_3_zero;
extern const float layer_3_scale;

extern const int8_t layer_3_weights[1440];
extern const int16_t layer_3_weights_s16[1440];

#endif // end of MLP_PARAMS
