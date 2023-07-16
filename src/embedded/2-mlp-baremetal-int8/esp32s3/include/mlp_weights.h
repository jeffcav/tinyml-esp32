#ifndef MLP_WEIGHTS
#define MLP_WEIGHTS

#include <stdint.h>

#define LAYER_INPUT_LEN 132
#define LAYER_1_LEN 96
#define LAYER_3_LEN 15

extern const float input_zero;
extern const float input_scale;

extern const int8_t layer_1_weights_zero;
extern const float layer_1_weights_scale;

extern const int8_t layer_1_zero;
extern const float layer_1_scale;

extern const int8_t layer_1_weights8[12672];
extern const int16_t layer_1_weights16[12672];
extern const float layer_1_weights[12672];
extern const int8_t layer_3_weights_zero;
extern const float layer_3_weights_scale;

extern const int8_t layer_3_zero;
extern const float layer_3_scale;

extern const int8_t layer_3_weights8[1440];
extern const int16_t layer_3_weights16[1440];
extern const float layer_3_weights[1440];

#endif // end of MLP_PARAMS
