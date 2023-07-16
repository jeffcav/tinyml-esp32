#include <stdio.h>
#include <string.h>

#include <sdkconfig.h>
#include <driver/uart.h>
#include <driver/gpio.h>

#include <qnnops.h>
#include <mlp_weights.h>

#include <esp_dsp.h>

#define UART_NUM UART_NUM_0

#define CMD_NOOP 0
#define CMD_INFERENCE_BEGIN 1

uart_config_t uart_config = {
    .baud_rate = 115200,
    .data_bits = UART_DATA_8_BITS,
    .parity = UART_PARITY_DISABLE,
    .stop_bits = UART_STOP_BITS_1,
    .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
    .source_clk = UART_SCLK_DEFAULT,
};

float input[132];
int8_t input_quantized[132];
int16_t input_quantized_s16[132];

float buffer[96];
int8_t buffer_int8[96];
int32_t buffer_int32[96];
int16_t buffer_s16[96];

struct qlayer l1_qparams, l3_qparams;

void setup_quantization() {
    l1_qparams = (struct qlayer) {
        .input = {
            .zero = input_zero,
            .scale = input_scale
        },
        .weights = {
            .zero = layer_1_weights_zero,
            .scale = layer_1_weights_scale,
        },
        .output = {
            .zero = layer_1_zero,
            .scale = layer_1_scale,
        }
    };

    l3_qparams = (struct qlayer) {
        .input = {
            .zero = layer_1_zero,
            .scale = layer_1_scale
        },
        .weights = {
            .zero = layer_3_weights_zero,
            .scale = layer_3_weights_scale,
        },
        .output = {
            .zero = layer_3_zero,
            .scale = layer_3_scale,
        }
    };
}

void setup_uart() {
    ESP_ERROR_CHECK(uart_param_config(UART_NUM, &uart_config));
    ESP_ERROR_CHECK(uart_set_pin(UART_NUM, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
    ESP_ERROR_CHECK(uart_driver_install(UART_NUM, 1024 * 2, 1024 * 2, 0, NULL, 0));
}

/**
 * Magic numbers explained:
 * 
 * 132 is the input size
 * 96 is the number of neurons in the first hidden layer
 * 15 is the number of neurons in the output layer
 * 
*/
int run_mlp(const float *input) {
    int output;

    quantize(input, l1_qparams.input.scale, l1_qparams.input.zero, input_quantized, 132);

    mvm(layer_1_weights, input_quantized, buffer_int32, l1_qparams.input.zero, 96, 132);
    dequantize(&l1_qparams, buffer_int32, buffer, 96);

    relu(buffer, buffer, 96);
    quantize(buffer, l3_qparams.input.scale, l3_qparams.input.zero, buffer_int8, 96);

    mvm(layer_3_weights, buffer_int8, buffer_int32, l3_qparams.input.zero, 15, 96);
    dequantize(&l3_qparams, buffer_int32, buffer, 15);

    output = argmax(buffer, 15);

    return output;
}

/**
 * Magic numbers explained:
 * 
 * 132 is the input size
 * 96 is the number of neurons in the first hidden layer
 * 15 is the number of neurons in the output layer
 * 
*/
int run_mlp_s16(const float *input) {
    int output;

    quantize_s16(input, l1_qparams.input.scale, (int16_t)l1_qparams.input.zero, input_quantized_s16, 132);

    mvm_s16(layer_1_weights_s16, input_quantized_s16, buffer_s16, (int16_t)l1_qparams.input.zero, 96, 132);
    dequantize_s16(&l1_qparams, buffer_s16, buffer, 96);

    relu(buffer, buffer, 96);
    quantize_s16(buffer, l3_qparams.input.scale, (int16_t)l3_qparams.input.zero, input_quantized_s16, 96);

    mvm_s16(layer_3_weights_s16, input_quantized_s16, buffer_s16, (int16_t)l3_qparams.input.zero, 15, 96);
    dequantize_s16(&l3_qparams, buffer_s16, buffer, 15);

    output = argmax(buffer, 15);

    return output;
}

void app_main(void)
{
    char cmd;
    int recv_bytes;
    int subject_id;
    int time_begin, time_end, time_elapsed;

    char msg_ready[] = "Ready\n";
    char msg_error[] = "Error\n";
    char msg_waiting[] = "Waiting for input\n";

    setup_quantization();
    setup_uart();

    uart_write_bytes(UART_NUM, msg_ready, strlen(msg_ready));

    while (1) {
        cmd = CMD_NOOP;

        uart_write_bytes(UART_NUM, msg_waiting, strlen(msg_waiting));

        while (cmd != CMD_INFERENCE_BEGIN)
            uart_read_bytes(UART_NUM, (char *)&cmd, sizeof(char), 100);

        recv_bytes = uart_read_bytes(UART_NUM, (float *)input, 132*4, 100000);
        if (recv_bytes != 132*sizeof(float))
            uart_write_bytes(UART_NUM, msg_error, strlen(msg_error));

        asm volatile("esync; rsr %0,ccount":"=a" (time_begin));
        subject_id = run_mlp_s16(input);
        asm volatile("esync; rsr %0,ccount":"=a" (time_end));

        uart_write_bytes(UART_NUM, &subject_id, sizeof(int));

        time_elapsed = time_end - time_begin;
        uart_write_bytes(UART_NUM, (int*)&time_elapsed, sizeof(int));
    }
}
