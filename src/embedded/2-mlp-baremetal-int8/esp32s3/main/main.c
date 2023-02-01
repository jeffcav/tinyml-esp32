#include <stdio.h>
#include <string.h>

#include <sdkconfig.h>
#include <driver/uart.h>
#include <driver/gpio.h>

#include <qnnops.h>
#include <mlp_weights.h>

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

float input[165];
int8_t input_quantized[165];

float buffer[96];
int8_t buffer_int8[96];
int32_t buffer_int32[96];

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
 * 165 is the input size
 * 96 is the number of neurons in the first hidden layer
 * 15 is the number of neurons in the output layer
 * 
*/
int run_mlp(const float *input) {
    int output;

    quantize(input, l1_qparams.input.scale, l1_qparams.input.zero, input_quantized, 165);

    mvm(layer_1_weights, input_quantized, buffer_int32, l1_qparams.input.zero, 96, 165);
    dequantize(&l1_qparams, buffer_int32, buffer, 96);

    relu(buffer, buffer, 96);
    quantize(buffer, l3_qparams.input.scale, l3_qparams.input.zero, buffer_int8, 96);

    mvm(layer_3_weights, buffer_int8, buffer_int32, l3_qparams.input.zero, 15, 96);
    dequantize(&l3_qparams, buffer_int32, buffer, 15);

    output = argmax(buffer, 15);

    return output;
}

void app_main(void)
{
    int r;
    int subject_id;
    int begin, end, elapsed;
    char cmd = CMD_NOOP;

    char msg_ready[] = "Ready\n";
    char msg_error[] = "Error\n";
    char msg_waiting[] = "Waiting for input\n";

    setup_quantization();
    setup_uart();

    uart_write_bytes(UART_NUM, msg_ready, strlen(msg_ready));

    while (1) {
        // log we are waiting for a new input
        uart_write_bytes(UART_NUM, msg_waiting, strlen(msg_waiting));

        while (cmd != CMD_INFERENCE_BEGIN)
            uart_read_bytes(UART_NUM, (char *)&cmd, sizeof(char), 100);
        cmd = CMD_NOOP;

        // read a new input
        r = uart_read_bytes(UART_NUM, (float *)input, 165*4, 100000);
        if (r != 165*sizeof(float))
            uart_write_bytes(UART_NUM, msg_error, strlen(msg_error));

        // fetch current CPU cycle count
        asm volatile("esync; rsr %0,ccount":"=a" (begin));

        subject_id = run_mlp(input);
        
        // fetch current CPU cycle count
        asm volatile("esync; rsr %0,ccount":"=a" (end));

        // log output
        uart_write_bytes(UART_NUM, &subject_id, sizeof(int));

        // log elapsed time
        elapsed = end-begin;
        uart_write_bytes(UART_NUM, (int*)&elapsed, sizeof(int));
    }
}
