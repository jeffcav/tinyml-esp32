#include <stdio.h>
#include <string.h>

#include <sdkconfig.h>
#include <driver/uart.h>
#include <driver/gpio.h>

#include <nnops.h>
#include <mlp_weights.h>

#define UART_NUM UART_NUM_0

#define CMD_NOOP 0
#define CMD_INFERENCE_FLOAT       0b0001
#define CMD_INFERENCE_FLOAT_ACCEL 0b0011
#define CMD_INFERENCE_INT8        0b0101
#define CMD_INFERENCE_INT8_ACCEL  0b1001


#define IS_INFERENCE(x) (x & 1)

uart_config_t uart_config = {
    .baud_rate = 115200,
    .data_bits = UART_DATA_8_BITS,
    .parity = UART_PARITY_DISABLE,
    .stop_bits = UART_STOP_BITS_1,
    .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
    .source_clk = UART_SCLK_DEFAULT,
};

float input[132];
int16_t fxp_input[132];

float buffer[96];
int16_t fxp_buffer[96];

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

int run_mlp(float *input) {
    int output;

    mvm(layer_1_weights, input, buffer, LAYER_1_LEN, LAYER_INPUT_LEN);
    relu(buffer, input, 96);

    mvm(layer_3_weights, input, buffer, LAYER_3_LEN, LAYER_1_LEN);
    output = argmax(buffer, LAYER_3_LEN);

    return output;
}

int run_mlpx(float *input) {
    int output;

    mvmx(layer_1_weights, input, buffer, LAYER_1_LEN, LAYER_INPUT_LEN);
    relu(buffer, input, 96);

    mvmx(layer_3_weights, input, buffer, LAYER_3_LEN, LAYER_1_LEN);
    output = argmax(buffer, LAYER_3_LEN);

    return output;
}

int run_mlp8(const float *input) {
    int output;

    quantize8(input, l1_qparams.input.scale, l1_qparams.input.zero, (int8_t*)fxp_input, LAYER_INPUT_LEN);

    mvm8(layer_1_weights8, (int8_t*)fxp_input, fxp_buffer, l1_qparams.input.zero, LAYER_1_LEN, LAYER_INPUT_LEN);
    dequantize16(&l1_qparams, fxp_buffer, buffer, 96);

    relu(buffer, buffer, LAYER_1_LEN);
    quantize8(buffer, l3_qparams.input.scale, l3_qparams.input.zero, (int8_t*)fxp_input, LAYER_1_LEN);

    mvm8(layer_3_weights8, (int8_t*)fxp_input, fxp_buffer, l3_qparams.input.zero, LAYER_3_LEN, LAYER_1_LEN);
    dequantize16(&l3_qparams, fxp_buffer, buffer, LAYER_3_LEN);

    output = argmax(buffer, LAYER_3_LEN);

    return output;
}

int run_mlp16(const float *input) {
    int output;

    quantize16(input, l1_qparams.input.scale, (int16_t)l1_qparams.input.zero, fxp_input, LAYER_INPUT_LEN);

    mvm16(layer_1_weights16, fxp_input, fxp_buffer, (int16_t)l1_qparams.input.zero, LAYER_1_LEN, LAYER_INPUT_LEN);
    dequantize16(&l1_qparams, fxp_buffer, buffer, LAYER_1_LEN);

    relu(buffer, buffer, LAYER_1_LEN);
    quantize16(buffer, l3_qparams.input.scale, (int16_t)l3_qparams.input.zero, fxp_input, LAYER_1_LEN);

    mvm16(layer_3_weights16, fxp_input, fxp_buffer, (int16_t)l3_qparams.input.zero, LAYER_3_LEN, LAYER_1_LEN);
    dequantize16(&l3_qparams, fxp_buffer, buffer, LAYER_3_LEN);

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

        while (!IS_INFERENCE(cmd))
            uart_read_bytes(UART_NUM, (char *)&cmd, sizeof(char), 100);

        recv_bytes = uart_read_bytes(UART_NUM, (float *)input, 132*4, 100000);
        if (recv_bytes != 132*sizeof(float))
            uart_write_bytes(UART_NUM, msg_error, strlen(msg_error));

        asm volatile("esync; rsr %0,ccount":"=a" (time_begin));
        switch(cmd) {
            case CMD_INFERENCE_FLOAT:
                subject_id = run_mlp(input);
                break;
            case CMD_INFERENCE_FLOAT_ACCEL:
                subject_id = run_mlpx(input);
                break;
            case CMD_INFERENCE_INT8:
                subject_id = run_mlp8(input);
                break;
            case CMD_INFERENCE_INT8_ACCEL:
                subject_id = run_mlp16(input);
                break;
            default:
                subject_id = run_mlp(input);
        }
        asm volatile("esync; rsr %0,ccount":"=a" (time_end));

        uart_write_bytes(UART_NUM, &subject_id, sizeof(int));

        time_elapsed = time_end - time_begin;
        uart_write_bytes(UART_NUM, (int*)&time_elapsed, sizeof(int));
    }
}
