#include <stdio.h>
#include <string.h>

#include <sdkconfig.h>
#include <driver/uart.h>
#include <driver/gpio.h>

#include <nnops.h>
#include <mlp_weights.h>

#define UART_NUM UART_NUM_0

#define CMD_NOOP 0
#define CMD_INFERENCE_BEGIN 1

float input[165];
float buffer[96];

uart_config_t uart_config = {
    .baud_rate = 115200,
    .data_bits = UART_DATA_8_BITS,
    .parity = UART_PARITY_DISABLE,
    .stop_bits = UART_STOP_BITS_1,
    .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
    .source_clk = UART_SCLK_DEFAULT,
};

/**
 * Magic numbers explained:
 * 
 * 165 is the input size
 * 96 is the number of neurons in the first hidden layer
 * 15 is the number of neurons in the output layer
 * 
*/
int run_mlp(float *input) {
    int output;

    mvm(layer_0_weights, input, buffer, 96, 165);
    relu(buffer, buffer, 96);

    mvm(layer_2_weights, buffer, buffer, 15, 96);
    output = argmax(buffer, 15);

    return output;
}

void setup_uart() {
    ESP_ERROR_CHECK(uart_param_config(UART_NUM, &uart_config));
    ESP_ERROR_CHECK(uart_set_pin(UART_NUM, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
    ESP_ERROR_CHECK(uart_driver_install(UART_NUM, 1024 * 2, 1024 * 2, 0, NULL, 0));
}

void app_main(void)
{
    char cmd;
    int subject_id;
    int recv_bytes;
    int time_begin, time_end, time_elapsed;

    char msg_ready[] = "Ready\n";
    char msg_error[] = "Error\n";
    char msg_waiting[] = "Waiting for input\n";

    setup_uart();
    uart_write_bytes(UART_NUM, msg_ready, strlen(msg_ready));

    while (1) {
        cmd = CMD_NOOP;

        uart_write_bytes(UART_NUM, msg_waiting, strlen(msg_waiting));

        while (cmd != CMD_INFERENCE_BEGIN)
            uart_read_bytes(UART_NUM, (char *)&cmd, sizeof(char), 100);

        recv_bytes = uart_read_bytes(UART_NUM, (float *)input, 165*4, 100000);

        if (recv_bytes != 165*sizeof(float))
            uart_write_bytes(UART_NUM, msg_error, strlen(msg_error));

        asm volatile("esync; rsr %0,ccount":"=a" (time_begin));
        subject_id = run_mlp(input);
        asm volatile("esync; rsr %0,ccount":"=a" (time_end));

        uart_write_bytes(UART_NUM, &subject_id, sizeof(int));

        time_elapsed = time_end - time_begin;
        uart_write_bytes(UART_NUM, (int*)&time_elapsed, sizeof(int));
    }
}
