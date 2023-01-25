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

// In a real-world implementation, we'd reuse
// memory between layers, but for educational
// purposes we're giving each a new buffer.
float input[165];
float layer_0_output[96];
float layer_1_output[96];
float layer_2_output[15];

uart_config_t uart_config = {
    .baud_rate = 115200,
    .data_bits = UART_DATA_8_BITS,
    .parity = UART_PARITY_DISABLE,
    .stop_bits = UART_STOP_BITS_1,
    .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
    .source_clk = UART_SCLK_DEFAULT,
};

void setup_uart() {
    ESP_ERROR_CHECK(uart_param_config(UART_NUM, &uart_config));
    ESP_ERROR_CHECK(uart_set_pin(UART_NUM, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
    ESP_ERROR_CHECK(uart_driver_install(UART_NUM, 1024 * 2, 1024 * 2, 0, NULL, 0));
}

void app_main(void)
{
    int r;
    char cmd = 0;
    int subject_id;
    int begin, end, elapsed;

    char msg_ready[] = "Ready\n";
    char msg_error[] = "Error\n";
    char msg_waiting[] = "Waiting for input\n";

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

        // neural net layer 0
        mvm(layer_0_weights, input, layer_0_output, 96, 165);
        add(layer_0_output, layer_0_bias, layer_0_output, 96);

        // neural net layer 1
        relu(layer_0_output, layer_1_output, 96);

        // neural net layer 2
        mvm(layer_2_weights, layer_1_output, layer_2_output, 15, 96);
        add(layer_2_output, layer_2_bias, layer_2_output, 15);

        // neural net output
        subject_id = argmax(layer_2_output, 15);

        // fetch current CPU cycle count
        asm volatile("esync; rsr %0,ccount":"=a" (end));

        // log output
        uart_write_bytes(UART_NUM, &subject_id, sizeof(int));

        // log elapsed time
        elapsed = end-begin;
        uart_write_bytes(UART_NUM, (int*)&elapsed, sizeof(int));
    }
}
