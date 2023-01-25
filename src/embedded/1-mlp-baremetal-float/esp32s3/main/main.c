#include <stdio.h>
#include <string.h>

#include <sdkconfig.h>
#include <driver/uart.h>
#include <driver/gpio.h>

#include <nnops.h>
#include <mlp_weights.h>

#define MY_UART UART_NUM_0

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
    ESP_ERROR_CHECK(uart_param_config(MY_UART, &uart_config));
    ESP_ERROR_CHECK(uart_set_pin(MY_UART, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
    ESP_ERROR_CHECK(uart_driver_install(MY_UART, 1024 * 2, 1024 * 2, 0, NULL, 0));
}

void app_main(void)
{
    int r;
    char cmd = 0;
    int subject_id;
    char msg_waiting[] = "Waiting for input\n";
    char msg_reading[] = "Reading input\n";
    char msg_processing[] = "Processing\n";
    char msg_ready[] = "Ready\n";
    char msg_error[] = "Error\n";
    
    setup_uart();
    uart_write_bytes(MY_UART, msg_ready, strlen(msg_ready));

    while (1) {
        uart_write_bytes(MY_UART, msg_waiting, strlen(msg_waiting));

        while (cmd != 1)
            uart_read_bytes(MY_UART, (char *)&cmd, sizeof(char), 100);

        uart_write_bytes(MY_UART, msg_reading, strlen(msg_reading));

        cmd = 0;
        r = uart_read_bytes(MY_UART, (float *)input, 165*4, 100000);
        if (r != 165*sizeof(float))
            uart_write_bytes(MY_UART, msg_error, strlen(msg_error));
        
        uart_write_bytes(MY_UART, msg_processing, strlen(msg_processing));

        // layer 0
        mvm(layer_0_weights, input, layer_0_output, 96, 165);
        add(layer_0_output, layer_0_bias, layer_0_output, 96);

        // layer 1
        relu(layer_0_output, layer_1_output, 96);

        // layer 2
        mvm(layer_2_weights, layer_1_output, layer_2_output, 15, 96);
        add(layer_2_output, layer_2_bias, layer_2_output, 15);

        subject_id = argmax(layer_2_output, 15);
        uart_write_bytes(MY_UART, &subject_id, sizeof(int));
    }
}
