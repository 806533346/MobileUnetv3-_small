/**
 * ESP32-S3 UART 流式图像分割推理 (三缓冲流水线版)
 *
 * ===== 协议 =====
 * PC → ESP32: [4B magic: 0xFEED0010 LE] [196608B INT8 NHWC data]
 * ESP32 → PC: [4B magic: 0xFEED0020 LE] [4B infer_time_us LE] [65536B uint8 mask]
 * PC → ESP32: [4B magic: 0xFEEDFFFF LE] → 停止
 *
 * ===== 流水线 =====
 * Core 0 (UART RX task): 持续接收数据到三缓冲，从不阻塞等待推理
 * Core 1 (inference):     取已就绪的缓冲做推理 + 回传结果
 *
 * 三缓冲 + 计数信号量设计确保 RX task 始终在调用 uart_read_bytes，
 * UART ring buffer 可以很小（8KB），避免 PSRAM 争用。
 *
 * ===== 预处理 (PC 侧完成) =====
 * albu.Resize(256,256) → albu.Normalize() → /255 → NHWC float32
 * → quantize: round(value * 16384) → clamp [-128,127] → INT8
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "driver/uart.h"
#if CONFIG_SPIRAM
#include "esp_psram.h"
#endif
#include "nvs_flash.h"

#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"
#include "test_config.h"

#if CONFIG_SPIRAM
#define EXT_MEM_CAPS MALLOC_CAP_SPIRAM
#else
#define EXT_MEM_CAPS (MALLOC_CAP_8BIT | MALLOC_CAP_32BIT)
#endif

static const char *TAG = "SEG";

static const uint32_t MAGIC_IMAGE  = 0xFEED0010;
static const uint32_t MAGIC_RESULT = 0xFEED0030;  // packed bitmask format
static const uint32_t MAGIC_STOP   = 0xFEEDFFFF;

static const int BAUD_DATA = 921600;
static const int DATA_SIZE = TEST_INPUT_SIZE;   // 196608
static const int MASK_SIZE = TEST_OUTPUT_SIZE;  // 65536
static const int PACKED_SIZE = TEST_OUTPUT_SIZE / 8;  // 8192 packed bits
static const int N_BUF = 3;  // triple buffering

// -------- shared state between cores --------
static int8_t  *input_buf[N_BUF];   // triple input buffers (PSRAM, 196KB each)
static uint8_t *output_buf[N_BUF];  // triple output buffers (PSRAM, 8KB packed each)
static volatile int fill_idx = 0;   // next buffer to fill (RX task writes)
static volatile int proc_idx = 0;   // next buffer to process (inference reads)

static SemaphoreHandle_t s_buf_filled;  // counting: buffers ready for inference
static SemaphoreHandle_t s_buf_free;    // counting: buffers available for RX
static volatile int g_img_count = 0;
static volatile bool running = true;

// model (set by app_main, read by infer_task)
static dl::Model *g_model = nullptr;
static std::string g_input_name;
static int g_input_exponent = -14;
static size_t g_output_count = TEST_OUTPUT_SIZE;


// ===== Core 0: UART RX task (never blocks — continuously drains ring buffer) =====
static void uart_rx_task(void *arg) {
    uint32_t magic;
    uint8_t tmp[512];

    while (running) {
        // Wait for a free buffer slot
        if (!xSemaphoreTake(s_buf_free, pdMS_TO_TICKS(5000))) {
            break;  // timeout (shouldn't happen in normal operation)
        }
        if (!running) break;

        // Read magic — drain in small chunks to keep ring buffer empty
        int n = uart_read_bytes(UART_NUM_0, &magic, 4, pdMS_TO_TICKS(5000));
        if (n < 4) {
            xSemaphoreGive(s_buf_free);  // return the slot
            if (!running) break;
            continue;  // timeout, retry
        }

        if (magic == MAGIC_STOP) {
            running = false;
            xSemaphoreGive(s_buf_filled);  // unblock inference
            break;
        }

        if (magic != MAGIC_IMAGE) {
            xSemaphoreGive(s_buf_free);  // return the slot
            continue;  // skip garbage
        }

        // Read image data in chunks — keeps UART ring buffer drained
        int total = 0;
        int fill = fill_idx;  // local copy of the buffer index
        while (total < DATA_SIZE) {
            int chunk = (DATA_SIZE - total) < (int)sizeof(tmp)
                        ? (DATA_SIZE - total) : (int)sizeof(tmp);
            n = uart_read_bytes(UART_NUM_0, tmp, chunk,
                                pdMS_TO_TICKS(5000));
            if (n <= 0) break;
            memcpy((uint8_t *)input_buf[fill] + total, tmp, n);
            total += n;
        }

        if (total < DATA_SIZE) {
            running = false;
            xSemaphoreGive(s_buf_filled);  // unblock inference
            break;
        }

        // Hand off to inference
        fill_idx = (fill + 1) % N_BUF;
        xSemaphoreGive(s_buf_filled);
    }

    vTaskDelete(NULL);
}


// ===== Core 1: Inference (runs on main task) =====
static void run_inference_loop() {
    while (running) {
        // Wait for a filled buffer
        if (!xSemaphoreTake(s_buf_filled, pdMS_TO_TICKS(10000))) {
            break;  // timeout
        }
        if (!running) break;

        int proc = proc_idx;  // local copy of the buffer index

        // Build input tensor (zero-copy: wraps the INT8 NHWC buffer directly)
        std::vector<int> input_shape = {1, TEST_INPUT_H, TEST_INPUT_W, TEST_INPUT_C};
        dl::TensorBase *input_tensor = new dl::TensorBase(
            input_shape, input_buf[proc], g_input_exponent,
            dl::DATA_TYPE_INT8, false, EXT_MEM_CAPS);
        std::map<std::string, dl::TensorBase *> inputs;
        inputs[g_input_name] = input_tensor;

        // Inference
        int64_t t0 = esp_timer_get_time();
        g_model->run(inputs);
        int64_t t1 = esp_timer_get_time();

        // Dequantize + binarize + pack into bits (8 pixels/byte)
        auto model_outputs = g_model->get_outputs();
        dl::TensorBase *output_tensor = model_outputs.begin()->second;
        float output_scale = DL_SCALE(output_tensor->exponent);
        const int8_t *quant_out = reinterpret_cast<const int8_t *>(output_tensor->data);
        uint8_t *packed = output_buf[proc];
        memset(packed, 0, PACKED_SIZE);
        for (size_t i = 0; i < g_output_count; i++) {
            float v = dl::dequantize(quant_out[i], output_scale);
            if (v > -2) packed[i >> 3] |= (1 << (i & 7));
        }

        delete input_tensor;

        // Send result: [4B magic] [4B infer_time_us] [8192B packed bits]
        uint32_t result_magic = MAGIC_RESULT;
        int32_t infer_us = (int32_t)(t1 - t0);
        uart_write_bytes(UART_NUM_0, &result_magic, 4);
        uart_write_bytes(UART_NUM_0, &infer_us, 4);
        uart_write_bytes(UART_NUM_0, packed, PACKED_SIZE);

        g_img_count = g_img_count + 1;

        // Return buffer to RX pool
        proc_idx = (proc + 1) % N_BUF;
        xSemaphoreGive(s_buf_free);
    }
}


extern "C" void app_main() {
    ESP_LOGI(TAG, "=== UART STREAMING INFERENCE (TRIPLE BUF) ===");

    // ===== 1. NVS =====
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        nvs_flash_init();
    }

    // ===== 2. Load model =====
    ESP_LOGI(TAG, "Loading model...");
    g_model = new dl::Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION,
                            0, dl::MEMORY_MANAGER_GREEDY, nullptr, false);

    auto model_inputs = g_model->get_inputs();
    dl::TensorBase *input_tpl = model_inputs.begin()->second;
    g_input_name = model_inputs.begin()->first;
    g_input_exponent = input_tpl->exponent;
    ESP_LOGI(TAG, "Input: name=%s exponent=%d inv_scale=%.0f",
             g_input_name.c_str(), g_input_exponent,
             1.0f / DL_SCALE(g_input_exponent));

    g_model->minimize();

    // ===== 3. Allocate triple buffers in PSRAM =====
    for (int i = 0; i < N_BUF; i++) {
        input_buf[i]  = (int8_t  *)heap_caps_calloc(DATA_SIZE, 1, EXT_MEM_CAPS);
        output_buf[i] = (uint8_t *)heap_caps_malloc(PACKED_SIZE, EXT_MEM_CAPS);
        if (!input_buf[i] || !output_buf[i]) {
            ESP_LOGE(TAG, "Buffer %d allocation failed", i);
            for (int j = 0; j <= i; j++) {
                if (input_buf[j])  heap_caps_free(input_buf[j]);
                if (output_buf[j]) heap_caps_free(output_buf[j]);
            }
            delete g_model;
            return;
        }
    }

    // ===== 4. Counting semaphores =====
    s_buf_filled = xSemaphoreCreateCounting(N_BUF, 0);  // max=3, initial=0
    s_buf_free   = xSemaphoreCreateCounting(N_BUF, N_BUF);  // max=3, initial=3

    // ===== 5. UART driver with small RX ring buffer =====
    // Small buffer (8KB) — RX task drains it continuously, no overflow
    uart_driver_install(UART_NUM_0, 8192, 8192, 0, NULL, 0);
    uart_write_bytes(UART_NUM_0, "READY\n", 6);

    // Wait for PC handshake
    char go;
    while (uart_read_bytes(UART_NUM_0, &go, 1, pdMS_TO_TICKS(200)) <= 0) {
        uart_write_bytes(UART_NUM_0, "READY\n", 6);
    }

    // ===== 6. Switch to binary mode =====
    vTaskDelay(pdMS_TO_TICKS(200));
    uart_set_baudrate(UART_NUM_0, BAUD_DATA);
    esp_log_level_set("*", ESP_LOG_NONE);

    // ===== 7. Launch RX task on Core 0 =====
    xTaskCreatePinnedToCore(uart_rx_task, "uart_rx", 8192,
                            NULL, configMAX_PRIORITIES - 1, NULL, 0);

    // ===== 8. Run inference loop on Core 1 (current core) =====
    {
        int64_t total_start = esp_timer_get_time();
        run_inference_loop();
        int64_t total_elapsed = esp_timer_get_time() - total_start;
        esp_log_level_set("*", ESP_LOG_INFO);
        ESP_LOGI(TAG, "=== DONE: %d images in %lld ms ===", g_img_count,
                 total_elapsed / 1000);
    }

    // ===== 9. Cleanup =====
    for (int i = 0; i < N_BUF; i++) {
        if (input_buf[i])  heap_caps_free(input_buf[i]);
        if (output_buf[i]) heap_caps_free(output_buf[i]);
    }
    vSemaphoreDelete(s_buf_filled);
    vSemaphoreDelete(s_buf_free);
    delete g_model;
}
