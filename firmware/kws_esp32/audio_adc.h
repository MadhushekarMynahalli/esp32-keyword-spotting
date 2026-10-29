// Audio capture: ESP32 built-in ADC1 -> I2S DMA -> FreeRTOS queue of 20 ms blocks.
#pragma once
#include <Arduino.h>
#include <driver/i2s.h>
#include <driver/adc.h>

#define ADC_CH ADC1_CHANNEL_6      // GPIO34

struct AudioBlock { int16_t s[KWS_HOP]; };    // centred samples (+-2048 counts)

static QueueHandle_t g_audioQ;
static volatile uint32_t g_dropped = 0;

static void audioTask(void*) {
  static uint16_t raw[KWS_HOP];
  AudioBlock blk;
  for (;;) {
    size_t bytes = 0;
    i2s_read(I2S_NUM_0, raw, sizeof(raw), &bytes, portMAX_DELAY);
    for (int i = 0; i < KWS_HOP; i++) blk.s[i] = (int16_t)(raw[i] & 0x0FFF) - 2048;
    if (xQueueSend(g_audioQ, &blk, 0) != pdTRUE) g_dropped++;   // processing too slow -> count it
  }
}

static void audioBegin() {
  g_audioQ = xQueueCreate(32, sizeof(AudioBlock));      // ~640 ms of slack
  i2s_config_t cfg = {};
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_ADC_BUILT_IN);
  cfg.sample_rate = KWS_SR;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
  cfg.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_MSB;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count = 8;
  cfg.dma_buf_len = 256;
  cfg.use_apll = false;
  i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL);
  i2s_set_adc_mode(ADC_UNIT_1, ADC_CH);
  adc1_config_width(ADC_WIDTH_BIT_12);
  adc1_config_channel_atten(ADC_CH, ADC_ATTEN_DB_11);
  i2s_adc_enable(I2S_NUM_0);
  xTaskCreatePinnedToCore(audioTask, "audio", 4096, NULL, 3, NULL, 0);   // core 0
}
