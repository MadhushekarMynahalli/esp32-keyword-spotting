// Audio input over USB serial (no analog hardware needed).
// The PC (python/tools/stream_to_esp32.py) sends frames: 0xA5 0x5A + 320 x int16 (little endian),
// samples already in "ADC counts" scale (+-2048 = full scale), 16 kHz.
#pragma once
#include <Arduino.h>

struct AudioBlock { int16_t s[KWS_HOP]; };

static QueueHandle_t g_audioQ;
static volatile uint32_t g_dropped = 0;

static void audioTask(void*) {
  AudioBlock blk;
  uint8_t* p = (uint8_t*)blk.s;
  int st = 0; size_t n = 0;
  for (;;) {
    int c = Serial.read();
    if (c < 0) { vTaskDelay(1); continue; }
    if (st == 0)      { if (c == 0xA5) st = 1; }
    else if (st == 1) { st = (c == 0x5A) ? 2 : (c == 0xA5 ? 1 : 0); n = 0; }
    else {
      p[n++] = (uint8_t)c;
      if (n == sizeof(blk.s)) {
        if (xQueueSend(g_audioQ, &blk, 0) != pdTRUE) g_dropped++;
        st = 0;
      }
    }
  }
}

static void audioBegin() {
  g_audioQ = xQueueCreate(32, sizeof(AudioBlock));
  xTaskCreatePinnedToCore(audioTask, "audio", 4096, NULL, 3, NULL, 0);
}
