// STAGE 2 -- Real-time keyword spotting on ESP32-WROOM-32 with TFLite Micro.
// Audio in, two modes (choose with USE_USB_AUDIO below):
//   1 = audio streamed from the PC over USB serial (no extra hardware)   <- default
//   0 = analog audio on GPIO34 (ADC1_CH6) through the bias circuit
// Requires: Arduino-ESP32 core 2.0.17, library "Chirale_TensorFlowLite".
// Generate kws_config.h + model_data.h with python/03_export_headers.py first.
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <esp_heap_caps.h>

#include "kws_config.h"
#include "model_data.h"
#include "dsp.h"

// ---------------- tunables ----------------
#define USE_USB_AUDIO  1        // 1 = PC streams audio over USB, 0 = analog input on GPIO34
#define AUDIO_BAUD     460800   // USB mode only; must match --baud in stream_to_esp32.py
#define LED_PIN        2        // on-board LED of most ESP32 DevKits
#define INFER_EVERY    4        // run the network every 4 hops = every 80 ms
#define RMS_GATE       20.0f    // ADC counts. Below this (over the last second) we assume silence.
#define DET_THRESH     0.80f    // averaged probability needed to report a keyword
#define SMOOTH_N       3        // average the last N predictions
#define REFRACT_MS     900      // ignore repeats of a detection for this long
#define DEBUG_EVERY_MS 500
constexpr int kArenaSize = 90 * 1024;
// ------------------------------------------

#if USE_USB_AUDIO
#include "audio_serial.h"
#else
#include "audio_adc.h"
#endif

static const tflite::Model* g_model_ptr;
static tflite::MicroInterpreter* g_interp;
static TfLiteTensor *g_in, *g_out;

static float   g_feat[KWS_NFRAMES][KWS_NMEL];   // rolling 1-second spectrogram
static int16_t g_hist[KWS_WIN];                 // last 30 ms of audio
static float   g_probs[SMOOTH_N][KWS_NCLASS];

static void fatal(const char* msg) {
  for (;;) { Serial.println(msg); delay(1000); }
}

static void tfBegin() {
  g_model_ptr = tflite::GetModel(g_model);
  static tflite::MicroMutableOpResolver<10> res;      // must include every op listed by 02_train script
  res.AddConv2D(); res.AddDepthwiseConv2D(); res.AddFullyConnected(); res.AddReshape();
  res.AddSoftmax(); res.AddMean(); res.AddAveragePool2D(); res.AddQuantize(); res.AddDequantize();
  uint8_t* arena = (uint8_t*)heap_caps_malloc(kArenaSize, MALLOC_CAP_8BIT);
  if (!arena) fatal("arena malloc failed");
  static tflite::MicroInterpreter interp(g_model_ptr, res, arena, kArenaSize);
  g_interp = &interp;
  if (g_interp->AllocateTensors() != kTfLiteOk) fatal("AllocateTensors failed (arena too small or op missing)");
  g_in = g_interp->input(0); g_out = g_interp->output(0);
  Serial.printf("model ok: arena used %u / %d bytes, input scale %.5f zp %d\n",
                (unsigned)g_interp->arena_used_bytes(), kArenaSize, g_in->params.scale, g_in->params.zero_point);
}

static void infer(float* probs) {
  const float s = g_in->params.scale; const int zp = g_in->params.zero_point;
  const float* f = &g_feat[0][0];
  int8_t* d = g_in->data.int8;
  for (int i = 0; i < KWS_NFRAMES * KWS_NMEL; i++) {
    int q = (int)lroundf(f[i] / s) + zp;
    d[i] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
  }
  if (g_interp->Invoke() != kTfLiteOk) { Serial.println("Invoke failed"); return; }
  for (int c = 0; c < KWS_NCLASS; c++)
    probs[c] = (g_out->data.int8[c] - g_out->params.zero_point) * g_out->params.scale;
}

static void procTask(void*) {
  AudioBlock blk;
  float rmsHist[50] = {0}; int rmsIdx = 0, hop = 0, sIdx = 0;
  uint32_t lastDet = 0, lastDbg = 0, inferMs = 0;
  float topP = 0; int topC = 0;
  for (;;) {
    xQueueReceive(g_audioQ, &blk, portMAX_DELAY);
    memmove(g_hist, g_hist + KWS_HOP, (KWS_WIN - KWS_HOP) * sizeof(int16_t));
    memcpy(g_hist + (KWS_WIN - KWS_HOP), blk.s, KWS_HOP * sizeof(int16_t));

    float frame[KWS_NMEL];
    rmsHist[rmsIdx] = dsp::logmel(g_hist, frame); rmsIdx = (rmsIdx + 1) % 50;
    memmove(&g_feat[0][0], &g_feat[1][0], (KWS_NFRAMES - 1) * KWS_NMEL * sizeof(float));
    memcpy(g_feat[KWS_NFRAMES - 1], frame, sizeof(frame));

    float maxRms = 0;
    for (int i = 0; i < 50; i++) if (rmsHist[i] > maxRms) maxRms = rmsHist[i];

    if (++hop >= INFER_EVERY) {
      hop = 0;
      float* p = g_probs[sIdx]; sIdx = (sIdx + 1) % SMOOTH_N;
      if (maxRms < RMS_GATE) {                       // nobody is talking -> skip the network
        for (int c = 0; c < KWS_NCLASS; c++) p[c] = (c == 0);
      } else {
        uint32_t t0 = millis(); infer(p); inferMs = millis() - t0;
      }
      float avg[KWS_NCLASS] = {0};
      for (int k = 0; k < SMOOTH_N; k++) for (int c = 0; c < KWS_NCLASS; c++) avg[c] += g_probs[k][c] / SMOOTH_N;
      topC = 0; for (int c = 1; c < KWS_NCLASS; c++) if (avg[c] > avg[topC]) topC = c;
      topP = avg[topC];

      if (topC >= 2 && topP >= DET_THRESH && millis() - lastDet > REFRACT_MS) {
        lastDet = millis();
        Serial.printf("DET,%s,%.2f\n", kLabels[topC], topP);
        if (!strcmp(kLabels[topC], "on"))  digitalWrite(LED_PIN, HIGH);   // demo action
        if (!strcmp(kLabels[topC], "off")) digitalWrite(LED_PIN, LOW);
      }
    }
    if (millis() - lastDbg > DEBUG_EVERY_MS) {
      lastDbg = millis();
      Serial.printf("DBG,%.1f,%u,%s,%.2f,%u\n", maxRms, (unsigned)inferMs, kLabels[topC], topP, (unsigned)g_dropped);
    }
  }
}

void setup() {
#if USE_USB_AUDIO
  Serial.setRxBufferSize(16384);     // must be called before begin()
  Serial.begin(AUDIO_BAUD);
#else
  Serial.begin(115200);
#endif
  pinMode(LED_PIN, OUTPUT);
  dsp::init();
  tfBegin();
  audioBegin();
  xTaskCreatePinnedToCore(procTask, "proc", 16384, NULL, 2, NULL, 1);     // core 1, big stack for TFLM
  Serial.println("listening ...");
}

void loop() { vTaskDelay(portMAX_DELAY); }
