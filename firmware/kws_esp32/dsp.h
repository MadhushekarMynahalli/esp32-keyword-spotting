// Log-mel feature extraction. C++ twin of python/features.py -- keep them identical.
#pragma once
#include <math.h>
#include <string.h>
#include <stdint.h>
#include "kws_config.h"

namespace dsp {
static float twc[KWS_NFFT / 2], tws[KWS_NFFT / 2];
static float re[KWS_NFFT], im[KWS_NFFT], pw[KWS_NFFT / 2 + 1];

inline void init() {
  for (int i = 0; i < KWS_NFFT / 2; i++) {
    twc[i] = cosf(2.0f * (float)M_PI * i / KWS_NFFT);
    tws[i] = -sinf(2.0f * (float)M_PI * i / KWS_NFFT);
  }
}

inline void fft() {                       // in-place radix-2, N = KWS_NFFT
  for (unsigned i = 1, j = 0; i < KWS_NFFT; i++) {
    unsigned bit = KWS_NFFT >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { float t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (unsigned len = 2; len <= KWS_NFFT; len <<= 1) {
    unsigned half = len >> 1, step = KWS_NFFT / len;
    for (unsigned i = 0; i < KWS_NFFT; i += len)
      for (unsigned k = 0; k < half; k++) {
        float wr = twc[k * step], wi = tws[k * step];
        float xr = re[i + k + half], xi = im[i + k + half];
        float tr = xr * wr - xi * wi, ti = xr * wi + xi * wr;
        re[i + k + half] = re[i + k] - tr; im[i + k + half] = im[i + k] - ti;
        re[i + k] += tr;                   im[i + k] += ti;
      }
  }
}

// frame: KWS_WIN samples in ADC counts. out: KWS_NMEL normalised log-mel values.
// returns the frame RMS (counts, DC removed) -- used as a loudness gate.
inline float logmel(const int16_t* frame, float* out) {
  float mean = 0;
  for (int n = 0; n < KWS_WIN; n++) mean += frame[n];
  mean /= KWS_WIN;
  float ss = 0;
  for (int n = 0; n < KWS_WIN; n++) {
    float v = frame[n] - mean;
    ss += v * v;
    re[n] = v * (1.0f / 2048.0f) * kHann[n];
    im[n] = 0;
  }
  for (int n = KWS_WIN; n < KWS_NFFT; n++) { re[n] = 0; im[n] = 0; }
  fft();
  for (int k = 0; k <= KWS_NFFT / 2; k++) pw[k] = re[k] * re[k] + im[k] * im[k];
  for (int m = 0; m < KWS_NMEL; m++) {
    float e = 0;
    for (int k = 0; k <= KWS_NFFT / 2; k++) e += kMel[m][k] * pw[k];
    out[m] = (logf(e + 1e-6f) - kNormMean) / kNormStd;
  }
  return sqrtf(ss / KWS_WIN);
}
}  // namespace dsp
