"""Live speech path with NO external mic:
   laptop built-in mic -> (this script: gain + 6.8 kHz anti-alias filter) -> headphone jack -> ESP32.

  python mic_to_jack.py --list
  python mic_to_jack.py --gain 2
"""
import argparse, time
import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt

ap = argparse.ArgumentParser()
ap.add_argument("--list", action="store_true"); ap.add_argument("--gain", type=float, default=1.0)
ap.add_argument("--in-dev", type=int); ap.add_argument("--out-dev", type=int)
ap.add_argument("--fs", type=int, default=0, help="sample rate (0 = output device default)")
a = ap.parse_args()
if a.list:
    print(sd.query_devices()); raise SystemExit

out_info = sd.query_devices(a.out_dev, "output")
fs = a.fs or int(out_info["default_samplerate"])
sos = butter(8, 6800 / (fs / 2), output="sos")        # the ESP32 samples at 16 kHz -> remove > 8 kHz first
zi = np.zeros((sos.shape[0], 2)); peak = 0.0


def cb(indata, outdata, frames, t, status):
    global zi, peak
    y, zi = sosfilt(sos, indata[:, 0] * a.gain, zi=zi)
    y = np.clip(y, -0.98, 0.98)
    peak = max(peak, float(np.abs(y).max()))
    outdata[:, 0] = y; outdata[:, 1] = y


print(f"fs={fs}  in={sd.query_devices(a.in_dev, 'input')['name']}  out={out_info['name']}")
print("Streaming mic -> headphone jack.  Ctrl+C to stop.  (If out-level never moves, raise --gain)")
with sd.Stream(samplerate=fs, channels=(1, 2), dtype="float32", callback=cb,
               device=(a.in_dev, a.out_dev), blocksize=1024):
    try:
        while True:
            time.sleep(0.4); print(f"\rout level {'#' * int(peak * 40):<40} {peak:4.2f}", end="", flush=True); peak = 0.0
    except KeyboardInterrupt:
        pass
